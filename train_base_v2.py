import argparse
import os
import sys
from datetime import datetime

import torch.nn as nn
import torch
from torch.utils.data import DataLoader

# tensorboard
from torch.utils.tensorboard import SummaryWriter

from models.mvssnet import get_mvss
from datasets.dataset import *
from tqdm import tqdm

# for dice loss
def dice_loss(out, gt, smooth = 1.0):
    gt = gt.view(-1)
    out = out.view(-1)

    intersection = (gt * out).sum()
    dice = (2.0 * intersection + smooth) / (torch.square(gt).sum() + torch.square(out).sum() + smooth) # TODO: need to confirm this matches what the paper says, and also the calculation/result is correct

    return 1.0 - dice

# for multiprocessing
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

# for removing damaged images
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def parse_args():
    parser = argparse.ArgumentParser()

    ## job
    parser.add_argument("--id", type=int, help="unique ID from Slurm")
    parser.add_argument("--run_name", type=str, default="MVSS-Net", help="run name")

    ## multiprocessing
    parser.add_argument('--dist_backend', default='nccl', choices=['gloo', 'nccl'], help='multiprocessing backend')
    parser.add_argument('--master_addr', type=str, default="localhost", help='address')
    parser.add_argument('--master_port', type=int, default=3721, help='address')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')

    ## dataset
    parser.add_argument("--paths_file", type=str, default="./files.txt", help="path to the file with input paths") # each line of this file should contain "/path/to/image.ext /path/to/mask.ext /path/to/edge.ext 1 (for fake)/0 (for real)"; for real image.ext, set /path/to/mask.ext and /path/to/edge.ext as a string None
    parser.add_argument("--val_paths_file", type=str, help="path to the validation set")
    parser.add_argument("--n_c_samples", type=int, help="samples per classes (None for non-controlled)")
    parser.add_argument("--val_n_c_samples", type=int, help="samples per classes for validation set (None for non-controlled)")

    parser.add_argument("--workers", type=int, default=0, help="number of cpu threads to use during batch generation")

    parser.add_argument("--image_size", type=int, default=512, help="size of the images")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    
    parser.add_argument("--batch_size", type=int, default=12, help="size of the batches") # no default value given by paper

    ## model
    parser.add_argument('--load_path', type=str, help='pretrained model or checkpoint for continued training')

    ## optimizer and scheduler
    parser.add_argument("--optim", choices=['adam', 'sgd'], default='adam', help="optimizer")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd: momentum of gradient")

    parser.add_argument('--patience', type=int, default=5, help='numbers of epochs to decay for ReduceLROnPlateau scheduler (None to disable)')

    parser.add_argument('--decay_epoch', type=int, help='numbers of epochs to decay for StepLR scheduler (low priority, None to disable)')

    ## training
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")

    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")

    parser.add_argument("--cond_epoch", type=int, default=0, help="epoch to start training from")
    
    parser.add_argument("--n_early", type=int, default=10, help="number of epochs for early stopping")

    ## losses
    parser.add_argument("--lambda_seg", type=float, default=0.16, help="pixel-scale loss weight (alpha)")
    parser.add_argument("--lambda_clf", type=float, default=0.04, help="image-scale loss weight (beta)")

    ## log
    parser.add_argument("--log_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="batch interval between model checkpoints")
    
    args = parser.parse_args()

    return args

def init_env(args, local_rank, global_rank):
    # for debug only
    torch.autograd.set_detect_anomaly(True)

    if (args.id is None):
        args.id = datetime.now().strftime("%Y%m%d%H%M%S")

    torch.cuda.set_device(local_rank)
    setup_for_distributed(global_rank == 0)

    # finalizing args, print here
    print(args)

    return args

def init_models(args):
    model = get_mvss(backbone='resnet50',
                         pretrained_base=True,
                         nclass=1,
                         sobel=True,
                         constrain=True,
                         n_input=args.channels,
                         ).cuda()

    return model

def init_dataset(args, global_rank, world_size, val = False):
    # return None if no validation set provided
    if (val and args.val_paths_file is None):
        print('No val set!')
        return None, None
    
    dataset = DeepfakeDataset((args.paths_file if not val else args.val_paths_file),
                              args.image_size,
                              args.id,
                              (args.n_c_samples if not val else args.val_n_c_samples),
                              val)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    
    local_batch_size = args.batch_size // world_size
    
    if (not val):
        print('Local batch size is {} ({}//{})!'.format(local_batch_size, args.batch_size, world_size))

    dataloader = DataLoader(dataset=dataset, batch_size=local_batch_size,num_workers=args.workers, pin_memory=True, drop_last=True, sampler=sampler, collate_fn=collate_fn)

    print('{} set size is {}!'.format(('Train' if not val else 'Val'), len(dataloader) * args.batch_size))

    return sampler, dataloader

def init_optims(args, world_size,
                model):
    
    # Optimizers
    local_lr = args.lr / world_size

    print('Local learning rate is {} ({}/{})!'.format(local_lr, args.lr, world_size))

    if (args.optim == 'adam'):
        print("Using optimizer adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=local_lr, betas=(args.b1, args.b2))
    elif (args.optim == 'sgd'):
        print("Using optimizer sgd")
        optimizer = torch.optim.SGD(model.parameters(), lr=local_lr, momentum=args.momentum)
    else:
        print("Unrecognized optimizer %s" % args.optim)
        sys.exit()

    return optimizer

def init_schedulers(args, optimizer):
    lr_scheduler = None

    # high priority for ReduceLROnPlateau (validation set required)
    if (args.val_paths_file and args.patience):
        print("Using scheduler ReduceLROnPlateau")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, 
                                                                  factor = 0.1,
                                                                  patience = args.patience)
    # low priority StepLR
    elif (args.decay_epoch):
        print("Using scheduler StepLR")
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,
                                                    step_size = args.decay_epoch,
                                                    gamma = 0.5)
    
    else:
        print("No scheduler used")
    
    return lr_scheduler

def load_dicts(args, get_module,
                model):
    # Load pretrained models
    if args.load_path != None and args.load_path != 'timm':
        print('Load pretrained model: {}'.format(args.load_path))

        if (not get_module):
            model.load_state_dict(torch.load(args.load_path))
        else:
            model.module.load_state_dict(torch.load(args.load_path))

    return model

# for saving checkpoints
def save_checkpoints(checkpoint_dir, id, epoch, step, get_module,
                    model):
    if (get_module):
        net = model.module
    else:
        net = model

    torch.save(net.state_dict(),
                os.path.join(checkpoint_dir, 'best.pth'))

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """denormalize image with mean and std
    """
    image = image.clone().detach().cpu()
    image = image * torch.tensor(std).view(3, 1, 1)
    image = image + torch.tensor(mean).view(3, 1, 1)
    return image


# a single step of prediction and loss calculation (same for both training and validating)
def predict_loss(args, data, model,
                 criterion_BCE,
                 gmp):
    # load data
    in_imgs, in_masks, in_edges, in_labels = data

    in_imgs = in_imgs.to('cuda', non_blocking=True)
    in_masks = in_masks.to('cuda', non_blocking=True)
    in_edges = in_edges.to('cuda', non_blocking=True)
    in_labels = in_labels.to('cuda', non_blocking=True).float()
    
    # predict
    #out_edges:经过ERB提取后的边缘特征输出
    #out_masks:经过Dual Attention后的分割图样
    out_edges, out_masks = model(in_imgs)
    out_edges = torch.sigmoid(out_edges)
    out_masks = torch.sigmoid(out_masks)

    # Pixel-scale loss
    loss_seg = dice_loss(out_masks, in_masks)

    # Edge loss
    loss_edg = dice_loss(out_edges, in_edges)

    # Image-scale loss (with GMP)
    out_labels = gmp(out_masks).squeeze()
    loss_clf = criterion_BCE(out_labels, in_labels)

    # Total loss
    alpha = args.lambda_seg
    beta = args.lambda_clf

    weighted_loss_seg = alpha * loss_seg
    weighted_loss_clf = beta * loss_clf
    weighted_loss_edg = (1.0 - alpha - beta) * loss_edg

    loss = weighted_loss_seg + weighted_loss_clf + weighted_loss_edg

    return loss, weighted_loss_seg, weighted_loss_clf, weighted_loss_edg, in_imgs, in_masks, in_edges, out_masks, out_edges


def train(args, global_rank, sync, get_module,
            model,
            train_sampler, dataloader, val_sampler, val_dataloader,
            optimizer,
            lr_scheduler):
    # Losses that are built-in in PyTorch
    criterion_BCE = nn.BCEWithLogitsLoss().cuda()
    # tensorboard
    if global_rank == 0:
        os.makedirs("logs", exist_ok=True)
        writer = SummaryWriter("logs/" + str(args.id) + "_" + args.run_name)
        checkpoint_dir = "checkpoints/" + str(args.id) + "_" + args.run_name
        os.makedirs(checkpoint_dir, exist_ok=True)

    # for early stopping
    best_val_loss = float('inf')
    n_last_epochs = 0
    early_stopping = False

    # GMP layer
    gmp = nn.MaxPool2d(args.image_size)

    for epoch in range(args.cond_epoch, args.n_epochs):

        train_sampler.set_epoch(epoch)

        print('Starting Epoch {}'.format(epoch))

        # loss sum for epoch
        epoch_total_seg = 0
        epoch_total_clf = 0
        epoch_total_edg = 0

        epoch_total_model = 0

        epoch_val_loss = 0

        # number of steps in one epoch 
        # can be replaced by len(dataloader), but kept as warm-up epochs may be added
        epoch_steps = 0

        # ------------------
        #  Train step
        # ------------------
        for step, data in tqdm(enumerate(dataloader),desc=f"Epoch {epoch+1}:",leave=True,total=len(dataloader)):
            curr_steps = epoch * len(dataloader) + step

            model.train()

            if (sync): optimizer.synchronize()
            optimizer.zero_grad()

            loss, weighted_loss_seg, weighted_loss_clf, weighted_loss_edg, in_imgs, in_masks, in_edges, out_masks, out_edges = predict_loss(args, data, model, criterion_BCE, gmp)

            # backward prop
            loss.backward()
            optimizer.step()

            # log losses for epoch
            epoch_steps += 1

            epoch_total_seg += weighted_loss_seg.item()
            epoch_total_clf += weighted_loss_clf.item()
            epoch_total_edg += weighted_loss_edg.item()
            epoch_total_model += loss.item()
            
            # --------------
            #  Log Progress (for certain steps)
            # --------------
            if step != 0 and step % args.log_interval == 0 and global_rank == 0:
                print(f"[Epoch {epoch}/{args.n_epochs - 1}] [Batch {step}/{len(dataloader)}] "
                    f"[Total Loss {loss:.3f}]"
                    f"[Pixel-scale Loss {weighted_loss_seg:.3e}]"
                    f"[Edge Loss {weighted_loss_edg:.3e}]"
                    f"[Image-scale Loss {weighted_loss_clf:.3e}]"
                    f"")

                writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], curr_steps)
                writer.add_scalar("Loss/Total Loss", loss, epoch * len(dataloader) + step)
                writer.add_scalar("Loss/Pixel-scale", weighted_loss_seg, curr_steps)
                writer.add_scalar("Loss/Edge", weighted_loss_edg, curr_steps)
                writer.add_scalar("Loss/Image-scale", weighted_loss_clf, curr_steps)
                in_imgs = denormalize(in_imgs)
                writer.add_images('Input Img', in_imgs, epoch * len(dataloader) + step)
                in_masks = in_masks.unsqueeze(1)
                writer.add_images('Input Mask', in_masks, epoch * len(dataloader) + step)
                writer.add_images('Output Mask', out_masks, epoch * len(dataloader) + step)
                writer.add_images('Input Edge', in_edges, epoch * len(dataloader) + step)
                writer.add_images('Output Edge', out_edges, epoch * len(dataloader) + step)

            # save model parameters
            '''
            if step != 0 and step % args.checkpoint_interval == 0 and global_rank == 0:
                save_checkpoints(checkpoint_dir, args.id, epoch, step, get_module,
                                 model)
            '''
        # ------------------
        #  Validation
        # ------------------
        print("start validation")
        if (args.val_paths_file and val_sampler and val_dataloader):
  
            val_sampler.set_epoch(epoch)

            model.eval()

            for step, data in tqdm(enumerate(val_dataloader),desc=f"Epoch {epoch+1}:",leave=True,total=len(val_dataloader)):
                with torch.no_grad():
                    loss, _, _, _, _, _, _, _, _ = predict_loss(args, data, model, criterion_BCE, gmp)

                    epoch_val_loss += loss.item()

            # early 
            if(epoch_val_loss<=0.01):
                early_stopping=True

        # ------------------
        #  Step
        # ------------------
        if (lr_scheduler):
            if (args.val_paths_file and args.patience):
                lr_scheduler.step(epoch_val_loss) # ReduceLROnPlateau
            elif (args.decay_epoch):
                lr_scheduler.step() # StepLR
            else:
                print("Error in scheduler step")
                sys.exit()

        # --------------
        #  Log Progress (for epoch)
        # --------------
        # loss average for epoch
        if epoch_steps != 0 and global_rank == 0:
            epoch_avg_seg = epoch_total_seg / epoch_steps
            epoch_avg_edg = epoch_total_edg / epoch_steps
            epoch_avg_clf = epoch_total_clf / epoch_steps
            epoch_avg_model = epoch_total_model / epoch_steps

            if (args.val_paths_file):
                epoch_val_loss_avg = epoch_val_loss / len(val_dataloader)
                best_val_loss_avg = best_val_loss / len(val_dataloader)
            else:
                epoch_val_loss_avg = 0
                best_val_loss_avg = 0

            print(f"[Epoch {epoch}/{args.n_epochs - 1}]"
                    f"[Epoch Total Loss {epoch_avg_model:.3f}]"
                    f"[Epoch Pixel-scale Loss {epoch_avg_seg:.3e}]"
                    f"[Epoch Edge Loss {epoch_avg_edg:.3e}]"
                    f"[Epoch Image-scale Loss {epoch_avg_clf:.3e}]"
                    f"[Epoch Val Loss {epoch_val_loss_avg:.3f} (best Val Loss {best_val_loss_avg:.3f} last for {n_last_epochs:d})]"
                    f"")

            writer.add_scalar("Epoch LearningRate", optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar("Epoch Loss/Total Loss", epoch_avg_model, epoch)
            writer.add_scalar("Epoch Loss/Pixel-scale", epoch_avg_seg, epoch)
            writer.add_scalar("Epoch Loss/Edge", epoch_avg_edg, epoch)
            writer.add_scalar("Epoch Loss/Image-scale", epoch_avg_clf, epoch)
            writer.add_scalar("Epoch Loss/Val", epoch_val_loss_avg, epoch)
            if torch.max(in_imgs) > 1 or torch.min(in_imgs) < 0:
                in_imgs = denormalize(in_imgs)
            writer.add_images('Epoch Input Img', in_imgs, epoch)
            if len(in_masks.shape) == 3:
                in_masks = in_masks.unsqueeze(1)
            writer.add_images('Epoch Input Mask', in_masks, epoch)
            writer.add_images('Epoch Output Mask', out_masks, epoch)
            writer.add_images('Epoch Input Edge', in_edges, epoch)
            writer.add_images('Epoch Output Edge', out_edges, epoch)

            # save model parameters
            if global_rank == 0:
                if epoch_val_loss_avg+epoch_avg_model <= best_val_loss:
                    best_val_loss =  epoch_val_loss_avg+epoch_avg_model
                    save_checkpoints(checkpoint_dir, args.id, epoch, 'end', # set step to a string 'end'
                                 get_module,
                                 model)

        # check early_stopping
        if (early_stopping):
            print('Early stopping')
            break

    print('Finished training')

    if global_rank == 0:
        writer.close()

    pass