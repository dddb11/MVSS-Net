import os

import torch.nn as nn

# multiprocessing
import torch.distributed as dist

from train_base import *

# constants
SYNC = False
GET_MODULE = True

def main():
    args = parse_args()

    # Init dist
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    args = init_env(args, local_rank, global_rank)

    # Init the process group
    print('Initializing Process Group...')
    dist.init_process_group(backend=args.dist_backend, init_method=("env://%s:%s" % (args.master_addr, args.master_port)),
        world_size=world_size, rank=global_rank)
    print('Process group ready!')

    model = init_models(args)
    
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    train_sampler, dataloader = init_dataset(args, global_rank, world_size)
    val_sampler, val_dataloader = init_dataset(args, global_rank, world_size, True)

    model = load_dicts(args, GET_MODULE, model)

    optimizer = init_optims(args, world_size, model)

    lr_scheduler = init_schedulers(args, optimizer)

    train(args, global_rank, SYNC, GET_MODULE,
            model,
            train_sampler, dataloader, val_sampler, val_dataloader,
            optimizer,
            lr_scheduler)

if __name__ == '__main__':
    main()