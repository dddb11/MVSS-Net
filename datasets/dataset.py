import random
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
from albumentations.core.transforms_interface import DualTransform
from typing import Callable, Dict
from albumentations.augmentations.geometric import Resize, RandomRotate90, HorizontalFlip, VerticalFlip
from albumentations.augmentations.geometric.functional import resize as fresize
from albumentations.augmentations.geometric.functional import hflip, hflip_cv2, vflip


class Transform_with_edge(DualTransform):
    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "edge": self.apply_to_edge,
        }

    def apply_to_edge(self, img: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError


class Resize_with_Edge(Transform_with_edge, Resize):
    def apply_to_edge(self, img: np.ndarray, interpolation=cv2.INTER_LINEAR, **params) -> np.ndarray:
        return fresize(img, height=self.height//4, width=self.width//4, interpolation=interpolation)


class Rotate_with_Edge(Transform_with_edge, RandomRotate90):
    def apply_to_edge(self, img: np.ndarray, factor=0, **params) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(img, factor))


class HorizontalFlip_with_Edge(Transform_with_edge, HorizontalFlip):
    def apply_to_edge(self, img: np.ndarray, **params) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return hflip_cv2(img)
        return hflip(img)

class VerticalFlip_with_Edge(Transform_with_edge, VerticalFlip):
    def apply_to_edge(self, img: np.ndarray, **params) -> np.ndarray:
        return vflip(img)


class Edge_generator(torch.nn.Module):
    """generate the 'edge bar' for a 0-1 mask Groundtruth of a image
    Algorithm is based on 'Morphological Dilation and Difference Reduction'
    
    Which implemented with fixed-weight Convolution layer with weight matrix looks like a cross,
    for example, if kernel size is 3, the weight matrix is:
        [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]

    """
    def __init__(self, kernel_size = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
    
    def _dilate(self, image, kernel_size=3):
        """Doings dilation on the image

        Args:
            image (_type_): 0-1 tensor in shape (B, C, H, W)
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert image.shape[2] > kernel_size and image.shape[3] > kernel_size, "Image must be larger than kernel size"
        
        kernel = torch.zeros((1, 1, kernel_size, kernel_size))
        kernel[0, 0, kernel_size // 2: kernel_size//2+1, :] = 1
        kernel[0, 0, :,  kernel_size // 2: kernel_size//2+1] = 1
        kernel = kernel.float()
        # print(kernel)
        res = F.conv2d(image, kernel.view([1,1,kernel_size, kernel_size]),stride=1, padding = kernel_size // 2)
        return (res > 0) * 1.0


    def _find_edge(self, image, kernel_size=3, return_all=False):
        """Find 0-1 edges of the image

        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        image = image.clone().float()
        shape = image.shape
        
        if len(shape) == 2:
            image = image.reshape([1, 1, shape[0], shape[1]])
        if len(shape) == 3:
            image = image.reshape([1, shape[0], shape[1], shape[2]])   
        assert image.shape[1] == 1, "Image must be single channel"
        
        img = self._dilate(image, kernel_size=kernel_size)
        
        erosion = self._dilate(1-image, kernel_size=kernel_size)

        diff = -torch.abs(erosion - img) + 1
        diff = (diff > 0) * 1.0
        # res = dilate(diff)
        diff = diff.numpy()
        if return_all :
            return diff, img, erosion
        else:
            return diff
    
    def forward(self, x, return_all=False):
        """
        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        return self._find_edge(x, self.kernel_size, return_all=return_all)

class RandomCopyMove(Transform_with_edge):
    def __init__(self,
        max_h = 256,
        max_w = 256,
        min_h = 50,
        min_w = 50,
        mask_value = 255,
        always_apply = False,
        p = 0.5,  
    ):
        super(RandomCopyMove, self).__init__(always_apply, p)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value
        self.edge_generator = Edge_generator(kernel_size=15)
        
    def _get_random_window(
        self, 
        img_height, 
        img_width, 
        window_height = None, 
        window_width = None
    ):
        assert self.max_h < img_height , f"Image height should larger than max_h, but get max_h:{self.max_h} img_height:{img_height}!"
         
        assert self.max_w < img_width , f"Image height should larger than max_h, but get max_h:{self.max_w} img_height:{img_width}!"
        
        if window_width == None or window_height == None:
            window_h = np.random.randint(self.min_h, self.max_h)
            window_w = np.random.randint(self.min_w, self.max_w)
        else:
            window_h = window_height
            window_w = window_width

        # position of left up corner of the window
        pos_h = np.random.randint(0, img_height - window_h)
        pos_w = np.random.randint(0, img_width - window_w)
        
        return pos_h, pos_w , window_h, window_w
        
        
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        image = img.copy()
        H, W, _ = image.shape
        # copy region:
        c_pos_h, c_pos_w, c_window_h, c_window_w = self._get_random_window(H, W)
        
        # past region, window size is defined by copy region:
        self.p_pos_h, self.p_pos_w, self.p_window_h, self.p_window_w = self._get_random_window(H, W, c_window_h, c_window_w)
          
        copy_region = image[
            c_pos_h: c_pos_h + c_window_h, 
            c_pos_w: c_pos_w + c_window_w, 
            : 
        ]
        image[
            self.p_pos_h : self.p_pos_h + self.p_window_h,
            self.p_pos_w : self.p_pos_w + self.p_window_w,
            :
        ] = copy_region
        return image
        

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """
        change the mask of manipulated region to 1
        """
    
        manipulated_region =  np.full((self.p_window_h, self.p_window_w), 1)
        img = img.copy()
        img[
            self.p_pos_h : self.p_pos_h + self.p_window_h,
            self.p_pos_w : self.p_pos_w + self.p_window_w,
        ] = self.mask_value
        return img

    def apply_to_edge(self, img: np.ndarray, **params) -> np.ndarray:
        mask = img.copy()
        return self.edge_generator((mask > 0.5) * 1.0)[0][0]
        
class RandomInpainting(Transform_with_edge):
    def __init__(self,
        max_h = 256,
        max_w = 256,
        min_h = 50,
        min_w = 50,
        mask_value = 255,
        always_apply = False,
        p = 0.5,  
    ):
        super(RandomInpainting, self).__init__(always_apply, p)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value
        self.edge_generator = Edge_generator(kernel_size=15)
        
    def _get_random_window(
        self, 
        img_height, 
        img_width, 
    ):
        assert self.max_h < img_height , f"Image height should larger than max_h, but get max_h:{self.max_h} img_height:{img_height}!"
         
        assert self.max_w < img_width , f"Image height should larger than max_h, but get max_h:{self.max_w} img_height:{img_width}!"
        
        window_h = np.random.randint(self.min_h, self.max_h)
        window_w = np.random.randint(self.min_w, self.max_w)

        # position of left up corner of the window
        pos_h = np.random.randint(0, img_height - window_h)
        pos_w = np.random.randint(0, img_width - window_w)
        
        return pos_h, pos_w , window_h, window_w
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.copy()
        img = np.uint8(img)
        H, W, C = img.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        # inpainting region
        self.pos_h, self.pos_w , self.window_h, self.window_w = self._get_random_window(H, W)
        mask[
            self.pos_h : self.pos_h+ self.window_h,
            self.pos_w : self.pos_w + self.window_w,
        ] = 1
        inpaint_flag = cv2.INPAINT_TELEA if random.random() > 0.5 else cv2.INPAINT_NS
        img = cv2.inpaint(img, mask, 3,inpaint_flag)
        return img
    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """
        change the mask of manipulated region to 1
        """
        img = img.copy()
        img[
            self.pos_h : self.pos_h+ self.window_h,
            self.pos_w : self.pos_w + self.window_w,
        ] = self.mask_value
        return img

    def apply_to_edge(self, img: np.ndarray, **params) -> np.ndarray:
        mask = img.copy()
        return self.edge_generator((mask > 0.5) * 1.0)[0][0]


class DeepfakeDataset(Dataset):
    def sampling(self, distribution, n_max):
        if self.n_c_samples is None:
            self.n_c_samples = n_max

        for label_str in distribution:
            list = distribution[label_str]
            n_list = len(list)
            if (n_list >= self.n_c_samples):
                # undersampling
                picked = random.sample(list, self.n_c_samples)
            else:
                # oversampling
                for _ in range(self.n_c_samples // n_list):
                    for i in list:
                        (input_image_path, mask_image_path,edge_image_path) = i
                        self.input_image_paths.append(input_image_path)
                        self.mask_image_paths.append(mask_image_path)
                        self.edge_image_paths.append(edge_image_path)
                        self.labels.append(int(label_str))

                picked = random.sample(list, self.n_c_samples % n_list)

            # for picked
            for p in picked:
                (input_image_path, mask_image_path,edge_image_path) = p
                self.input_image_paths.append(input_image_path)
                self.mask_image_paths.append(mask_image_path)
                self.edge_image_paths.append(edge_image_path)
                self.labels.append(int(label_str))

        return

    def __init__(self, paths_file, image_size, id, n_c_samples = None, val = False):
        self.image_size = image_size

        self.n_c_samples = n_c_samples

        self.val = val

        self.input_image_paths = []
        self.mask_image_paths = []
        self.edge_image_paths = []
        self.labels = []
        self.edge_generator = Edge_generator(kernel_size=15)
        
        if ('cond' not in paths_file):
            distribution = dict()
            n_max = 0

            with open(paths_file, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    parts = l.rstrip().split(' ')
                    input_image_path = parts[0]
                    mask_image_path = parts[1]
                    edge_image_path = parts[2]
                    label_str = parts[3]
                    if label_str == '1' and edge_image_path=='None':
                        self.edge_generator((mask>0.5) * 1.0)[0][0]


                    # add to distribution
                    if (label_str not in distribution):
                        distribution[label_str] = [(input_image_path, mask_image_path,edge_image_path)]
                    else:
                        distribution[label_str].append((input_image_path, mask_image_path,edge_image_path))

                    if (len(distribution[label_str]) > n_max):
                        n_max = len(distribution[label_str])

            self.sampling(distribution, n_max)

            # save final 
            save_path = 'cond_paths_file_' + str(id) + ('_train' if not val else '_val') + '.txt'
            with open(save_path, 'w') as f:
                for i in range(len(self.input_image_paths)):
                    f.write(self.input_image_paths[i] + ' ' + self.mask_image_paths[i] + ' ' + self.edge_image_paths[i] + ' ' + str(self.labels[i]) + '\n')

            print('Final paths file (%s) for %s saved to %s' % (('train' if not val else 'val'), str(id), save_path))

        else:
            print('Read from previous saved paths file %s' % (paths_file))

            with open(paths_file, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    parts = l.rstrip().split(' ')
                    self.input_image_paths.append(parts[0])
                    self.mask_image_paths.append(parts[1])
                    self.edge_image_paths.append(parts[2])
                    self.labels.append(int(parts[3]))

        # ----------
        #  TODO: Transforms for data augmentation (more augmentations should be added)
        # ----------
        self.triple_transforms = A.Compose([
            Resize_with_Edge(
                self.image_size,
                self.image_size
            ),
            Rotate_with_Edge(p=0.5),
            HorizontalFlip_with_Edge(),
            VerticalFlip_with_Edge(),
            RandomCopyMove(
                p=0.1
            ),
            RandomInpainting(
                p=0.1
            )
        ])
        self.transform_train = A.Compose([
            A.Resize(
                self.image_size,
                self.image_size
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=0.1,
                p=1
            ),
            # Rotate
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Blur(p=0.3),
            A.ImageCompression(
                quality_lower = 70,
                quality_upper = 100,
                p = 0.2
            ),
            RandomCopyMove(
                p = 0.1
            ),
            RandomInpainting(
                p = 0.1
            ),
            A.Normalize(),
            ToTensorV2()
        ])

        self.transform_train_edge = A.Compose([
            A.Resize(self.image_size // 4, self.image_size // 4), # specially for edge mask, as the paper uses H/4 and W/4
            A.HorizontalFlip(),
            A.VerticalFlip(),
            ToTensorV2()
        ])

        self.transform_val = A.Compose([
            A.Resize(self.image_size, self.image_size),
            ToTensorV2()
        ])

        self.transform_val_edge = A.Compose([
            A.Resize(self.image_size // 4, self.image_size // 4), # specially for edge mask, as the paper uses H/4 and W/4
            ToTensorV2()
        ])


    def __getitem__(self, item):
        # ----------
        # Read input image
        # ----------
        input_file_name = self.input_image_paths[item]
        input = cv2.cvtColor(cv2.imread(input_file_name), cv2.COLOR_BGR2RGB)

        height, width, _ = input.shape

        # ----------
        # Read mask
        # ----------
        mask_file_name = self.mask_image_paths[item]
        if (mask_file_name == "None"):
            mask = np.zeros((height, width), np.uint8) # a totally black mask for real image
        else:
            mask = cv2.imread(mask_file_name, cv2.IMREAD_GRAYSCALE)

        # ----------
        # Read edge
        # ----------
        edge_file_name = self.edge_image_paths[item]
        if (edge_file_name == "None"):
            edge = np.zeros((height, width), np.uint8) # a totally black edge mask for real image
        else:
            edge = cv2.imread(edge_file_name, cv2.IMREAD_GRAYSCALE)

        # ----------
        # Apply transform (the same for both image and mask)
        # ----------
        seed = np.random.randint(2147483647) # make a seed with numpy generator 

        random.seed(seed)
        if (not self.val):
            res = self.transform_train(image = input, mask = mask)
            input = res['image']
            mask = res['mask']
        else:
            input = self.transform_val(image = input)['image']

        random.seed(seed)
        if (not self.val):
            pass
        else:
            mask = self.transform_val(image = mask)['image']
        mask = mask / 255.0

        random.seed(seed)
        if (not self.val):
            edge = self.edge_generator((mask>0.5) * 1.0)[0][0]
            edge = self.transform_val_edge(image = edge)['image']
            
        else:
            edge = self.transform_val_edge(image = edge)['image']
            edge = edge / 255.0

        return input, mask, edge, self.labels[item]

    def __len__(self):
        return len(self.input_image_paths)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img = plt.imread("/root/workspace/MVSS/CASIAv1/TP/Sp/Sp_D_CND_A_pla0005_pla0023_0281.jpg")
    mask = cv2.imread("/root/workspace/MVSS/CASIAv1/Gt/Sp_D_CND_A_pla0005_pla0023_0281_gt.png", cv2.IMREAD_GRAYSCALE)
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.subplot(1, 4 ,2)
    plt.imshow(mask)
    trans = A.Compose([
            A.Resize(
                512,
                512
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=0.1,
                p=1
            ),
            # Rotate
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Blur(p=0.3),
            A.ImageCompression(
                quality_lower = 70,
                quality_upper = 100,
                p = 0.2
            ),
            RandomCopyMove(
                p = 0.1
            ),
            RandomInpainting(
                p = 0.1
            ),
            # A.Normalize(),
            # ToTensorV2()
        ])

    res = trans(image= img, mask = mask)
    plt.subplot(1, 4, 3)
    plt.imshow(res['image'])
    plt.subplot(1, 4, 4)
    plt.imshow(res['mask'])
    plt.savefig("view.png")