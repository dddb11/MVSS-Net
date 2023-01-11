import random
import numpy as np
import cv2

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

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
        self.transform_train = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(),
            A.VerticalFlip(),
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
            input = self.transform_train(image = input)['image']
        else:
            input = self.transform_val(image = input)['image']
        input = input / 255.0

        random.seed(seed)
        if (not self.val):
            mask = self.transform_train(image = mask)['image']
        else:
            mask = self.transform_val(image = mask)['image']
        mask = mask / 255.0

        random.seed(seed)
        if (not self.val):
            edge = self.transform_train_edge(image = edge)['image']
        else:
            edge = self.transform_val_edge(image = edge)['image']
        edge = edge / 255.0

        return input, mask, edge, self.labels[item]

    def __len__(self):
        return len(self.input_image_paths)
