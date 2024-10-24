import os
import random

import torch
from torch.utils.data import Dataset

class DatasetMultipleSubdomains(Dataset):
    def __init__(self, image_labels, image_dir, mask_dir, transform=None, target_transform=None, 
                 data_augmentation=None, patch_size = 640, subdomains_dist=(1,1)):
        """
        Args:
        - image_labels (list): list of file names of the labels (ground truth and inputs) that should be included in the dataset
        - image_dir (str)    : directory containing the inputs
        - mask_dir (str)     : directory containing the ground truth labels
        - transform (func)   : transformation that should be applied to the input data
        - target_transform   : transformation that should be applied to the target data
        - data_augmentation  : data augmentation that should be applied to the input and output labels
        - patch_size         : pathc size to be used for training: if smaller than image size, patches with this size will be generated!
        - subdomain dist (int, int): splitting (nx, ny) of the input into subdomains, nx in x-direction and ny in y-direction

        """

        self.img_labels = image_labels
        self.img_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation
        self.subdomains_dist = subdomains_dist
        self.patch_size = patch_size
        self.half_precision : bool = True

    def __len__(self):
        return len(self.img_labels)
    
    def __split_image(self, full_image):
        """
        Splits the image into subdomains (if subdomain_dist != (1,1), otherwise only one (sub)domain is used)
        """
        subdomain_tensors = []
        subdomain_height = full_image.shape[2] // self.subdomains_dist[0]
        subdomain_width = full_image.shape[1] // self.subdomains_dist[1]

        for i in range(self.subdomains_dist[0]):
            for j in range(self.subdomains_dist[1]):
                subdomain = full_image[:, j * subdomain_height: (j + 1) * subdomain_height,
                            i * subdomain_width: (i + 1) * subdomain_width]
                subdomain_tensors.append(subdomain)

        return subdomain_tensors        
    
    def __crop_patch(self, full_image, full_mask):
        """
        Crops a patch from the global image (if the patch size is smaller than the actual image size)
        """
        _, height, width = full_image.shape
        patch_height, patch_width = self.patch_size, self.patch_size

        if height < patch_height or width < patch_width:
            raise ValueError("Patch size must be smaller than image size.")
        
        top = random.randint(0, height - patch_height)
        left = random.randint(0, width - patch_width)
        
        image_patch = full_image[:, top:top + patch_height, left:left + patch_width]
        mask_patch = full_mask[:, top:top + patch_height, left:left + patch_width]

        return image_patch, mask_patch

    def __getitem__(self, idx):
        img_name = self.img_labels[idx]
        
        img_path =  os.path.join(self.img_dir, f"{img_name}")                
        mask_path =  os.path.join(self.mask_dir, f"{img_name}")

        image = torch.load(img_path)
        mask = torch.load(mask_path)

        images = []

        image, mask = self.__crop_patch(image, mask)

        if self.data_augmentation:
            image, mask = self.data_augmentation(image, mask)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
            
        images = self.__split_image(image)      

        if self.half_precision:
            images = [image.half() for image in images]
            mask = mask.half()


        return images, mask