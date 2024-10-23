import os
import numpy as np

from .dataloaders import DatasetMultipleSubdomains
from torch.utils.data import DataLoader

class Args:
    def __init__(self, subdomains_dist=(1, 1), batch_size_training=16, batch_size_testing=16):
        self.subdomains_dist = subdomains_dist
        self.batch_size_training = batch_size_training
        self.batch_size_testing = batch_size_testing

def init_data(args, image_dir, mask_dir):

        if args is None:
                print("Warning: 'args' is None. Using default values for subdomain_dist (1,1) and batch_sizes (16).")
                args = Args()  # Create a new Args object with default values

        image_labels = os.listdir(image_dir)
        split=(np.array([0.8,0.19,0.1]) * len(image_labels)).astype(int)
        train_dataset = DatasetMultipleSubdomains(image_labels=image_labels[:split[0]], image_dir=image_dir, mask_dir=mask_dir, transform=None,
                                            target_transform=None, data_augmentation=None, subdomains_dist=args.subdomains_dist, patch_size=640)

        val_dataset = DatasetMultipleSubdomains(image_labels=image_labels[split[0]:split[0]+split[1]], image_dir=image_dir, mask_dir=mask_dir, transform=None,
                                            target_transform=None, data_augmentation=None, subdomains_dist=args.subdomains_dist, patch_size=640)

        test_dataset = DatasetMultipleSubdomains(image_labels=image_labels[split[0]+split[1]:], image_dir=image_dir, mask_dir=mask_dir, transform=None,
                                            target_transform=None, data_augmentation=None, subdomains_dist=args.subdomains_dist, patch_size=640)

        # Define dataloaders
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size_training, shuffle=True) 
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size_testing, shuffle=False)
        dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size_testing, shuffle=False)

        return {"train": dataloader_train, "val": dataloader_val, "test": dataloader_test}, {"train": train_dataset, "val": val_dataset, "test": test_dataset}
