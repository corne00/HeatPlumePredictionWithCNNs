import os
import numpy as np

from .dataloaders import DatasetMultipleSubdomains
from torch.utils.data import DataLoader

def init_data(data_settings, image_dir, mask_dir, num_samples=None, crop_size = (None, None)):

    image_labels = os.listdir(image_dir)
    if num_samples is not None:
        image_labels = image_labels[:num_samples]
    split=(np.array([0.8,0.19,0.1]) * len(image_labels)).astype(int)
    train_dataset = DatasetMultipleSubdomains(image_labels=image_labels[:split[0]], image_dir=image_dir, mask_dir=mask_dir, transform=None,
                                        target_transform=None, data_augmentation=None, subdomains_dist=data_settings["subdomains_dist"], patch_size=640, crop_size=crop_size)

    val_dataset = DatasetMultipleSubdomains(image_labels=image_labels[split[0]:split[0]+split[1]], image_dir=image_dir, mask_dir=mask_dir, transform=None,
                                        target_transform=None, data_augmentation=None, subdomains_dist=data_settings["subdomains_dist"], patch_size=640, crop_size=crop_size)

    test_dataset = DatasetMultipleSubdomains(image_labels=image_labels[split[0]+split[1]:], image_dir=image_dir, mask_dir=mask_dir, transform=None,
                                        target_transform=None, data_augmentation=None, subdomains_dist=data_settings["subdomains_dist"], patch_size=640, crop_size=crop_size)

    # Define dataloaders
    dataloader_train = DataLoader(train_dataset, batch_size=data_settings["batch_size_training"], shuffle=True) 
    dataloader_val = DataLoader(val_dataset, batch_size=data_settings["batch_size_testing"], shuffle=False)
    dataloader_test = DataLoader(test_dataset, batch_size=data_settings["batch_size_testing"], shuffle=False)

    return {"train": dataloader_train, "val": dataloader_val, "test": dataloader_test}
