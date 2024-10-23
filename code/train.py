import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler

from models import *
from utils import parse_args, save_args_to_json, plot_results
from utils.losses import WeightedMSELoss, ThresholdedMSELoss
from dataprocessing import DatasetMultipleSubdomains
from utils.train_utils import *
from utils.visualization import *

# Load and save the arguments from the arge parser
args = parse_args()
try:
    os.makedirs(args.save_path)
except:
    print("Results directory already exists!")
    exit()

save_args_to_json(args=args, filename=os.path.join(args.save_path, "args.json"))

# Check if we have half precision
half_precision = torch.cuda.is_available()
data_type = torch.float16 if half_precision else torch.float32

# Half precision scaler
scaler = GradScaler(enabled=half_precision)

# Set datasets
# image_dir = "/scratch/e451412/data/test_data/Inputs"
# mask_dir = "/scratch/e451412/data/test_data/Labels"

# train_dataset = DatasetMultipleSubdomains(image_labels=[f"RUN_{i}.pt" for i in range(3)], image_dir=image_dir, mask_dir=mask_dir, transform=None,
#                                     target_transform=None, data_augmentation=None, subdomains_dist=args.subdomains_dist, patch_size=640)

# val_dataset = DatasetMultipleSubdomains(image_labels=[f"RUN_{i}.pt" for i in range(3)], image_dir=image_dir, mask_dir=mask_dir, transform=None,
#                                     target_transform=None, data_augmentation=None, subdomains_dist=args.subdomains_dist, patch_size=640)

# test_dataset = DatasetMultipleSubdomains(image_labels=[f"RUN_{i}.pt" for i in range(3)], image_dir=image_dir, mask_dir=mask_dir, transform=None,
#                                     target_transform=None, data_augmentation=None, subdomains_dist=args.subdomains_dist, patch_size=640)

# Set datasets
image_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_1000dp inputs_pki outputs_t/Inputs"
mask_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_1000dp inputs_pki outputs_t/Labels"

from hyperparam_optuna import init_data

# args.batch_size_training = trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32])
_, datasets = init_data(args, image_dir, mask_dir)
train_dataset, val_dataset, test_dataset = datasets['train'], datasets['val'], datasets['test']

# Define dataloaders
dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size_training, shuffle=True) 
dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size_testing, shuffle=False)
dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size_testing, shuffle=False)

# Set devices
devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
print("Available GPUs:", devices)

unet, val_losses, training_losses = train_parallel_model(model=MultiGPU_UNet_with_comm, dataloader_val=dataloader_val, dataloader_train=dataloader_train,
                                                         scaler=scaler, data_type=data_type, half_precision=True, train_dataset=train_dataset, val_dataset=val_dataset,
                                                         comm=args.comm, num_epochs=args.num_epochs, num_comm_fmaps=args.num_comm_fmaps, 
                                                         save_path=args.save_path, subdomains_dist=args.subdomains_dist, exchange_fmaps=args.exchange_fmaps,
                                                         padding=args.padding, depth=args.depth, kernel_size=args.kernel_size, communication_network=None,
                                                         complexity=args.complexity, dropout_rate=0.0, devices=devices, num_convs=args.num_convs,
                                                         weight_decay_adam=0, loss_fn_alpha=1, lr=0.0001, loss_func=ThresholdedMSELoss(), val_loss_func=ThresholdedMSELoss(), verbose=True)

plot_results(unet=unet, savepath=args.save_path, epoch_number="best", train_dataset=train_dataset, val_dataset=val_dataset)

# Save the losses
data = {
    "val_losses": val_losses,
    "training_losses": training_losses
}

# Save to a JSON file
with open(os.path.join(args.save_path, 'losses.json'), 'w') as json_file:
    json.dump(data, json_file)

# Call the function to compute average training loss and plot losses
avg_training_losses = average_training_losses(training_losses=training_losses, 
                                              dataloader_train=dataloader_train, 
                                              num_epochs=args.num_epochs)

plot_losses(avg_training_losses=avg_training_losses, val_losses=val_losses, save_path=args.save_path)
print("Finished!")


