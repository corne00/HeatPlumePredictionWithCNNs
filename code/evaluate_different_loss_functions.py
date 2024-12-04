# Standard library imports
import os
import json
import pathlib
import random
import numpy as np
import argparse

import torch
from torch.cuda.amp import GradScaler
import optuna
from utils.losses_pixelwise import *

from models import *
from utils import parse_args, save_args_to_json, plot_results
from dataprocessing import init_data
from utils.train_utils import *
from utils.visualization import *

from utils.losses import *

NUM_CHANNELS = 5
DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
HALF_PRECISION = torch.cuda.is_available()
DATA_TYPE = torch.float16 if torch.cuda.is_available() else torch.float32
SCALER = GradScaler(enabled=HALF_PRECISION)

def load_unet(folder):

    with open(os.path.join(folder, 'args.json'), 'r') as json_file:
        args_dict = json.load(json_file)

    args = argparse.Namespace(**args_dict)

    model = MultiGPU_UNet_with_comm(n_channels=NUM_CHANNELS, n_classes=1, input_shape=(640, 640), num_comm_fmaps=args.num_comm_fmaps, devices=DEVICES, depth=args.depth,
                                    subdom_dist=args.subdomains_dist, bilinear=False, comm=args.comm, complexity=args.complexity, dropout_rate=0.0, 
                                    kernel_size=args.kernel_size, padding=args.padding, communicator_type=None, comm_network_but_no_communication=(not args.exchange_fmaps), 
                                    communication_network_def=None, num_convs=args.num_convs)
    
    try:
        model.load_weights(os.path.join(folder, 'unet.pth'), device=DEVICES[0])
    except Exception as e:
        print(rf"Loading UNet failed with exception {e}. Returning randomly initialized UNet now.")

    return model, args

def evaluate(unet, val_loss_func, dataloader):
    unet.eval()
    error_map = compute_validation_loss(unet, val_loss_func, dataloader, DEVICES[0], data_type=DATA_TYPE, half_precision=HALF_PRECISION, verbose=False)

    return error_map


def plot_different_loss_functions(unet, loss_functions, dataset, savepath=None, num_images=1):   
    unet.eval()
    
    def process_images(images):
        unet.eval() 
        with torch.no_grad():
            predictions = unet([img.unsqueeze(0) for img in images]).cpu()
        full_images = unet.concatenate_tensors([img.unsqueeze(0) for img in images]).squeeze().cpu()

        return full_images, predictions
    
    
    num_losses = len(loss_functions)

    plt.figure(figsize=(2.5*(num_losses + 2), 2.5*num_images))
    for id_im in range(num_images):
        # Retrieve split images and mask for the current image
        split_images, mask = dataset[id_im]
        image, pred = process_images(split_images)

        # Loop over each loss function to calculate and display the loss
        for id_loss, loss_func in enumerate(loss_functions):
            # Compute subplot index in the current row for each loss function
            ax = plt.subplot(num_images, num_losses + 2, id_im * (num_losses + 2) + id_loss + 1)
            
            # Calculate the pixel-wise loss
            loss = loss_func(pred.squeeze(), mask.squeeze())
            print("Shape of loss: ", loss.shape)
            
            # Display the loss as an image
            im = plt.imshow(loss.squeeze(), cmap="RdBu_r")
            plt.title(f"{loss_func.name}", size=5)  # Use the class name of the loss function
            plt.axis('off')
            
            # cbar = plt.colorbar(im, ax=ax, shrink=0.5)
            # cbar.ax.tick_params(labelsize=3)  # Adjust colorbar label text size


        # Plot the ground truth mask
        plt.subplot(num_images, num_losses + 2, id_im * (num_losses + 2) + num_losses + 1)
        plt.imshow(mask.squeeze(), cmap="RdBu_r", vmin=0, vmax=1)
        plt.title("Ground truth", size=5)
        plt.axis('off')

        # Plot the predicted image
        plt.subplot(num_images, num_losses + 2, id_im * (num_losses + 2) + num_losses + 2)
        plt.imshow(pred.squeeze(), cmap="RdBu_r", vmin=0, vmax=1)
        plt.title("Prediction", size=5)
        plt.axis('off')

    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=200)
        plt.close()
    else:
        plt.show()
        plt.close()

def find_folders_with_unet_pth(directory):
    """
    Collects all folders in the specified directory that contain a file named 'unet.pth'.
    """
    folders_with_unet = []

    # Iterate through the directory
    for root, _, files in os.walk(directory):
        if 'unet.pth' in files:
            folders_with_unet.append(root)
    return folders_with_unet

def compute_validation_loss(model, loss_fn, dataloader, device, data_type, half_precision, verbose=False):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        with (torch.autocast(device_type='cuda', dtype=data_type) if half_precision else contextlib.nullcontext()):
            for images, masks in tqdm(dataloader, disable=(not verbose)):
                images = [im.half() for im in images]
                masks = masks.to(device, dtype=torch.half)

                predictions = model(images)
                loss = loss_fn(predictions, masks)
                total_loss += loss.item()
                num_batches += 1
        
    average_loss = total_loss / num_batches
    return average_loss

if __name__=="__main__":
    # Load dataset
    image_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_5000dp inputs_pkixy outputs_t/Inputs"
    mask_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_5000dp inputs_pkixy outputs_t/Labels"
    
    dataloaders = init_data(data_settings={"subdomains_dist":(1,1), "batch_size_training": 2, "batch_size_testing": 2}, image_dir=image_dir, mask_dir=mask_dir)

    base_directory = "/scratch/e451412/code/results/pkixy_5000_new"
    folders = find_folders_with_unet_pth(base_directory)
    random.shuffle(folders)

    loss_funcs = [PixelwiseL1Loss(), PixelwiseMSELoss(), CombiLoss(alpha=0.25), CombiLoss(alpha=0.5), CombiLoss(alpha=0.75), ThresholdedMAELoss(threshold=0.02, weight_ratio=0.1), WeightedMSELoss(epsilon=0.1), PixelwiseRMSELoss()]

    for folder in folders:
        print(rf"Evaluating folder: {folder}", flush=True)

        # Define the paths for the loss comparison file and visualization
        losses_path = os.path.join(folder, 'loss_comparison.json')
        image_path = os.path.join(folder, 'losses_visualized.png')
        
        # Check if the loss comparison file already exists
        if os.path.exists(losses_path):
            print(rf"Loss comparison file already exists: {losses_path}. Skipping this folder: {folder}", flush=True)
            continue  # Skip to the next folder
        
        unet, args = load_unet(folder)
        plot_different_loss_functions(unet, loss_funcs, dataset=dataloaders['val'].dataset, savepath=image_path, num_images=2)
        
        try:
            losses = {}
            for loss_func in loss_funcs:
                val_loss = compute_validation_loss(model=unet, data_type=DATA_TYPE, device=DEVICES[0], dataloader=dataloaders['val'], half_precision=HALF_PRECISION,
                                                loss_fn=loss_func, verbose=True)
                losses[str(loss_func.name)] = val_loss

            # Save losses as JSON in the current folder
            losses_path = os.path.join(folder, 'loss_comparison.json')
            with open(losses_path, 'w') as json_file:
                json.dump(losses, json_file, indent=4)

        except Exception as e:
            print(rf"Exception occured: {e}. Skipping this folder: {folder}", flush=True)




