# Standard library imports
import os
import time
import datetime
import random 
import json
import contextlib

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler

import optuna

from models import *
from utils import parse_args, save_args_to_json, plot_results
from dataprocessing import DatasetMultipleSubdomains
from utils.train_utils import *
from utils.visualization import *

def init_data(args, image_dir, mask_dir):
        image_labels = os.listdir(image_dir)
        split=[800,190,10]
        
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

def evaluate(args, unet, losses, dataloaders, datasets):
    plot_results(unet=unet, savepath=args.save_path, epoch_number="best", train_dataset=datasets["train"], val_dataset=datasets["val"])
    
    # Save to a JSON file
    with open(os.path.join(args.save_path, 'losses.json'), 'w') as json_file:
        json.dump(losses, json_file)

    # Call the function to compute average training loss and plot losses
    avg_training_losses = average_training_losses(training_losses=losses["training_losses"], 
                                                dataloader_train=dataloaders["train"], 
                                                num_epochs=args.num_epochs)

    plot_losses(avg_training_losses=avg_training_losses, val_losses=losses["val_losses"], save_path=args.save_path)


def objective(trial):
    # Load and save the arguments from the arge parser
    args = parse_args()
    try:
        os.makedirs(args.save_path)
    except:
        print("Results directory already exists!")
        exit()


    # Check if we have half precision
    half_precision = torch.cuda.is_available()
    data_type = torch.float16 if half_precision else torch.float32

    # Half precision scaler
    scaler = GradScaler(enabled=half_precision)

    # Set devices
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    print("Available GPUs:", devices)

    # Set datasets
    image_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_1000dp inputs_pki outputs_t/Inputs"
    mask_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_1000dp inputs_pki outputs_t/Labels"
    
    try:
        # args.batch_size_training = trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32])
        dataloaders, datasets = init_data(args, image_dir, mask_dir)

        # Generate the model
        args.depth = trial.suggest_categorical("depth", [2, 3, 4, 5])
        args.complexity = trial.suggest_categorical("complexity", [2, 4, 8, 16, 32, 64])
        args.kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        args.num_convs = trial.suggest_categorical("num_convs", [1, 2, 3, 4])

        # Generate the optimizers
        lr = trial.suggest_categorical("lr", [1e-3, 1e-4, 1e-5])
        weight_decay_adam = trial.suggest_categorical("weight_decay", [0, 1e-5])
        loss_fn_alpha = trial.suggest_categorical("loss_alpha", [0, 0.25, 0.5, 0.75, 1.])

        save_args_to_json(args=args, filename=os.path.join(args.save_path, "args.json"))
        
        unet, val_losses, training_losses = train_parallel_model(model=MultiGPU_UNet_with_comm, dataloader_val=dataloaders["val"], dataloader_train=dataloaders["train"], scaler=scaler, data_type=data_type, half_precision=True, train_dataset=datasets["train"], val_dataset=datasets["val"], comm=args.comm, num_epochs=args.num_epochs, num_comm_fmaps=args.num_comm_fmaps,  save_path=args.save_path, subdomains_dist=args.subdomains_dist, exchange_fmaps=args.exchange_fmaps, padding=args.padding, depth=args.depth, kernel_size=args.kernel_size, communication_network=None, complexity=args.complexity, dropout_rate=0.0, devices=devices, num_convs=args.num_convs, weight_decay_adam=weight_decay_adam, loss_fn_alpha=loss_fn_alpha, lr=lr)
        
        loss = np.min(val_losses)

        # Save and calculate losses
        data = {"val_losses": val_losses,"training_losses": training_losses}
        evaluate(args, unet, data, dataloaders, datasets)
        
    except Exception as e:
        print(f"Training failed with exception: {e}")
        loss = 1
    
    print("Finished!")
    return loss


if __name__ == "__main__":
    print("Running")
    study = optuna.create_study(direction="minimize", storage="sqlite:////scratch/e451412/code/results/hyperparam_tuning_new.db", study_name="unet", load_if_exists=True)
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print("    {}: {}".format(key, value))