# Standard library imports
import os
import json
import pathlib
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import optuna
from utils.losses import ThresholdedMSELoss, WeightedMSELoss

from models import *
from utils import parse_args, save_args_to_json, plot_results
from dataprocessing import DatasetMultipleSubdomains
from utils.train_utils import *
from utils.visualization import *

def init_data(args, image_dir, mask_dir):
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
    args.save_path = f"{STUDY_DIR}/{trial.number}"
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
    image_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_5000dp inputs_pki outputs_t/Inputs"
    mask_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_5000dp inputs_pki outputs_t/Labels"
    
    try:
        args.batch_size_training = 16 #trial.suggest_categorical("batch_size", [16])
        dataloaders, datasets = init_data(args, image_dir, mask_dir)

        # Generate the model
        args.depth = 4 #trial.suggest_categorical("depth", [4])
        args.complexity = 16 #trial.suggest_categorical("complexity", [16])
        args.kernel_size = 7 #trial.suggest_categorical("kernel_size", [7])
        args.num_convs = 3 #trial.suggest_categorical("num_convs", [3])

        # Define loss function options
        loss_functions = {
            "mse": torch.nn.MSELoss(),
            "combi": CombiLoss(0.5),
            "l1": torch.nn.L1Loss(),
            "thresholded_mse": ThresholdedMSELoss(),
            "weighted_mse_epsilon_0_1": WeightedMSELoss(epsilon=0.1),
            "weighted_mse_epsilon_0_001": WeightedMSELoss(epsilon=0.001),
        }

        # Generate the optimizers
        lr = 1e-4 # trial.suggest_categorical("lr", [1e-4])
        weight_decay_adam = 0 #trial.suggest_categorical("weight_decay", [0])
        loss_fn_alpha = 1. #trial.suggest_categorical("loss_alpha", [1.])

        # Select the loss function based on the trial's suggestion
        loss_function_name = trial.suggest_categorical("loss function", list(loss_functions.keys()))
        loss_function = loss_functions[loss_function_name]
        
        # Add the selected values to args
        args.lr = lr
        args.loss_function = loss_function_name
        args.weight_decay_adam = weight_decay_adam
        args.loss_fn_alpha = 1.

        save_args_to_json(args=args, filename=os.path.join(args.save_path, "args.json"))

        unet, val_losses, training_losses = train_parallel_model(model=MultiGPU_UNet_with_comm, dataloader_val=dataloaders["val"], 
                                                                 dataloader_train=dataloaders["train"], scaler=scaler, data_type=data_type, 
                                                                 half_precision=True, train_dataset=datasets["train"], val_dataset=datasets["val"], 
                                                                 comm=args.comm, num_epochs=args.num_epochs, num_comm_fmaps=args.num_comm_fmaps,  
                                                                 save_path=args.save_path, subdomains_dist=args.subdomains_dist, 
                                                                 exchange_fmaps=args.exchange_fmaps, padding=args.padding, 
                                                                 depth=args.depth, kernel_size=args.kernel_size, communication_network=None, 
                                                                 complexity=args.complexity, dropout_rate=0.0, devices=devices, 
                                                                 num_convs=args.num_convs, weight_decay_adam=weight_decay_adam, 
                                                                 loss_fn_alpha=loss_fn_alpha, lr=lr,
                                                                 loss_func=loss_function, val_loss_func=torch.nn.L1Loss(), verbose=False)
        
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
    STUDY_DIR = "/scratch/e451412/code/results/pki_5000_loss_functions"
    study_dir = pathlib.Path(STUDY_DIR)
    study_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="minimize", storage=f"sqlite:///{STUDY_DIR}/hyperparam_opti.db", study_name="search", load_if_exists=True)
    study.optimize(objective, n_trials=50)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print("    {}: {}".format(key, value))