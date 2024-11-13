# Standard library imports
import os
import json
import pathlib
import numpy as np

import torch
from torch.cuda.amp import GradScaler
import optuna
from argparse import Namespace
from typing import Dict

from models import *
from utils.arg_parser import parse_args, save_args_to_json
from utils.visualization import plot_results
from dataprocessing import init_data
from utils.train_utils import *
from utils.visualization import *
from utils.losses import ThresholdedMAELoss, WeightedMAELoss, CombiRMSE_and_MAELoss, CombiLoss, EnergyLoss
from dataprocessing.dataloaders import DatasetMultipleSubdomains

STUDY_DIR = "/scratch/sgs/pelzerja/DDUNet/code/results/unittesting"

def evaluate(args:Namespace, unet:MultiGPU_UNet_with_comm, losses:Dict[str,list], datasets:Dict[str,DatasetMultipleSubdomains]):
    plot_results(unet=unet, savepath=args.save_path, epoch_number="best", train_dataset=datasets["train"], val_dataset=datasets["val"])
    
    # Save to a JSON file
    with open(os.path.join(args.save_path, 'losses.json'), 'w') as json_file:
        json.dump(losses, json_file)

def objective(trial):
    # Load and save the arguments from the arge parser
    args = parse_args()
    args.save_path = f"{STUDY_DIR}/{trial.number}"
    try:
        os.makedirs(args.save_path)
    except:
        print("Results directory already exists!", flush=True)
        exit()


    # Check if we have half precision
    half_precision = torch.cuda.is_available()
    data_type = torch.float16 if half_precision else torch.float32

    # Half precision scaler
    scaler = GradScaler(enabled=half_precision)

    # Set devices
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    devices = ["cuda:2"]
    print("Available GPUs:", devices, flush=True)

    # Set datasets
    # data_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_5000dp inputs_pki outputs_t/"
    data_dir = "/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_large_square_6hp_varyK_5000dp inputs_pki outputs_t/"
    image_dir = data_dir+"Inputs"
    label_dir  = data_dir+"Labels"
    num_channels = 3
    try:
        args.batch_size_training = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
        dataloaders, datasets = init_data(args, image_dir, label_dir)

        # Generate the model
        args.depth = trial.suggest_categorical("depth", [4, 5, 6, 7])
        args.complexity = trial.suggest_categorical("complexity", [8, 16, 32, 64])
        args.kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        args.num_convs = trial.suggest_categorical("num_convs", [1, 2, 3])

        # Define loss function options
        loss_functions = {
            "mse": torch.nn.MSELoss(),
            "combi_0_75": CombiLoss(0.75),
            "combi_0_5": CombiLoss(0.5),
            "combi_0_25": CombiLoss(0.25),
            "l1": torch.nn.L1Loss(),
            # "combi_RMSE_MAE": CombiRMSE_and_MAELoss(),
            "thresholded_mae_0_02": ThresholdedMAELoss(threshold=0.02, weight_ratio=0.1),
            "thresholded_mae_0_04": ThresholdedMAELoss(threshold=0.04, weight_ratio=0.1),
            "thresholded_mae_0_01": ThresholdedMAELoss(threshold=0.01, weight_ratio=0.1),
            "weighted_mae_epsilon_0_1": WeightedMAELoss(epsilon=0.1),
            # "weighted_mae_epsilon_0_2": WeightedMAELoss(epsilon=0.2),
            # "weighted_mae_epsilon_0_05": WeightedMAELoss(epsilon=0.05),
            # "energy_mse": EnergyLoss(data_dir=data_dir, dataset=dataloaders["val"])
        }

        track_loss_functions = {
            "mse": torch.nn.MSELoss(),
            "l1": torch.nn.L1Loss(),
        }

        # Generate the optimizers
        args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True) #suggest_categorical("lr", [1e-3, 2e-4, 1e-4, 5e-5, 1e-5])
        args.weight_decay_adam = trial.suggest_categorical("weight_decay", [0, 1e-4, 1e-3, 1e-2, 1e-1])

        # Select the loss function based on the trial's suggestion
        loss_function_names = list(loss_functions.keys())
        args.loss_function = trial.suggest_categorical("loss function", loss_function_names)
        loss_function = loss_functions[args.loss_function]
        
        save_args_to_json(args=args, filename=os.path.join(args.save_path, "args.json"))

        unet, data = train_parallel_model(model=MultiGPU_UNet_with_comm, dataloader_val=dataloaders["val"], 
                                                                    dataloader_train=dataloaders["train"], scaler=scaler, data_type=data_type, 
                                                                    half_precision=True, train_dataset=datasets["train"], val_dataset=datasets["val"], 
                                                                    comm=args.comm, num_epochs=args.num_epochs, num_comm_fmaps=args.num_comm_fmaps,  
                                                                    save_path=args.save_path, subdomains_dist=args.subdomains_dist, 
                                                                    exchange_fmaps=args.exchange_fmaps, padding=args.padding, 
                                                                    depth=args.depth, kernel_size=args.kernel_size, communication_network=None, 
                                                                    complexity=args.complexity, dropout_rate=0.0, devices=devices, 
                                                                    num_convs=args.num_convs, weight_decay_adam=args.weight_decay_adam, lr=args.lr,
                                                                    loss_func=loss_function, val_loss_func=loss_functions[args.val_loss], verbose=False,
                                                                    num_channels=num_channels, plot_freq=10, track_loss_functions=track_loss_functions)
        
        loss = np.min(data["val_losses"])

        # Save and calculate losses
        evaluate(args, unet, data, datasets)
        
    except Exception as e:
        print(f"Training failed with exception: {e}", flush=True)
        raise optuna.TrialPruned()
    
    print("Finished!", flush=True)
    return loss


if __name__ == "__main__":
    print("Running")
    
    # STUDY_DIR = "/scratch/e451412/code/results/pki_5000_loss_functions"
    STUDY_DIR = "/scratch/sgs/pelzerja/DDUNet/code/results/pki_5000_exhaustive_search"

    study_dir = pathlib.Path(STUDY_DIR)
    study_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="minimize", storage=f"sqlite:///{STUDY_DIR}/hyperparam_opti.db", study_name="search", load_if_exists=True)
    study.optimize(objective, n_trials=10)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print("    {}: {}".format(key, value))

    print("Complete trials:")
    for trial in complete_trials:
        print("  Trial {}: {}".format(trial.number, trial.value))

    print("Pruned trials:")
    for trial in pruned_trials:
        print("  Trial {}: {}".format(trial.number, trial.value))
