# Standard library imports
import os
import json
import pathlib
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import optuna
from argparse import Namespace
from typing import Dict

from models import *
from utils.prepare_settings import prepare_settings, save_args_to_json
from utils.visualization import plot_results
from dataprocessing import init_data
from utils.train_utils import *
from utils.visualization import *
from utils.losses import ThresholdedMAELoss, WeightedMAELoss, CombiRMSE_and_MAELoss, CombiLoss, EnergyLoss, matchLoss
from dataprocessing.dataloaders import DatasetMultipleSubdomains

STUDY_DIR = "/scratch/sgs/pelzerja/DDUNet/code/results/unittesting"

def evaluate(unet:MultiGPU_UNet_with_comm, losses:Dict[str,list], dataloaders:Dict[str,DataLoader], save_path: pathlib.Path):
    plot_results(unet=unet, savepath=save_path, epoch_number="best", dataloaders=dataloaders)
    
    # Save to a JSON file
    with open(save_path / 'losses.json', 'w') as json_file:
        json.dump(losses, json_file)

def objective(trial):
    # Load and save the arguments from the arge parser and default settings
    settings, save_path = prepare_settings()

    # OPTUNAT: OVERWRITE 
    save_path = f"{STUDY_DIR}/{trial.number}"
    try:
        os.makedirs(save_path)
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

    # OPTUNAT: OVERWRITE 
    # data_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_5000dp inputs_pki outputs_t/"
    settings["data"]["dir"] = "/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_large_square_6hp_varyK_5000dp inputs_pki outputs_t/"
    settings["data"]["num_channels"] = 3


    try:
        settings["data"]["batch_size_training"] = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
        dataloaders = init_data(settings["data"], image_dir=settings["data"]["dir"]+"Inputs", mask_dir=settings["data"]["dir"]+"Labels")

        # Generate the model
        settings["model"]["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])
        settings["model"]["UNet"]["depth"] = trial.suggest_categorical("depth", [4, 5, 6, 7])
        settings["model"]["UNet"]["complexity"] = trial.suggest_categorical("complexity", [8, 16, 32, 64])
        settings["model"]["UNet"]["num_convs"] = trial.suggest_categorical("num_convs", [1, 2, 3])

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
            # "energy_mse": EnergyLoss(data_dir=data_dir, device=devices[0]),
        }

        track_loss_functions = {
            "mse": torch.nn.MSELoss(),
            "l1": torch.nn.L1Loss(),
            # "energy":EnergyLoss(data_dir=data_dir, device=devices[0]),
        }

        # Generate the optimizers
        settings["training"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True) #suggest_categorical("lr", [1e-3, 2e-4, 1e-4, 5e-5, 1e-5])
        settings["training"]["adam_weight_decay"] = trial.suggest_categorical("weight_decay", [0, 1e-4, 1e-3, 1e-2, 1e-1])

        # Select the loss function based on the trial's suggestion
        settings["training"]["train_loss"] = trial.suggest_categorical("loss function", list(loss_functions.keys()))
        
        print("final settings", settings)
        # dump settings to save_path
        with open(save_path / 'settings.yaml', 'w') as f:
            yaml.dump(settings, f)

        loss_func = loss_functions[settings["training"]["train_loss"]]
        val_loss_func = loss_functions[settings["training"]["val_loss"]]
        model = MultiGPU_UNet_with_comm(settings, devices=devices)
        unet, data = train_parallel_model(model, dataloaders, settings, devices, save_path, scaler=scaler, data_type=data_type,  half_precision=half_precision, loss_func=loss_func, val_loss_func=val_loss_func, track_loss_functions=track_loss_functions) 
        
        loss = np.min(data["val_losses"])

        # Save and calculate losses
        evaluate(unet, data, dataloaders, save_path)
        
    except Exception as e:
        print(f"Training failed with exception: {e}", flush=True)
        raise optuna.TrialPruned()
    
    print("Finished!", flush=True)
    return loss

def run():
    # Load and save the arguments from the arge parser and default settings
    settings, save_path = prepare_settings()

    # Check if we have half precision
    half_precision = torch.cuda.is_available()
    data_type = torch.float16 if half_precision else torch.float32
    # Half precision scaler
    scaler = GradScaler(enabled=half_precision)

    # Set devices
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    devices = ["cuda:2"]
    print("Available GPUs:", devices, flush=True)

    dataloaders = init_data(settings["data"], image_dir=settings["data"]["dir"]+"Inputs", mask_dir=settings["data"]["dir"]+"Labels")
    
    track_loss_functions = {
        "mse": torch.nn.MSELoss(),
        "l1": torch.nn.L1Loss(),
        "energy":EnergyLoss(data_dir=settings["data"]["dir"], device=devices[0]),
    }

    print("final settings", settings)
    # dump settings to save_path
    with open(save_path / 'settings.yaml', 'w') as f:
        yaml.dump(settings, f)

    loss_func = matchLoss(settings["training"]["train_loss"], data_dir=settings["data"]["dir"], device=devices[0])
    val_loss_func = matchLoss(settings["training"]["val_loss"])
    model = MultiGPU_UNet_with_comm(settings, devices=devices)
    unet, data = train_parallel_model(model, dataloaders, settings, devices, save_path, scaler=scaler, data_type=data_type,  half_precision=half_precision, loss_func=loss_func, val_loss_func=val_loss_func, track_loss_functions=track_loss_functions) 

    # Save and calculate losses
    evaluate(unet, data, dataloaders, save_path)
        
    print("Finished!", flush=True)

if __name__ == "__main__":
    hyperparam_search = False
    print(f"Running {'hyperparameter search' if hyperparam_search else 'single run'}", flush=True)
    
    # STUDY_DIR = "/scratch/e451412/code/results/pki_5000_loss_functions"
    STUDY_DIR = "/scratch/sgs/pelzerja/DDUNet/code/results/test_energy_loss"

    study_dir = pathlib.Path(STUDY_DIR)
    study_dir.mkdir(parents=True, exist_ok=True)

    if hyperparam_search:
        study = optuna.create_study(direction="minimize", storage=f"sqlite:///{STUDY_DIR}/hyperparam_opti.db", study_name="search", load_if_exists=True)
        study.optimize(objective, n_trials=20)

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

    else:
        run()