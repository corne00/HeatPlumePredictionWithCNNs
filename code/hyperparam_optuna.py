# Library imports
import json
import pathlib
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import optuna
from typing import Dict

from models import *
from utils.prepare_settings import prepare_settings, init_hyperparams_and_settings
from utils.visualization import plot_results
from dataprocessing import init_data
from utils.train_utils import *
from utils.visualization import *
from utils.losses import ThresholdedMAELoss, WeightedMAELoss, CombiRMSE_and_MAELoss, CombiLoss, EnergyLoss, matchLoss
from dataprocessing.dataloaders import DatasetMultipleSubdomains

STUDY_DIR = "/scratch/sgs/pelzerja/DDUNet/code/results/testing_sett"

def evaluate(unet:MultiGPU_UNet_with_comm, losses:Dict[str,list], dataloaders:Dict[str,DataLoader], save_path: pathlib.Path):
    plot_results(model=unet, savepath=save_path, epoch_number="best", dataloaders=dataloaders)
    
    # Save to a JSON file
    with open(save_path / 'losses.json', 'w') as json_file:
        json.dump(losses, json_file)

def objective(trial):
    # Load and save the arguments from the arge parser and default settings
    hyperparams, settings = init_hyperparams_and_settings(path=pathlib.Path(STUDY_DIR))

    # OPTUNAT: OVERWRITE 
    save_path = pathlib.Path(f"{STUDY_DIR}/{trial.number}")
    save_path.mkdir(parents=True, exist_ok=False)


    # Check if we have half precision
    half_precision = torch.cuda.is_available()
    data_type = torch.float16 if half_precision else torch.float32
    # Half precision scaler
    scaler = GradScaler(enabled=half_precision)

    # Set devices
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    # devices = ["cuda:2"]
    print("Available GPUs:", devices, flush=True)

    # OPTUNAT: OVERWRITE 
    # data_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_5000dp inputs_pki outputs_t/"
    settings["data"]["dir"] = hyperparams["data"]
    settings["model"]["UNet"]["num_channels"] = hyperparams["num_channels"]


    try:
        track_loss_functions = {
            "mse": torch.nn.MSELoss(),
            "l1": torch.nn.L1Loss(),
            # "energy":EnergyLoss(data_dir=data_dir, device=devices[0]),
        }

        settings["data"]["batch_size_training"] = trial.suggest_categorical("batch_size", [int(item) for item in hyperparams["batch_size"]])
        dataloaders = init_data(settings["data"], image_dir=settings["data"]["dir"]+"Inputs", mask_dir=settings["data"]["dir"]+"Labels")

        # Generate trial suggestions
        settings["model"]["kernel_size"] = trial.suggest_categorical("kernel_size", [int(item) for item in hyperparams["kernel_size"]])
        settings["model"]["UNet"]["depth"] = trial.suggest_categorical("depth", [int(item) for item in hyperparams["depth"]])
        settings["model"]["UNet"]["complexity"] = trial.suggest_categorical("complexity", [int(item) for item in hyperparams["complexity"]])
        settings["model"]["UNet"]["num_convs"] = trial.suggest_categorical("num_convs", [int(item) for item in hyperparams["num_convs"]])

        settings["training"]["lr"] = trial.suggest_float("lr", float(hyperparams["lr"]["min"]),  float(hyperparams["lr"]["max"]), log=hyperparams["lr"]["log"]) #suggest_categorical("lr", [1e-3, 2e-4, 1e-4, 5e-5, 1e-5])
        settings["training"]["adam_weight_decay"] = trial.suggest_categorical("weight_decay", [float(item) for item in hyperparams["weight_decay"]])

        settings["training"]["train_loss"] = trial.suggest_categorical("loss function", hyperparams["loss_functions"])
        
        print("trial settings", settings)

        # dump settings to save_path
        with open(save_path / 'settings.yaml', 'w') as f:
            yaml.dump(settings, f)

        loss_func = matchLoss(settings["training"]["train_loss"])
        val_loss_func = matchLoss(settings["training"]["val_loss"])
        model = MultiGPU_UNet_with_comm(settings, devices=devices)
        model, data = train_parallel_model(model, dataloaders, settings, devices, save_path, scaler=scaler, data_type=data_type,  half_precision=half_precision, loss_func=loss_func, val_loss_func=val_loss_func, track_loss_functions=track_loss_functions) 
        
        loss = np.min(data["val_losses"])

        # Save and calculate losses
        evaluate(model, data, dataloaders, save_path)
        
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
    model, data = train_parallel_model(model, dataloaders, settings, devices, save_path, scaler=scaler, data_type=data_type,  half_precision=half_precision, loss_func=loss_func, val_loss_func=val_loss_func, track_loss_functions=track_loss_functions) 

    # Save and calculate losses
    evaluate(model, data, dataloaders, save_path)
        
    print("Finished!", flush=True)

if __name__ == "__main__":
    hyperparam_search = True
    print(f"Running {'hyperparameter search' if hyperparam_search else 'single run'}", flush=True)
    
    # STUDY_DIR = "/scratch/e451412/code/results/pki_5000_loss_functions"
    STUDY_DIR = "/scratch/sgs/pelzerja/DDUNet/code/results/opti_settings"
    

    study_dir = pathlib.Path(STUDY_DIR)
    study_dir.mkdir(parents=True, exist_ok=True)

    if hyperparam_search:
        study = optuna.create_study(direction="minimize", storage=f"sqlite:///{STUDY_DIR}/hyperparam_opti.db", study_name="search", load_if_exists=True)
        study.optimize(objective, n_trials=1)

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