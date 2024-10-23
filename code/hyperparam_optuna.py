# Standard library imports
import os
import json
import pathlib
import numpy as np

import torch
from torch.cuda.amp import GradScaler
import optuna
from utils.losses import ThresholdedMSELoss, WeightedMSELoss

from models import *
from utils import parse_args, save_args_to_json, plot_results
from dataprocessing import init_data
from utils.train_utils import *
from utils.visualization import *
from utils.losses import CombiLoss

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
        print("Results directory already exists!", flush=True)
        exit()


    # Check if we have half precision
    half_precision = torch.cuda.is_available()
    data_type = torch.float16 if half_precision else torch.float32

    # Half precision scaler
    scaler = GradScaler(enabled=half_precision)

    # Set devices
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    print("Available GPUs:", devices, flush=True)

    # Set datasets
    image_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_5000dp inputs_pkixy outputs_t/Inputs"
    mask_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_5000dp inputs_pkixy outputs_t/Labels"
    
    try:
        args.batch_size_training = trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32])
        dataloaders, datasets = init_data(args, image_dir, mask_dir)

        # Generate the model
        args.depth = trial.suggest_categorical("depth", [2, 3, 4, 5, 6])
        args.complexity = trial.suggest_categorical("complexity", [8, 16, 32])
        args.kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        args.num_convs = trial.suggest_categorical("num_convs", [2, 3, 4])

        # Define loss function options
        loss_functions = {
            "mse": torch.nn.MSELoss(),
            "combi": CombiLoss(0.5),
            "combi_RMSE_MAE": CombiRMSE_and_MAELoss(),
            "l1": torch.nn.L1Loss(),
            "thresholded_mse": ThresholdedMSELoss(),
            "weighted_mse_epsilon_0_1": WeightedMSELoss(epsilon=0.1),
            "weighted_mse_epsilon_0_001": WeightedMSELoss(epsilon=0.001),
        }

        # Generate the optimizers
        lr = trial.suggest_categorical("lr", [1e-3, 2e-4, 1e-4, 5e-5, 1e-5])
        weight_decay_adam = 0 #trial.suggest_categorical("weight_decay", [0])
        loss_fn_alpha = trial.suggest_categorical("loss_alpha", [1., 0., 0.25, 0.75, 0.5])

        # Select the loss function based on the trial's suggestion
        loss_function_name = trial.suggest_categorical("loss function", list(loss_functions.keys()))
        loss_function = loss_functions[loss_function_name]
        
        # Add the selected values to args
        args.lr = lr
        args.loss_function = loss_function_name
        args.weight_decay_adam = weight_decay_adam
        args.loss_fn_alpha = 1.

        save_args_to_json(args=args, filename=os.path.join(args.save_path, "args.json"))

        unet, data = train_parallel_model(model=MultiGPU_UNet_with_comm, dataloader_val=dataloaders["val"], 
                                                                    dataloader_train=dataloaders["train"], scaler=scaler, data_type=data_type, 
                                                                    half_precision=True, train_dataset=datasets["train"], val_dataset=datasets["val"], 
                                                                    comm=args.comm, num_epochs=args.num_epochs, num_comm_fmaps=args.num_comm_fmaps,  
                                                                    save_path=args.save_path, subdomains_dist=args.subdomains_dist, 
                                                                    exchange_fmaps=args.exchange_fmaps, padding=args.padding, 
                                                                    depth=args.depth, kernel_size=args.kernel_size, communication_network=None, 
                                                                    complexity=args.complexity, dropout_rate=0.0, devices=devices, 
                                                                    num_convs=args.num_convs, weight_decay_adam=weight_decay_adam, 
                                                                    loss_fn_alpha=loss_fn_alpha, lr=lr,
                                                                    loss_func=loss_function, val_loss_func=loss_functions[args.val_loss], verbose=False,
                                                                 num_channels=5, plot_freq=1, track_loss_functions=loss_functions)
        
        loss = np.min(data["val_losses"])

        # Save and calculate losses
        evaluate(args, unet, data, dataloaders, datasets)
        
    except Exception as e:
        print(f"Training failed with exception: {e}", flush=True)
        loss = 1
    
    print("Finished!", flush=True)
    return loss


if __name__ == "__main__":
    print("Running", flush=True)
    STUDY_DIR = "/scratch/e451412/code/results/pkixy_5000_new"
    study_dir = pathlib.Path(STUDY_DIR)
    study_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="minimize", storage=f"sqlite:///{STUDY_DIR}/hyperparam_opti.db", study_name="search", load_if_exists=True)
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