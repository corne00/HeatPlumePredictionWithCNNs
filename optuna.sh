#!/bin/bash
#
#SBATCH --job-name="heat plume prediction"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=3:59:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1             # Request four GPUs
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=8GB         # Require 8GB of memory per GPU
#SBATCH --account=research-eemcs-diam
#SBATCH --output=/scratch/e451412/slurm_outputs/slurm-opti-%j.out

# Load modules
module load 2024r1 openmpi py-torch 
module load py-numpy py-geopandas py-tqdm py-pillow 
module load py-matplotlib py-geopandas cuda py-scikit-learn
module load py-pip
python -m pip install --user optuna
python -m pip install --user tensorboard

python -m pip install --user typing-extensions --upgrade

# Execute the command
cd /scratch/e451412/code
srun python ./hyperparam_optuna.py --num_epochs 50

