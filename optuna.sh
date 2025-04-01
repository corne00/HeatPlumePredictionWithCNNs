#!/bin/bash
#
#SBATCH --job-name="heat plume prediction"
#SBATCH --partition=gpu-v100 # gpu-v100
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-gpu=32GB         # Require 8GB of memory per GPU
#SBATCH --account=research-eemcs-diam
#SBATCH --output=/projects/ddu_net_heat_plume_prediction/HeatPlumePredictionWithCNNs/slurm_outputs/slurm-opti-%j.out

# Load modules
module load 2024r1 openmpi py-torch 
module load py-numpy py-geopandas py-tqdm py-pillow 
module load py-matplotlib py-geopandas cuda py-scikit-learn
module load py-pip

# Execute the command
cd /projects/ddu_net_heat_plume_prediction/HeatPlumePredictionWithCNNs/code
srun python ./hyperparam_optuna.py --study_dir "/projects/ddu_net_heat_plume_prediction/HeatPlumePredictionWithCNNs/code/results/tests_week_14/scenario_3/with_vs_without_communication_network"