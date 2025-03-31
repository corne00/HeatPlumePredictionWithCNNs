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
#SBATCH --output=/projects/ddu_net_heat_plume_prediction/HeatPlumePredictionWithCNNs/slurm-opti-%j.out

# Load modules
module load 2024r1 openmpi py-torch 
module load py-numpy py-geopandas py-tqdm py-pillow 
module load py-matplotlib py-geopandas cuda py-scikit-learn
module load py-pip

# Execute the command
cd /projects/ddu_net_heat_plume_prediction/HeatPlumePredictionWithCNNs/code
srun python ./_test_ddu_net_without_padding.py

