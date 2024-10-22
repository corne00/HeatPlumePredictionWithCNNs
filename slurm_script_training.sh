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
#SBATCH --output=/scratch/e451412/slurm_outputs/slurm-train-%j.out

# Load modules
module load 2023r1 openmpi py-torch 
module load py-numpy py-geopandas py-tqdm py-pillow 
module load py-matplotlib py-geopandas cuda py-scikit-learn

# Execute the command
cd /scratch/e451412/code

# srun python train.py --complexity 16 --depth 4 --kernel_size 7 --num-convs 3 --num_epochs 500 --save_path "./results/unet_depth_4_complexity_16"
srun python train.py --complexity 16 --depth 4 --kernel_size 5 --num-convs 2 --num_epochs 200 --save_path "./results/unet_data_loss_1"