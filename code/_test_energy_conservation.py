import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataprocessing import *
from utils import parse_args

image_dir = "/projects/ddu_net_heat_plume_prediction/HeatPlumePredictionWithCNNs/data/dataset_large_square_6hp_varyK_5000dp inputs_pkixy outputs_t/Inputs"
mask_dir = "/projects/ddu_net_heat_plume_prediction/HeatPlumePredictionWithCNNs/data/dataset_large_square_6hp_varyK_5000dp inputs_pkixy outputs_t/Labels"

args = parse_args()

_, datasets = init_data(args, image_dir, mask_dir)

temperature_field = datasets['train'][0][1]
inputs = datasets['train'][0][0]

# inputs[0].shape #pkixy