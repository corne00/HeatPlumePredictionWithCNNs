import torch
import yaml
import pathlib

from models import MultiGPU_UNet_with_comm

# Set image sizes of training and testing evaluations
train_size = (2560, 2560)
eval_size = (5120, 5120)

# Specify paths and load data
settings_path = ".../*.yaml"
ddunet_path = ".../*.pth"
settings = yaml.safe_load(open(settings_path))

# Initialize a model given the settings
devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]

# Load the old subdomain disttribution and update it according to the new eval_size
subdom_dist = settings["subdom_dist"]
subdomains_dist_new = (subdom_dist[0] * (eval_size // train_size), subdom_dist[1] * (eval_size // train_size))
settings["subdom_dist"] = subdomains_dist_new
print("Subdomain distribution used for evaluating images:", subdomains_dist_new)

# Initialize the new model
model = MultiGPU_UNet_with_comm(settings, devices=devices)
print("Loading pretrained model from ", ddunet_path)
model.load_weights(load_path=ddunet_path, device=devices[0])
