import torch
import torch.nn as nn
import numpy as np
import yaml
from copy import deepcopy

from dataprocessing.equations_of_state import eos_water_density_IFC67, eos_water_enthalphy
from dataprocessing.data_utils import NormalizeTransform

class WeightedMSELoss(nn.Module):
    def __init__(self, epsilon:float=1e-1, only_target_based:bool=False):
        """
        Args:
        - epsilon (float) : value to make sure that every point in the domain gets at least some weight
        - only_target_based (bool) : if True, only targets are used for constructing weight mask
        """
        super(WeightedMSELoss, self).__init__()
        self.epsilon = epsilon
        self.only_target_based = only_target_based
        self.name = rf"WeightedMSELoss (e={self.epsilon})"
        self.mse = nn.MSELoss()
    
    def forward(self, prediction, target):
        # Calculate the element-wise maximum between prediction and target
        if self.only_target_based:
            weight = target + self.epsilon
        else:
            weight = torch.max(prediction, target) + self.epsilon
        
        # Return the weighted mean squared error
        return torch.mean(weight * self.mse(prediction, target))
        
class WeightedMAELoss(nn.Module):
    def __init__(self, epsilon:float=1e-1, only_target_based:bool=False):
        """
        Args:
        - epsilon (float) : value to make sure that every point in the domain gets at least some weight
        - only_target_based (bool) : if True, only targets are used for constructing weight mask
        """
        super(WeightedMAELoss, self).__init__()
        self.epsilon = epsilon
        self.only_target_based = only_target_based
        self.name = rf"WeightedMSELoss (e={self.epsilon})"
        self.mae = nn.L1Loss(reduction="none")
    
    def forward(self, prediction, target):
        # Calculate the element-wise maximum between prediction and target
        if self.only_target_based:
            weight = target + self.epsilon
        else:
            weight = torch.max(prediction, target) + self.epsilon
        
        # Return the weighted mean absolute error
        return torch.mean(weight * self.mae(prediction, target))
    
class ThresholdedMAELoss(nn.Module):
    """
    Function that puts more weight on pixels close to the stream lines.
    """
    def __init__(self, threshold:float=0.02, weight_ratio=0.1):
        super(ThresholdedMAELoss, self).__init__()
        self.threshold = threshold
        self.weight_ratio = weight_ratio
        self.name = rf"ThresholdedMAE (t={threshold}, w={weight_ratio})"
        self.mae = nn.L1Loss()
    def forward(self, prediction, target):
        # Calculate the element-wise maximum between prediction and target
        weight = torch.where(target > self.threshold, 1., self.weight_ratio)
                
        # Return the weighted mean absolute error
        return torch.mean(weight * self.mae(prediction, target))
    
class CombiLoss(nn.Module):
    """
    Loss function that combines MSE and MAE loss with a certain ratio alpha
    """
    def __init__(self, alpha: float = 1.):
        super(CombiLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.alpha = alpha
        self.name = rf"CombiLoss (a={alpha})"

    def forward(self, x, y):
        return self.alpha * self.mse(x, y) + (1. - self.alpha) * self.mae(x, y)

class CombiRMSE_and_MAELoss(nn.Module):
    """
    Loss function that combines MSE and MAE loss with a certain ratio alpha
    """
    def __init__(self):
        super(CombiRMSE_and_MAELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.name = rf"CombiLoss RMSE and MAE"

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y)) + self.mae(x, y)

class RMSELoss(nn.MSELoss):
    """
    Computation loss of the MSE loss with option for pixel loss (to see the local errors better)
    """
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.name = "RMSELoss"
    def forward(self, input, target):
        return torch.sqrt(super(RMSELoss, self).forward(input, target))
       

class EnergyLoss(nn.Module):
    def __init__(self, data_dir, dataloader):
        super(EnergyLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.norm_info = yaml.load(open(data_dir+"info.yaml"), Loader=yaml.SafeLoader)
        assert "Liquid X-Velocity [m_per_y]" in self.norm_info["Inputs"], "Velocity-x not in Inputs"
        self.norm = NormalizeTransform(self.norm_info)
        inputs, _ = next(iter(dataloader))
        inputs = torch.tensor(inputs)
        print(len(inputs[0]), inputs[0].shape)
        self.inputs_unnormed = self.norm.reverse(deepcopy(inputs[0]), "Inputs")
        self.pressure = self.inputs_unnormed[self.norm_info["Inputs"]["Liquid Pressure [Pa]"]["index"]]
        self.vx = self.inputs_unnormed[self.norm_info["Inputs"]["Liquid X-Velocity [m_per_y]"]["index"]]
        self.vy = self.inputs_unnormed[self.norm_info["Inputs"]["Liquid X-Velocity [m_per_y]"]["index"]]
        self.ids_normed = inputs[0][self.norm_info["Inputs"]["Material ID"]["index"]]

    def forward(self, prediction, target):
        # TODO target not needed
        predicted_T_unnormed = self.norm.reverse(deepcopy(prediction), "Labels")
        # TODO dimensions! expect 2D
        print(predicted_T_unnormed.shape, self.vx.shape, self.ids_normed.shape)
        loss = energy_loss(self.pressure, predicted_T_unnormed, self.vx, self.vy, self.ids_normed, self.mse_loss)
        return loss

def energy_loss(pressure, predicted_temperature, vx, vy, ids_normed, mse_loss):
    #  based on : Manuel Hirche, Bachelor thesis, 2023
    resolution = 5. #m

    # cond_dry = 0.65
    # cond_sat = 1.0
    # sl  = 1 #? saturation of liquid?
    thermal_conductivity = 1 #cond_dry + torch.sqrt(sl) * (cond_sat - cond_dry)
    density, molar_density = eos_water_density_IFC67(predicted_temperature, pressure)
    enthalpy = eos_water_enthalphy(predicted_temperature, pressure)
    T_grad = torch.gradient(predicted_temperature)
    energy_u = torch.gradient((molar_density * vx * enthalpy) - (thermal_conductivity * T_grad[0]/resolution))[0]/resolution
    energy_v = torch.gradient((molar_density * vy * enthalpy) - (thermal_conductivity * T_grad[1]/resolution))[1]/resolution
    energy = energy_u + energy_v

    inflow_energy = energy_hps(ids_normed, resolution, density)
    energy -= inflow_energy*0.5

    energy_loss = mse_loss(torch.Tensor(energy), torch.zeros_like(torch.Tensor(energy)))
    return energy_loss

def energy_hps(ids, resolution, density):
    specific_heat_water = 4200 # [J/kgK]
    density_water = density # [kg/m^3]
    temp_diff = 5 # [K]
    volumetric_flow_rate = 0.00024 # [m^3/s]

    hp_energy = specific_heat_water * density_water * temp_diff * volumetric_flow_rate * 1/resolution**3
    hp_energy = hp_energy * ids

    kernel = torch.tensor([[-1,0,1],
                           [0,0,0],
                           [1,0,-1]],dtype=torch.float32)
    
    hp_energy = torch.nn.functional.conv2d(hp_energy.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1)

    return (hp_energy[0,0])

if __name__ == "__main__":
    # Dummy data for prediction and target
    prediction = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.5, 1.8, 2.5, 0.0, 0.0, 0.1])
    
    # CombiRMSE_and_MAELoss
    combi_rmse_mae_loss = CombiRMSE_and_MAELoss()
    loss_combi_rmse_mae = combi_rmse_mae_loss(prediction, target)
    print(f"Combi RMSE and MAE Loss (mean): {loss_combi_rmse_mae.item()}")
    
    # --- WeightedMSELoss ---
    wmse_loss = WeightedMSELoss(epsilon=1e-1)
    loss_wmse = wmse_loss(prediction, target)
    print(f"WeightedMSE Loss (mean): {loss_wmse.item()}")

    # --- WeightedMAELoss ---
    wmae_loss = WeightedMAELoss(epsilon=1e-1)
    loss_wmae = wmae_loss(prediction, target)
    print(f"WeightedMAE Loss (mean): {loss_wmae.item()}")
    
    # --- ThresholdedMSELoss ---
    thresholded_mse_loss = ThresholdedMAELoss(threshold=0.02, weight_ratio=0.1)
    loss_thresholded = thresholded_mse_loss(prediction, target)
    print(f"Thresholded MSE Loss (mean): {loss_thresholded.item()}")
    
    # --- CombiLoss (combined MSE and MAE) ---
    combi_loss = CombiLoss(alpha=0.5)
    loss_combi = combi_loss(prediction, target)
    print(f"Combi Loss (mean): {loss_combi.item()}")
    
    # --- PixelwiseMSELoss ---
    pixelwise_mse_loss = torch.nn.MSELoss()
    loss_pixelwise_mse = pixelwise_mse_loss(prediction, target)
    print(f"Pixelwise MSE Loss (mean): {loss_pixelwise_mse.mean().item()}")
    
    # --- PixelwiseL1Loss ---
    pixelwise_l1_loss = torch.nn.L1Loss()
    loss_l1 = pixelwise_l1_loss(prediction, target)
    print(f"L1 Loss (mean): {loss_l1.mean().item()}")
