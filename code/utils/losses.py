import torch
import torch.nn as nn
import yaml
from copy import deepcopy
import contextlib

from dataprocessing.equations_of_state import eos_water_density_IFC67, eos_water_enthalphy
from dataprocessing.data_utils import NormalizeTransform

def matchLoss(name:str, **kwargs):
    if name == "mse":
        return nn.MSELoss()
    elif name in ["mae", "l1"]:
        return nn.L1Loss()
    elif name == "weighted_mse":
        return WeightedMSELoss(**kwargs)
    elif name == "combi_0_75":
        return CombiLoss(0.75)
    elif name == "combi_0_5":
        return CombiLoss(0.5)
    elif name == "combi_0_25":
        return CombiLoss(0.25)
    elif name == "combi_RMSE_MAE":
        return CombiRMSE_and_MAELoss()
    elif name == "thresholded_mae_0_02":
        return ThresholdedMAELoss(threshold=0.02, weight_ratio=0.1)
    elif name == "thresholded_mae_0_04":
        return ThresholdedMAELoss(threshold=0.04, weight_ratio=0.1)
    elif name == "thresholded_mae_0_01":
        return ThresholdedMAELoss(threshold=0.01, weight_ratio=0.1)
    elif name == "weighted_mae_epsilon_0_1":
        return WeightedMAELoss(epsilon=0.1)
    elif name == "weighted_mae_epsilon_0_2":
        return WeightedMAELoss(epsilon=0.2)
    elif name == "weighted_mae_epsilon_0_05":
        return WeightedMAELoss(epsilon=0.05)
    elif name == "energy":
        return EnergyLoss(**kwargs)
    elif name == "mse_energy":
        return CombiLoss(0.999, EnergyLoss(**kwargs))
    else:
        raise ValueError(f"Loss function {name} not found. Consider extending 'matchLoss' function.")

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
    def __init__(self, alpha: float = 1., second_loss:nn.Module = nn.L1Loss()):
        super(CombiLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.secondary_loss_function = second_loss
        self.alpha = alpha
        self.name = rf"CombiLoss (a={alpha}) with {self.secondary_loss_function}"

    def forward(self, predictions, labels, inputs):
        if isinstance(self.secondary_loss_function, EnergyLoss):
            eval_second = self.secondary_loss_function(predictions, inputs)
        else:
            eval_second = self.secondary_loss_function(predictions, labels)

        return self.alpha * self.mse(predictions, labels) + (1. - self.alpha) * eval_second

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
       

class EnergyLoss(torch.nn.Module):
    def __init__(self, data_dir, device:str="cuda:0", data_type=torch.float32, half_precision=False, keep_dim:bool=False):
        super(EnergyLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.norm_info = yaml.load(open(data_dir+"info.yaml"), Loader=yaml.SafeLoader)
        assert "Liquid X-Velocity [m_per_y]" in self.norm_info["Inputs"], "Velocity-x not in Inputs"
        self.norm = NormalizeTransform(self.norm_info)
        self.device = device
        self.data_type = data_type
        self.half_precision = half_precision
        self.kernel = torch.tensor([[-1,0,1],
                                    [0,0,0],
                                    [1,0,-1]],dtype=data_type, device=self.device).unsqueeze(0).unsqueeze(0)
        self.keep_dim = keep_dim
        self.norm_factor = 1e-16

    def forward(self, prediction, inputs):
        inputs_full = deepcopy(inputs[0].detach()).to(torch.float32)
        self.inputs_unnormed = self.norm.reverse(deepcopy(inputs_full), "Inputs")
        self.pressure = self.inputs_unnormed[self.norm_info["Inputs"]["Liquid Pressure [Pa]"]["index"]].to(self.device)
        self.vx = self.inputs_unnormed[self.norm_info["Inputs"]["Liquid X-Velocity [m_per_y]"]["index"]].to(self.device)
        self.vy = self.inputs_unnormed[self.norm_info["Inputs"]["Liquid Y-Velocity [m_per_y]"]["index"]].to(self.device)
        self.ids_normed = inputs[0][:,self.norm_info["Inputs"]["Material ID"]["index"]].to(self.device)
        predicted_T_unnormed = self.norm.reverse(deepcopy(prediction.detach().to(torch.float32).requires_grad_(True)), "Labels").squeeze()
        # TODO dimensions! expect 2D
        loss = energy_loss(self.pressure, predicted_T_unnormed, self.vx, self.vy, self.ids_normed, self.mse_loss, self.kernel, data_type=self.data_type, half_precision=self.half_precision, device=self.device, keep_dim=self.keep_dim)
        return loss * self.norm_factor

def energy_loss(pressure, predicted_temperature, vx, vy, ids_normed, mse_loss, kernel, data_type=torch.float32, half_precision=False, device:str='cuda:0', keep_dim:bool=False):
    #  based on : Manuel Hirche, Bachelor thesis, 2023
    resolution = 5. #m
    # cond_dry = 0.65
    # cond_sat = 1.0
    # sl  = 1 #? saturation of liquid?
    thermal_conductivity = 1 #cond_dry + torch.sqrt(sl) * (cond_sat - cond_dry)
    # Calculate density, molar_density, and enthalpy
    density, molar_density = eos_water_density_IFC67(predicted_temperature, pressure)
    enthalpy = eos_water_enthalphy(predicted_temperature, pressure)
    # Calculate temperature gradients
    T_grad = torch.gradient(predicted_temperature, dim=(1,2))
    # Calculate energy components
    energy_u = torch.gradient((molar_density * vx * enthalpy) - (thermal_conductivity * T_grad[0]/resolution), dim=(1,2))[0]/resolution
    energy_v = torch.gradient((molar_density * vy * enthalpy) - (thermal_conductivity * T_grad[1]/resolution), dim=(1,2))[1]/resolution
    energy = energy_u + energy_v

    # Calculate inflow energy
    inflow_energy = energy_hps(ids_normed, resolution, density, kernel, data_type=data_type, half_precision=half_precision, device=device)
    energy -= inflow_energy #*0.5

    # Calculate energy loss
    if not keep_dim:
        energy_loss = mse_loss(energy, torch.zeros_like(energy))
    else:
        energy_loss = torch.nn.MSELoss(reduction="none")(energy, torch.zeros_like(energy))
    return energy_loss

def energy_hps(ids, resolution, density, kernel, data_type=torch.float32, half_precision=False, device:str='cuda:0'):

    with (torch.autocast(device_type=device, dtype=data_type) if half_precision else contextlib.nullcontext()):
        specific_heat_water = 4200 # [J/kgK]
        density_water = density # [kg/m^3]
        temp_diff = 5 # [K]
        volumetric_flow_rate = 0.00024 # [m^3/s]

        hp_energy = specific_heat_water * density_water * temp_diff * volumetric_flow_rate * 1/resolution**3
        hp_energy = hp_energy * ids
        
        hp_energy = torch.nn.functional.conv2d(hp_energy.unsqueeze(1), kernel, padding=1)

    return (hp_energy[:,0])