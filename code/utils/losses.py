import torch
import torch.nn as nn

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
