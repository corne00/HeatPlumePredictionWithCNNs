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

    
    def forward(self, prediction, target):
        # Calculate the element-wise maximum between prediction and target
        if self.only_target_based:
            weight = target + self.epsilon
        else:
            weight = torch.max(prediction, target) + self.epsilon
        
        # Compute the squared error
        mse = (prediction - target) ** 2
        
        # Return the weighted mean squared error
        return weight * mse

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

    
    def forward(self, prediction, target):
        # Calculate the element-wise maximum between prediction and target
        if self.only_target_based:
            weight = target + self.epsilon
        else:
            weight = torch.max(prediction, target) + self.epsilon
        
        # Return the weighted mean absolute error
        return weight * torch.nn.L1Loss(reduction='none')(prediction, target)

class ThresholdedMAELoss(nn.Module):
    """
    Function that puts more weight on pixels close to the stream lines.
    """
    def __init__(self, threshold:float=0.02, weight_ratio=0.1):
        super(ThresholdedMAELoss, self).__init__()
        self.threshold = threshold
        self.weight_ratio = weight_ratio
        self.name = rf"ThresholdedMAE (t={threshold}, w={weight_ratio})"

    def forward(self, prediction, target):
        # Calculate the element-wise maximum between prediction and target
        weight = torch.where(target > self.threshold, 1., self.weight_ratio)
                
        # Return the weighted mean absolute error
        weighted_mae = weight * torch.nn.L1Loss(reduction='none')(prediction, target)

        return weighted_mae
    
class CombiLoss(nn.Module):
    """
    Loss function that combines MSE and MAE loss with a certain ratio alpha
    """
    def __init__(self, alpha: float = 1):
        super(CombiLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')  # Set to 'none' for pixel-wise computation
        self.mae = nn.L1Loss(reduction='none')   # Set to 'none' for pixel-wise computation
        self.alpha = alpha
        self.name = rf"CombiLoss (a={alpha})"

    def forward(self, x, y):
        return self.alpha * self.mse(x, y) + (1 - self.alpha) * self.mae(x, y)

class CombiRMSE_and_MAELoss(nn.Module):
    """
    Loss function that combines MSE and MAE loss with a certain ratio alpha
    """
    def __init__(self):
        super(CombiRMSE_and_MAELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')  # Set to 'none' for pixel-wise computation
        self.mae = nn.L1Loss(reduction='none')   # Set to 'none' for pixel-wise computation
        self.name = rf"CombiLoss RMSE and MAE)"

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y)) + self.mae(x, y)
        
class PixelwiseMSELoss(nn.MSELoss):
    """
    Computation loss of the MSE loss with option for pixel loss (to see the local errors better)
    """
    def __init__(self):
        super(PixelwiseMSELoss, self).__init__(reduction='none')  # Set reduction to 'none'
        self.name = "MSELoss"
    def forward(self, input, target):
        return super(PixelwiseMSELoss, self).forward(input, target)

class PixelwiseRMSELoss(nn.MSELoss):
    """
    Computation loss of the MSE loss with option for pixel loss (to see the local errors better)
    """
    def __init__(self):
        super(PixelwiseRMSELoss, self).__init__(reduction='none')  # Set reduction to 'none'
        self.name = "RMSELoss"

    def forward(self, input, target):
        return torch.sqrt(super(PixelwiseRMSELoss, self).forward(input, target))
        
class PixelwiseL1Loss(nn.L1Loss):
    """
    Computation loss of the MAE loss with option for pixel loss (to see the local errors better)
    """
    def __init__(self):
        super(PixelwiseL1Loss, self).__init__(reduction='none')  # Set reduction to 'none'
        self.name = "MAELoss"
        
    def forward(self, input, target):
        # Calculate the element-wise absolute error
        return super(PixelwiseL1Loss, self).forward(input, target)


if __name__ == "__main__":
    # Dummy data for prediction and target
    prediction = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.5, 1.8, 2.5, 0.0, 0.0, 0.1])
    
    # --- WeightedMSELoss ---
    wmse_loss = WeightedMSELoss(epsilon=1e-1)
    loss_wmse_pixelwise = wmse_loss(prediction, target)
    print(f"WeightedMSE Loss (pixel-wise): {loss_wmse_pixelwise}")
    
    # --- ThresholdedMSELoss ---
    thresholded_mse_loss = ThresholdedMAELoss(threshold=0.02, weight_ratio=0.1)
    loss_thresholded_pixelwise = thresholded_mse_loss(prediction, target)
    print(f"Thresholded MSE Loss (pixel-wise): {loss_thresholded_pixelwise}")
    
    # --- CombiLoss (combined MSE and MAE) ---
    combi_loss = CombiLoss(alpha=0.5)
    loss_combi_pixelwise = combi_loss(prediction, target)
    print(f"Combi Loss (pixel-wise): {loss_combi_pixelwise}")
    
    # --- PixelwiseMSELoss ---
    pixelwise_mse_loss = PixelwiseMSELoss()
    loss_pixelwise_mse_pixelwise = pixelwise_mse_loss(prediction, target)
    print(f"Pixelwise MSE Loss (pixel-wise): {loss_pixelwise_mse_pixelwise}")
    
    # --- PixelwiseL1Loss ---
    pixelwise_l1_loss = PixelwiseL1Loss()
    loss_pixelwise_l1_pixelwise = pixelwise_l1_loss(prediction, target)
    print(f"Pixelwise L1 Loss (pixel-wise): {loss_pixelwise_l1_pixelwise}")
