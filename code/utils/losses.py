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

    
    def forward(self, prediction, target):
        # Calculate the element-wise maximum between prediction and target
        if self.only_target_based:
            weight = target + self.epsilon
        else:
            weight = torch.max(prediction, target) + self.epsilon
        
        # Compute the squared error
        mse = (prediction - target) ** 2
        
        # Return the weighted mean squared error
        weighted_mse = weight * mse
        return weighted_mse.mean()

class ThresholdedMSELoss(nn.Module):
    def __init__(self):
        super(ThresholdedMSELoss, self).__init__()
    
    def forward(self, prediction, target):
        # Calculate the element-wise maximum between prediction and target
        weight = torch.where(target > 0.02, 1, 0.1)
        
        # Compute the squared error
        mse = (prediction - target) ** 2
        
        # Return the weighted mean squared error
        weighted_mse = weight * mse
        return weighted_mse.mean()
    
# Example usage
if __name__ == "__main__":
    # Dummy data for prediction and target
    prediction = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.5, 1.8, 2.5, 0.0, 0.0, 0.1])
    
    # Create an instance of the custom WMSE loss
    wmse_loss = WeightedMSELoss(epsilon=1e-1)
    
    # Calculate the WMSE loss
    loss = wmse_loss(prediction, target)
    
    print(f"WMSE Loss: {loss.item()}")
