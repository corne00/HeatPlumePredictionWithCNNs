import torch

from utils.losses import WeightedMAELoss, ThresholdedMAELoss, CombiRMSE_and_MAELoss, CombiLoss, WeightedMSELoss

def test_CombiRMSE_andMAELoss():
    # Fixture ("inputs")
    prediction = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.5, 1.8, 2.5, 0.0, 0.0, 0.1])
    
    # Expected result
    expected = 0.519

    # Actual result
    combi_rmse_mae_loss = CombiRMSE_and_MAELoss()
    loss_combi_rmse_mae = combi_rmse_mae_loss(prediction, target)
    
    # Test 
    assert torch.isclose(loss_combi_rmse_mae, torch.tensor(expected), atol=1e-3)

def test_WeightedMSELoss():
    # Fixture ("inputs")
    prediction = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.5, 1.8, 2.5, 0.0, 0.0, 0.1])
    
    # Expected result
    expected = 0.1100

    # Actual result
    wmse_loss = WeightedMSELoss(epsilon=1e-1)
    loss_wmse = wmse_loss(prediction, target)
    
    # Test 
    assert torch.isclose(loss_wmse, torch.tensor(expected), atol=1e-3)

def test_WeightedMAELoss():
    # Fixture ("inputs")
    prediction = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.5, 1.8, 2.5, 0.0, 0.0, 0.1])
    
    # Expected result
    expected = 0.4650

    # Actual result
    wmae_loss = WeightedMAELoss(epsilon=1e-1)
    loss_wmae = wmae_loss(prediction, target)
    
    # Test 
    assert torch.isclose(loss_wmae, torch.tensor(expected), atol=1e-3)

def test_ThresholdedMSELoss():
    # Fixture ("inputs")
    prediction = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.5, 1.8, 2.5, 0.0, 0.0, 0.1])
    
    # Expected result
    expected = 0.1517

    # Actual result
    thresholded_mse_loss = ThresholdedMAELoss(threshold=0.02, weight_ratio=0.1)
    loss_thresholded = thresholded_mse_loss(prediction, target)
    
    # Test 
    assert torch.isclose(loss_thresholded, torch.tensor(expected), atol=1e-3)

def test_CombiLoss():
    # Fixture ("inputs")
    prediction = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.5, 1.8, 2.5, 0.0, 0.0, 0.1])
    
    # Expected result
    expected = 0.1542

    # Actual result
    combi_loss = CombiLoss(alpha=0.5)
    loss_combi = combi_loss(prediction, target)
    
    # Test 
    assert torch.isclose(loss_combi, torch.tensor(expected), atol=1e-3)

def test_PixelwiseMSELoss():
    # Fixture ("inputs")
    prediction = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.5, 1.8, 2.5, 0.0, 0.0, 0.1])
    
    # Expected result
    expected = 0.0917

    # Actual result
    pixelwise_mse_loss = torch.nn.MSELoss()
    loss_pixelwise_mse = pixelwise_mse_loss(prediction, target)
    
    # Test 
    assert torch.isclose(loss_pixelwise_mse, torch.tensor(expected), atol=1e-3)

def test_PixelwiseL1Loss():
    # Fixture ("inputs")
    prediction = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.5, 1.8, 2.5, 0.0, 0.0, 0.1])
    
    # Expected result
    expected = 0.2167

    # Actual result
    pixelwise_l1_loss = torch.nn.L1Loss()
    loss_l1 = pixelwise_l1_loss(prediction, target)
    
    # Test 
    assert torch.isclose(loss_l1, torch.tensor(expected), atol=1e-3)

if __name__ == "__main__":
    test_CombiRMSE_andMAELoss()
    test_WeightedMSELoss()
    test_WeightedMAELoss()
    test_ThresholdedMSELoss()
    test_CombiLoss()
    test_PixelwiseMSELoss()
    test_PixelwiseL1Loss()