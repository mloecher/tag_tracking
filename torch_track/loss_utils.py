import torch
import torch.nn as nn

#Custom loss functions, no of these ended up being used, can delete

def custom_mse_loss(test, target):
    return ((test - target) ** 2).sum() / test.data.nelement()

def weighted_mse_loss(test, target):
    test = torch.reshape(test, [test.shape[0], 2, -1])
    target = torch.reshape(target, [target.shape[0], 2, -1])

    # Get the displacement magnitude of truth
    hypot = torch.hypot(target[:, 0], target[:, 1])
    
    # Make a weighting vector that ranges from 1-2
    max_hypot = torch.max(hypot, 1)[0]
    scaler = hypot / max_hypot[:, None]
    scaler += 1.0

    # Weight the difference term
    diff = (test - target)
    diff *= scaler[:, None, :]

    # Return MSE
    return (diff ** 2).sum() / test.data.nelement()

def multi_loss(test, target):
    loss_MSE = nn.MSELoss()(test, target)
    loss_MAE = nn.L1Loss()(test, target)
    loss_custom = custom_mse_loss(test, target)
    loss_weighted = weighted_mse_loss(test, target)

    return loss_MSE, loss_MAE, loss_custom, loss_weighted

def two_loss(test, target):
    loss_MSE = nn.MSELoss()(test, target)
    loss_MAE = nn.L1Loss()(test, target)

    return loss_MSE, loss_MAE
