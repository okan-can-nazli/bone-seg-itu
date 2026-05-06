import torch


def dice_score(pred, target, smooth=1e-9): # smooth: prevents 0/0 condition
    pred = (torch.sigmoid(pred) > 0.5).float()  # prediction is 0 OR 1
    intersection = torch.sum(pred * target)
    dice = (2 * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    return dice 
