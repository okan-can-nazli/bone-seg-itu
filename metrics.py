import torch
from scipy.spatial.distance import cdist
import numpy as np



def dice_score(pred, target, smooth=1e-9): # smooth: prevents 0/0 condition
    pred = (torch.sigmoid(pred) > 0.5).float()  # prediction is 0 OR 1
    intersection = torch.sum(pred * target)
    dice = (2 * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    return dice 


# boundry error metric: ignore the worst %5 point of distances after that return the highest point distance
def hd95(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float().cpu().numpy() # np doesnt work on gpu
    target = target.cpu().numpy()

    # list of bones coordinates
    pred_points = np.argwhere(pred == 1)
    target_points = np.argwhere(target == 1)

    if len(pred_points) == 0 or len(target_points) == 0:
        return 0.0

    all_distances = cdist(pred_points, target_points) # distances between each pred_point and target_point
    d1 = all_distances.min(axis=1)  # pred → target: unexisted-bone prediction
    d2 = all_distances.min(axis=0)  # target → pred: missed real bones

    return float(np.percentile(np.concatenate([d1, d2]), 95))