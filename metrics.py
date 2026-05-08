import torch
from scipy.spatial.distance import cdist
import numpy as np



def dice_score(pred, target, smooth=1e-9): # smooth: prevents 0/0 condition
    pred = (torch.sigmoid(pred) > 0.5).float()  # prediction is 0 OR 1
    intersection = torch.sum(pred * target)
    dice = (2 * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    return dice 


# boundry error metric: ignore the worst %5 point of distances after that return the highest point distance
from scipy.ndimage import distance_transform_edt

def hd95(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()
    target = target.cpu().numpy()
    
    pred = pred.squeeze()
    target = target.squeeze()
    
    if not pred.any() or not target.any():
        return 0.0
    
    # distance transform — cdist'ten çok daha az bellek kullanır
    pred_dist   = distance_transform_edt(~pred.astype(bool))
    target_dist = distance_transform_edt(~target.astype(bool))
    
    d1 = pred_dist[target.astype(bool)]
    d2 = target_dist[pred.astype(bool)]
    
    return float(np.percentile(np.concatenate([d1, d2]), 95))