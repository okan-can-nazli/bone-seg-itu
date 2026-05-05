import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 # image processing lib

#! one X-ray → multiple per-bone masks → merge into single binary mask


#! patient 283's image folder contains .gif file 



# image_paths → ["/path/1/1.jpg", "/path/2/2.jpg", ...] — file
# mask_folders → ["/path/1/", "/path/2/", ...] - folder
# transform provides augmentation
def __init__(self, image_paths, mask_folders, transform=None): 
    self.image_paths = image_paths
    self.mask_folders = mask_folders
    self.transform = transform
    
def __len__(self):
    return len(self.image_paths) # 499

def __getitem__(self,idx):
    
    # get & set image
    image = cv2.imread(self.image_paths[idx]) # format : (H,W,3) , BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB
    image = cv2.resize(image, (512, 512))
    
    # get & set mask
    mask = load_merged_mask(self.mask_folders[idx])
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST) # mask MUST contain only 0 OR 1
    
    
    if self.transform is not None:
        augmented = self.transform(image=image, mask=mask) # flip/rotate (random)
        image = augmented["image"]
        mask = augmented["mask"]
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # numpy (H,W,3) → tensor (3,H,W), normalize (0,1)
        # uint8 standart (255)
        
        mask = torch.from_numpy(mask).float() # numpy (H,W) → tensor (H,W)
        
        return image, mask
    
def load_merged_mask():