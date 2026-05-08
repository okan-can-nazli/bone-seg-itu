import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 # image processing lib


#! patient 283's image folder contains .gif file 

class BoneSegDataset(Dataset):


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
        mask = load_merged_mask(self.mask_folders[idx]) #! one X-ray → multiple per-bone masks → merge into single binary mask

        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST) # mask MUST contain only 0 OR 1
        
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"].float()  # Normalize already handled /255
            mask = augmented["mask"].float()
        
        else:
            
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # numpy (H,W,3) → tensor (3,H,W), normalize (0,1)
            # uint8 standart (255)
            
            mask = torch.from_numpy(mask).float() # numpy (H,W) → tensor (H,W)
            
        return image, mask
        
        
        
########################################################################################################################################
        
        
        
def load_merged_mask(mask_folder):
    npy_files = []
    for sub in os.listdir(mask_folder):
        sub_path = os.path.join(mask_folder, sub)
        if os.path.isdir(sub_path):
            for f in os.listdir(sub_path):
                if f.endswith(".npy"):
                    npy_files.append(os.path.join(sub_path, f))
            
    if not npy_files: # prevention for inconsistent data
        raise ValueError(f"No masks found in {mask_folder}")

    masks = []
    for path in npy_files:
        masks.append(np.load(path))
        
    stacked = np.stack(masks)           # (N, 512, 512)
    merged = np.any(stacked, axis=0).astype(np.uint8)  # (512, 512), 0 OR 1
    
    return merged

def build_file_lists(images_dir, masks_dir):
    images_ls = []
    masks_ls = []
    
    # sorted() provides index matching between iimages and masks (based on index)
    
    for sub_folder in sorted([f for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f)) and f.isdigit()], key=lambda x: int(x)):
        sub_path = os.path.join(images_dir, sub_folder)
        for f in os.listdir(sub_path):
            if f.lower().endswith(".jpg"):
                images_ls.append(os.path.join(sub_path, f))
                
    for sub_folder in sorted([f for f in os.listdir(masks_dir) if os.path.isdir(os.path.join(masks_dir, f)) and f.isdigit()], key=lambda x: int(x)):        
        sub_path = os.path.join(masks_dir, sub_folder)
        masks_ls.append(sub_path)
    
    return images_ls, masks_ls