import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold # provides train/val index splits into 5-fold
from tqdm import tqdm # training progress bar 
import matplotlib.pyplot as plt

import albumentations as Augment # augmentation lib
from albumentations.pytorch import ToTensorV2 # convert np array into tensor

from dataset import BoneSegDataset, build_file_lists
from unet import get_model
from losses import bce_dice_loss
from metrics import dice_score, hd95

#####################
#! Constants
LEARNING_RATE = 1e-4
EPOCH = 50
#####################


def main():
    
    # LOCAL
    # IMAGE_DIR = "Data/images"
    # MASK_DIR = "Data/masks"
    IMAGE_DIR = "/kaggle/input/datasets/okancannazli/bones-seg/New_Labels-20260504T191710Z-3-001/New_Labels"
    MASK_DIR  = "/kaggle/input/datasets/okancannazli/bones-seg/New_masks-20260504T191902Z-3-001/New_masks"
    OUTPUT_DIR = "/kaggle/working/outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths, mask_folders = build_file_lists(IMAGE_DIR, MASK_DIR)

    kf = KFold(n_splits=5, shuffle=True, random_state=42) # tr:400/val:100 sample each fold

    train_transform = Augment.Compose([
        Augment.HorizontalFlip(p=0.5),
        Augment.RandomRotate90(p=0.5),
        Augment.ShiftScaleRotate(p=0.3),
        Augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = Augment.Compose([
        Augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):  # 400,100
        print(f"\n--- Fold {fold+1}/5 ---")

        # len = 400
        train_images = [image_paths[i] for i in train_idx]
        train_masks  = [mask_folders[i] for i in train_idx]
        
        # len = 100
        val_images   = [image_paths[i] for i in val_idx]
        val_masks    = [mask_folders[i] for i in val_idx]

        train_dataset = BoneSegDataset(image_paths=train_images, mask_folders=train_masks, transform=train_transform)
        val_dataset   = BoneSegDataset(image_paths=val_images, mask_folders=val_masks, transform=val_transform)

        # batches into groups of 8 for training/validation
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = get_model().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_dice = 0.0
        best_path = os.path.join(OUTPUT_DIR, f"fold{fold+1}_best.pth")
        
        #for visualation
        train_losses = []
        val_dices_per_epoch = []

        for epoch in range(EPOCH):
            
            # --- TRAİNİNG ---
            train_loss = 0.0
            model.train()

            
            
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH} Train"):
                images  = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                preds = model(images)
                masks = masks.unsqueeze(1)  # (8,512,512) → (8,1,512,512)
                loss  = bce_dice_loss(preds, masks) # bce dice loss
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # --- VALIDATİON ---
            model.eval()
            val_dices = []
            val_hd95s = []

            with torch.no_grad():
                for images, masks in val_loader:
                    images  = images.to(device)
                    masks = masks.to(device)
                    preds = model(images)
                    masks = masks.unsqueeze(1)  # (8,512,512) → (8,1,512,512)
                    
                    val_dices.append(dice_score(preds, masks).item())
                    val_hd95s.append(hd95(preds, masks))

            mean_dice = np.mean(val_dices)
            mean_hd95 = np.mean(val_hd95s)

            print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Dice: {mean_dice:.4f} | HD95: {mean_hd95:.2f}px")

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Best model saved (Dice={best_dice:.4f})")
                
            train_losses.append(train_loss / len(train_loader))
            val_dices_per_epoch.append(mean_dice)
        
        #visualation

        # Loss/Dice graph
        epochs_range = range(1, EPOCH + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(train_losses, label="Train Loss")
        ax1.set_title(f"Fold {fold+1} - Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax2.plot(val_dices_per_epoch, label="Val Dice", color="green")
        ax2.set_title(f"Fold {fold+1} - Dice Score")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fold{fold+1}_metrics.png"), dpi=300)
        plt.close()
        
        
        
        
        results.append({"fold": fold+1, "dice": best_dice, "hd95": mean_hd95})

    print("\n=== RESULTS ===")
    for r in results:
        print(f"Fold {r['fold']}: Dice={r['dice']:.4f}, HD95={r['hd95']:.2f}px")

    dices = [r["dice"] for r in results]
    hd95s = [r["hd95"] for r in results]
    print(f"\nOverall: Dice={np.mean(dices):.4f} ± {np.std(dices):.4f}, HD95={np.mean(hd95s):.2f} ± {np.std(hd95s):.2f}px")

if __name__ == "__main__":
    main()