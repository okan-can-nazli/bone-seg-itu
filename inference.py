import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import albumentations as Augment
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold

from dataset import build_file_lists, load_merged_mask
from unet import get_model

def run_inference(output_dir, image_dir, mask_dir, n_samples=6):
    
    IMAGE_DIR = image_dir
    MASK_DIR  = mask_dir
    OUTPUT_DIR = output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths, mask_folders = build_file_lists(IMAGE_DIR, MASK_DIR)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = Augment.Compose([
        Augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # fold 1 val seti
    train_idx, val_idx = list(kf.split(image_paths))[0]
    val_images = [image_paths[i] for i in val_idx]
    val_masks  = [mask_folders[i] for i in val_idx]

    model = get_model().to(device)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/fold1_best.pth", map_location=device))
    model.eval()

    random.seed(42)
    samples = random.sample(range(len(val_images)), n_samples)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    fig.suptitle("Bone Segmentation Results\nDice=0.9403 | HD95=4.67px", 
                 fontsize=16, fontweight='bold')

    for row, idx in enumerate(samples):
        img = cv2.imread(val_images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (512, 512))

        mask = load_merged_mask(val_masks[idx])
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        aug = val_transform(image=img_resized)
        img_tensor = aug["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(img_tensor).squeeze()).cpu().numpy()
        pred_bin = (pred > 0.5).astype(np.uint8)

        axes[row, 0].imshow(img_resized, cmap='gray')
        axes[row, 0].set_title("X-Ray Image")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(img_resized, cmap='gray')
        axes[row, 1].imshow(mask, alpha=0.5, cmap='Reds')
        axes[row, 1].set_title("Ground Truth Mask")
        axes[row, 1].axis('off')

        axes[row, 2].imshow(img_resized, cmap='gray')
        axes[row, 2].imshow(pred_bin, alpha=0.5, cmap='Blues')
        axes[row, 2].set_title("Predicted Mask")
        axes[row, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "predictions_visualization.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    run_inference(
        output_dir="/kaggle/working/outputs",
        image_dir="/kaggle/input/datasets/okancannazli/bones-seg/New_Labels-20260504T191710Z-3-001/New_Labels",
        mask_dir="/kaggle/input/datasets/okancannazli/bones-seg/New_masks-20260504T191902Z-3-001/New_masks",
    )