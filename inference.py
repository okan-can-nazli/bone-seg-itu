import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as Augment
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold

from dataset import build_file_lists, load_merged_mask
from unet import get_model


def visualize_predictions(output_dir, image_dir, mask_dir, best_fold_idx):
    """
    best_fold_idx: 0-indexed fold number (e.g. fold 1 → 0)
    """
    os.makedirs(output_dir, exist_ok=True)

    image_paths, mask_folders = build_file_lists(image_dir, mask_dir)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = Augment.Compose([
        Augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Best fold'un val setini al
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        if fold == best_fold_idx:
            val_images = [image_paths[i] for i in val_idx]
            val_masks  = [mask_folders[i] for i in val_idx]
            break

    # Model yükle
    model = get_model().to(device)
    model.load_state_dict(torch.load(
        os.path.join(output_dir, f"fold{best_fold_idx+1}_best.pth"),
        map_location=device))
    model.eval()

    # Her sample için Dice hesapla
    sample_dices = []
    with torch.no_grad():
        for idx in range(len(val_images)):
            img = cv2.imread(val_images[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (512, 512))

            mask = load_merged_mask(val_masks[idx])
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

            aug = val_transform(image=img_resized)
            img_tensor = aug["image"].unsqueeze(0).to(device)
            pred = torch.sigmoid(model(img_tensor).squeeze()).cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8)

            intersection = (pred_bin * mask).sum()
            dice = (2 * intersection + 1e-9) / (pred_bin.sum() + mask.sum() + 1e-9)
            sample_dices.append((idx, float(dice), img_resized, mask, pred_bin))

    # En iyi 3 + en kötü 3
    sample_dices.sort(key=lambda x: x[1])
    worst = sample_dices[:3]
    best  = sample_dices[-3:][::-1]
    selected = best + worst
    labels = ['Best 1', 'Best 2', 'Best 3', 'Worst 1', 'Worst 2', 'Worst 3']

    # Prediction visualization
    fig, axes = plt.subplots(6, 3, figsize=(12, 24))
    fig.suptitle(f"Bone Segmentation — Best & Worst Predictions (Fold {best_fold_idx+1})",
                 fontsize=14, fontweight='bold')

    for row, (idx, dice, img_resized, mask, pred_bin) in enumerate(selected):
        axes[row, 0].imshow(img_resized, cmap='gray')
        axes[row, 0].set_title(f"{labels[row]} | Dice: {dice:.4f}")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(img_resized, cmap='gray')
        axes[row, 1].imshow(mask, alpha=0.5, cmap='Reds')
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis('off')

        axes[row, 2].imshow(img_resized, cmap='gray')
        axes[row, 2].imshow(pred_bin, alpha=0.5, cmap='Blues')
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictions_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: predictions_visualization.png")

    del model
    torch.cuda.empty_cache()


def save_results_chart(output_dir, results):
    """
    results: list of dicts with 'fold', 'dice', 'hd95'
    """
    folds = [f"Fold {r['fold']}" for r in results]
    dices = [r['dice'] for r in results]
    hd95s = [r['hd95'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(folds, dices, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.axhline(np.mean(dices), color='red', linestyle='--', label=f'Mean: {np.mean(dices):.4f}')
    for i, v in enumerate(dices):
        ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax1.set_ylim([0.88, 1.0])
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Dice Score per Fold')
    ax1.legend()

    ax2.bar(folds, hd95s, color='coral', alpha=0.8, edgecolor='black')
    ax2.axhline(np.mean(hd95s), color='red', linestyle='--', label=f'Mean: {np.mean(hd95s):.2f}px')
    for i, v in enumerate(hd95s):
        ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('HD95 (pixels)')
    ax2.set_title('HD95 per Fold')
    ax2.legend()

    plt.suptitle('5-Fold Cross Validation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: results_chart.png")