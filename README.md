# Bone Segmentation in X-Ray Images

Binary semantic segmentation of bones in X-ray images using U-Net with a pretrained ResNet34 encoder.

## Results

| Fold | Dice ↑ | HD95 ↓ (px) |
|------|--------|-------------|
| 1    | 0.9418 | 3.52        |
| 2    | 0.9425 | 2.67        |
| 3    | 0.9368 | 8.36        |
| 4    | 0.9424 | 2.83        |
| 5    | 0.9324 | 4.63        |
| **Mean** | **0.9392 ± 0.0040** | **4.40 ± 2.09** |

## Architecture

- **Model:** U-Net with ResNet34 encoder (ImageNet pretrained)
- **Library:** segmentation-models-pytorch
- **Input:** 3-channel RGB, 512×512 px
- **Output:** 1-channel binary mask (bone=1, background=0)
- **Parameters:** 24,436,369 trainable

## Dataset

- 499 X-ray images (500 total, 1 skipped — patient 283 contains .gif)
- Multiple per-bone .npy masks merged into a single binary mask at runtime
- Image formats: .jpg, .jpeg, .png

## Training

- **Loss:** BCE + Dice combined (0.5 weight each)
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR (T_max=50, eta_min=1e-6)
- **Epochs:** 50 per fold
- **Batch size:** 8 (train) / 1 (validation)
- **Augmentations:** HorizontalFlip, RandomRotate90, ShiftScaleRotate
- **Platform:** Kaggle — GPU T4 x2

## Project Structure

```
bone-seg-itu/
├── dataset.py           # Dataset class + file list builder
├── unet.py              # U-Net model (segmentation-models-pytorch)
├── losses.py            # BCE + Dice combined loss
├── metrics.py           # Dice score + HD95 (Hausdorff Distance)
├── cross_validation.py  # 5-fold CV training — main script
└── inference.py         # Best & worst prediction visualization
```

## How to Run

### On Kaggle

1. Upload dataset to Kaggle
2. Clone this repo in a notebook cell:
```bash
!git clone https://github.com/okan-can-nazli/bone-seg-itu.git
%cd bone-seg-itu
!pip install segmentation-models-pytorch
!python cross_validation.py
```

### Local

Update the directory paths in `cross_validation.py`:
```python
IMAGE_DIR = "path/to/New_Labels"
MASK_DIR  = "path/to/New_masks"
OUTPUT_DIR = "./outputs"
```

Then:
```bash
pip install segmentation-models-pytorch albumentations torch torchvision
python cross_validation.py
```

## Key Design Decisions

- **Pretrained encoder:** ImageNet weights on ResNet34 enable fast convergence on medical images
- **BCE+Dice loss:** BCE stabilizes early training, Dice forces accurate overlap
- **HD95 over HD100:** 95th percentile is more robust to boundary outliers
- **val batch_size=1:** Ensures accurate per-sample Dice and HD95 computation
