#Converts PNG X-rays + NPY masks to nnU-Net format.


import os
import sys
import json
import numpy as np
import cv2

from Unet.dataset import build_file_lists, load_merged_mask

#########################################################################################################
#! Constants

DATASET_ID   = "001"
DATASET_NAME = "BoneSeg"

#! Directories

# LOCAL DIRS
IMAGE_DIR  = "../Data/images"
MASK_DIR   = "../Data/masks"
OUTPUT_DIR = "nnunet_raw"

# KAGGLE DIRS
# IMAGE_DIR  = "/kaggle/input/datasets/okancannazli/bones-seg/New_Labels-20260504T191710Z-3-001/New_Labels"
# MASK_DIR   = "/kaggle/input/datasets/okancannazli/bones-seg/New_masks-20260504T191902Z-3-001/New_masks"
# OUTPUT_DIR = "/kaggle/working/nnunet_raw"
#########################################################################################################


def main():

    images_out = os.path.join(OUTPUT_DIR, "imagesTr")
    labels_out = os.path.join(OUTPUT_DIR, "labelsTr")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    image_paths, mask_folders = build_file_lists(IMAGE_DIR, MASK_DIR)
    print(f"Converting {len(image_paths)} samples...")

    case_ids = []

    for idx, (img_path, mask_folder) in enumerate(zip(image_paths, mask_folders)):
        case_id = f"bone_{idx:04d}"
        case_ids.append(case_id)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join(images_out, f"{case_id}_0000.png"), img_gray)

        mask = load_merged_mask(mask_folder)
        cv2.imwrite(os.path.join(labels_out, f"{case_id}.png"), mask.astype(np.uint8))

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(image_paths)} done")

    dataset_json = {
        "channel_names": {"0": "X-Ray"},
        "labels": {"background": 0, "bone": 1},
        "numTraining": len(case_ids),
        "file_ending": ".png"
    }

    with open(os.path.join(OUTPUT_DIR, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDone. {len(case_ids)} samples saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()