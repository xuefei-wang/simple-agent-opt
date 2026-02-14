"""
MedSAM data preparation utility.

Downloads and prepares the MedSAM validation data for agentic optimization
and evaluation.

## How to Use

### 1. Download the Data

Download and place the following files in a `data/` directory:

- imgs.zip: https://drive.google.com/file/d/1XCmJMbtB0ICBSzknrsCk_RsjFdrituGN/view?usp=drive_link
  Contains `.npz` files with `"imgs"` and `"bbox"` keys.

- gts.zip: https://drive.google.com/file/d/1BrPXFwf7ncp4vsv0TXtxtj09ICfjQm9J/view?usp=drive_link
  Contains `.npz` files with ground truth segmentation masks.

These files come from the validation set of the MedSAM CVPR Codabench
competition: https://www.codabench.org/competitions/1847/

### 2. Prepare the Data

Unzip both files inside the `data/` directory.

### 3. Run this script

    python -m utils.medsam_data --data_dir /path/to/data

The script will:
- Select a subset of 2D images per modality:
  - 66 dermoscopy images
  - 379 X-ray images
- Randomly shuffle images with their bounding boxes and masks
- Split into:
  - Validation set (for agentic optimization): 25 images per modality
  - Test set (for evaluation): 33 dermoscopy, 190 X-ray
"""

import os
import pickle
import argparse

import numpy as np
from skimage import transform


def _get_binary_masks(nonbinary_mask):
    """
    Given nonbinary mask which encodes N masks, return N binary masks which
    should encode the same information.

    Parameters:
        nonbinary_mask: ndarray of shape (H, W)

    Returns:
        binary_masks: ndarray of shape (N, H, W)
    """
    binary_masks = []
    for i in np.unique(nonbinary_mask)[1:]:
        binary_mask = (nonbinary_mask == i).astype(np.uint8)
        binary_masks.append(binary_mask.copy())
    binary_masks = np.stack(binary_masks, axis=0)
    return binary_masks


def get_data(data_dir, modality, exp_type):
    """Prepare resized data for a given modality and experiment type.

    Args:
        data_dir: Path to the directory containing imgs/ and gts/ subdirectories.
        modality: One of "xray" or "dermoscopy".
        exp_type: Experiment split, e.g. "val_filenames_5", "val_filenames_25",
                  "test_filenames_25".
    """
    query_name = "X-Ray" if modality == "xray" else "Dermoscopy"
    imgs_2d_and_3d = os.listdir(os.path.join(data_dir, "imgs"))
    imgs_2d = [f for f in imgs_2d_and_3d if f.startswith("2D")]
    imgs_2d_modality = [f for f in imgs_2d if query_name in f]
    print(f"{len(imgs_2d_modality)} images in {modality} modality")

    np.random.seed(42)
    split_resulting_len = len(imgs_2d_modality) // 2
    val_filenames_bank = np.random.choice(
        imgs_2d_modality, size=split_resulting_len, replace=False
    )
    test_filenames_bank = [
        f for f in imgs_2d_modality if f not in val_filenames_bank
    ]

    file_str = f"{modality}_{exp_type}"
    print("Starting experiment", file_str)

    if exp_type.startswith("val"):
        filebank = val_filenames_bank
    else:
        filebank = test_filenames_bank
    sample_size = int(exp_type.split("_")[-1])

    # ========================== Unpack ==========================
    print(f"\nUnpacking {len(filebank)} images...")
    val_raw_imgs, val_raw_boxes, val_raw_gts = [], [], []
    for i, img_filename in enumerate(filebank):
        img_data = np.load(os.path.join(data_dir, f"imgs/{img_filename}"))
        mask_data = np.load(os.path.join(data_dir, f"gts/{img_filename}"))

        image, boxes, nonbinary_mask = (
            img_data["imgs"],
            img_data["boxes"],
            mask_data["gts"],
        )
        num_masks = 0
        for box, mask in zip(boxes, _get_binary_masks(nonbinary_mask)):
            x1, y1, x2, y2 = box
            box_string = f"[{x1},{y1},{x2},{y2}]"
            val_raw_imgs.append(image)
            val_raw_boxes.append(box_string)
            val_raw_gts.append(mask)
            num_masks += 1

        print(f"Processed idx {i}: {img_filename} -> {num_masks} masks")
    print(f"Finished unpacking images into {len(val_raw_imgs)}.\n")

    # ========================== Resize ==========================
    print("Resizing images...")
    random_5_indices = np.random.choice(
        len(val_raw_imgs), size=sample_size, replace=False
    )
    imgs_to_resize = [val_raw_imgs[i] for i in random_5_indices]
    boxes_to_resize = [val_raw_boxes[i] for i in random_5_indices]
    gts_to_use = [val_raw_gts[i] for i in random_5_indices]

    resized_imgs, resized_boxes = [], []
    resized_gts = gts_to_use
    for i, (img_np, box_str) in enumerate(zip(imgs_to_resize, boxes_to_resize)):
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np

        H, W, _ = img_3c.shape

        # Resize image to 1024x1024
        img_1024 = transform.resize(
            img_3c,
            (1024, 1024),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)

        img_1024 = img_1024 / 255.0
        resized_imgs.append(img_1024)

        # Scale box to 1024x1024
        box_np = np.array([[int(x) for x in box_str[1:-1].split(",")]])
        box_scaled = box_np / np.array([W, H, W, H]) * 1024
        resized_boxes.append(box_scaled)

        print(
            f"file {i} | og img shape {img_np.shape} | box_str {box_str} -> {box_scaled.shape}"
        )

    resized_filepath = os.path.join(data_dir, f"resized_{file_str}.pkl")
    os.makedirs(os.path.dirname(resized_filepath), exist_ok=True)
    with open(resized_filepath, "wb") as f:
        pickle.dump((resized_imgs, resized_boxes, resized_gts), f)
    print(f"Saved resized data to {resized_filepath}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MedSAM data for agentic optimization"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing imgs/ and gts/ subdirectories",
    )
    args = parser.parse_args()

    for modality in ["xray", "dermoscopy"]:
        for exp_type in [
            "val_filenames_5",
            "test_filenames_25",
            "val_filenames_25",
        ]:
            get_data(args.data_dir, modality, exp_type)


if __name__ == "__main__":
    main()
