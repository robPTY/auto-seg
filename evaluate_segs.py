#
# Script to evaluate two separate segmentations against the ground truth (GT).
#
# usage: python evaluate_segs.py --m1_path <mask.tif> --m2_path <mask.tif> --gt_path <gt.tif>
#
import argparse
import numpy as np
import tifffile as tiff

def iou(pred, target) -> float:
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return intersection / union if union > 0 else 0

def dice(pred, target) -> float:
    intersection = np.sum(pred * target)
    return 2 * intersection / (np.sum(pred) + np.sum(target)) if (np.sum(pred) + np.sum(target)) > 0 else 0

def main(args) -> int:
    m1 = tiff.imread(args.m1_path)
    m2 = tiff.imread(args.m2_path)
    gt = tiff.imread(args.gt_path)

    m1_flat = m1.flatten()
    m2_flat = m2.flatten()
    gt_flat = gt.flatten()

    m1_flat = (m1_flat > 0.5).astype(np.float64)
    m2_flat = (m2_flat > 0.5).astype(np.float64)
    gt_flat = (gt_flat > 0.5).astype(np.float64)


    print("Evaluating m1 against GT:")
    iou_score = iou(m1_flat, gt_flat)
    dice_score = dice(m1_flat, gt_flat)
    print(f"IoU: {iou_score}, Dice: {dice_score}")

    print("Evaluating m2 against GT:")
    iou_score = iou(m2_flat, gt_flat)
    dice_score = dice(m2_flat, gt_flat)
    print(f"IoU: {iou_score}, Dice: {dice_score}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--m1_path", 
        type=str,
        required=True,
        help="Path to the first segmentation mask (TIFF format)"
    )

    parser.add_argument(
        "--m2_path", 
        type=str,
        required=True,
        help="Path to the second segmentation mask (TIFF format)"
    )

    parser.add_argument(
        "--gt_path",
        type=str,
        required=True,
        help="Path to the ground truth segmentation mask (TIFF format)"
    )

    args = parser.parse_args()

    main(args) 
