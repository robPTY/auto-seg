#
# Script that takes in as input 3d images, creates three separate views, runs the 3D annotator on each of them, and then merges the resulting segmentation masks into a single 3D segmentation mask.
# Note: pre-computing embeddings for each image is a must unless cached.
#
# example usage: python pipeline.py --input img.tif --output segmentation.tif --fusion_mode union --padding 10'
#
import argparse
import numpy as np
import tifffile as tiff

from micro_sam.util import precompute_image_embeddings, get_sam_model
from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder, TiledInstanceSegmentationWithDecoder

MODEL_TYPE = "vit_l_em_organelles"

def transform_input(input_path: str, padding: int) -> None:
    """Transforms the input 3D image into three separate views (XY, XZ, YZ) and applies padding if specified."""
    vol_zyx = tiff.imread(input_path) # assuming it is (Z, Y, X) given standard conventions
    vol_yzx = vol_zyx.transpose(1, 0, 2) # (Y, Z, X)
    vol_xzy = vol_zyx.transpose(2, 0, 1) # (X, Z, Y)

    if padding:
        vol_zyx = np.pad(vol_zyx, ((padding, padding), (0, 0), (0, 0)), mode='reflect')
        vol_yzx = np.pad(vol_yzx, ((padding, padding), (0, 0), (0, 0)), mode='reflect')
        vol_xzy = np.pad(vol_xzy, ((padding, padding), (0, 0), (0, 0)), mode='reflect')

    assert vol_zyx.shape == vol_yzx.shape == vol_xzy.shape, "The three views must have the same shape after transformation and padding."

    tiff.imwrite("pad_img_zyx.tif" if padding else "img_zyx.tif", vol_zyx)
    tiff.imwrite("pad_img_yzx.tif" if padding else "img_yzx.tif", vol_yzx)
    tiff.imwrite("pad_img_xzy.tif" if padding else "img_xzy.tif", vol_xzy)

    return 

def precompute_embeddings(zyx_path: str, yzx_path: str, xzy_path: str) -> None: 
    """Given the three separate (and possibly padded) views, pre-compute the embeddings for each of them and save them to disk."""

    zyx_output = "zyx_embeddings"
    yzx_output = "yzx_embeddings"
    xzy_output = "xzy_embeddings"

    predictor, _ = get_sam_model(model_type=MODEL_TYPE, return_state=True) # add checkpoint_path to load weights from somewhere if available.

    zyx_embeddings = precompute_image_embeddings(predictor=predictor, input_=zyx_path, save_path=zyx_output)
    yzx_embeddings = precompute_image_embeddings(predictor=predictor, input_=yzx_path, save_path=yzx_output)
    xzy_embeddings = precompute_image_embeddings(predictor=predictor, input_=xzy_path, save_path=xzy_output)

    return

# Note: from an architecture perspective, I could just call the API from micro_sam which has this exact function.
# but gonna keep this here for now.
def run_ais(img_path: str, embedding_path: str) -> None:
    """Run the automatic instance segmentation on an image view, using the pre-computed embeddings."""
    raw_image = tiff.imread(img_path)
    segmenter_class = InstanceSegmentationWithDecoder

def merge_segmentations(zyx_path: str, yzx_path: str, xzy_path: str, padding: int, fusion_mode: str) -> None:
    """Given the three segmentation masks resulting from running the annotator on each of the views, merge them into a single 3D segmentation mask."""
    zyx = tiff.imread(zyx_path) # (Z, Y, X)
    yzx = tiff.imread(yzx_path).transpose(1, 0, 2) # (Y, Z, X) -> (Z, Y, X)
    xzy = tiff.imread(xzy_path).transpose(2, 0, 1) # (X, Z, Y) -> (Z, Y, X)

    # Undo padding if it was applied
    if padding:
        zyx = zyx[padding:-padding, :, :]
        yzx = yzx[:, padding:-padding, :]
        xzy = xzy[:, :, padding:-padding]

    assert zyx.shape == yzx.shape == xzy.shape, "The three segmentation masks must have the same shape after undoing padding."

    m1 = zyx > 0
    m2 = yzx > 0
    m3 = xzy > 0

    if fusion_mode == 'union':
        merged = (m1 | m2 | m3).astype(np.uint8)
    elif fusion_mode == 'intersection':
        merged = (m1 & m2 & m3).astype(np.uint8)
    elif fusion_mode == 'voting':
        votes = m1.astype(np.uint8) + m2.astype(np.uint8) + m3.astype(np.uint8)
        merged = (votes >= 2).astype(np.uint8) # majority voting

    tiff.imwrite("merged_segmentation.tif", merged)
    return

def main() -> int:
    # --------------------
    # 1 TRANSFORM INPUT
    # --------------------
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        type=str, 
        required=True, 
        help='Path to the input 3D image (TIFF format)'
    )

    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help='Path to the output segmentation mask (TIFF format)'
    )

    parser.add_argument(
        '--fusion_mode', 
        type=str, 
        choices=['union', 'intersection', 'voting'], 
        default='union', 
        help='Method to fuse the three segmentation masks (default: union)'
    )

    parser.add_argument(
        '--padding', 
        type=int, 
        default=0, 
        help='Amount of padding to add to the images (default: 0, no padding)'
    )

    main()