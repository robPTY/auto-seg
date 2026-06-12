#
# Script to merge segmentation masks from three views (XY, XZ, YZ) into one 3D mask.
#
import argparse

import numpy as np
import tifffile as tiff


def read_view(path: str, transpose_axes: tuple[int, int, int], crop_axis: int, padding: int) -> np.ndarray:
    """Load one view, orient it back to ZYX, and remove view-axis padding."""
    volume = tiff.imread(path).transpose(transpose_axes)

    if padding <= 0:
        return volume

    slices = [slice(None)] * volume.ndim
    slices[crop_axis] = slice(padding, -padding)
    return volume[tuple(slices)]


def fuse_masks(volumes: list[np.ndarray], fusion_mode: str) -> np.ndarray:
    masks = [(volume > 0) for volume in volumes]

    if fusion_mode == "union":
        fused = np.logical_or.reduce(masks)
    elif fusion_mode == "intersection":
        fused = np.logical_and.reduce(masks)
    elif fusion_mode == "voting":
        votes = sum(mask.astype(np.uint8) for mask in masks)
        fused = votes >= 2
    else:
        raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

    return fused.astype(np.uint8)


def validate_shapes(volumes: dict[str, np.ndarray]) -> None:
    shapes = {name: volume.shape for name, volume in volumes.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Segmentation shapes do not match after alignment: {shapes}")


def merge_segmentations(
    xy_path: str,
    xz_path: str,
    yz_path: str,
    output_path: str,
    padding: int,
    fusion_mode: str,
) -> None:
    # Each tuple is: name, path, transpose back to ZYX, padded axis after transpose.
    views = [
        ("xy", xy_path, (0, 1, 2), 0),
        ("xz", xz_path, (1, 0, 2), 1),
        ("yz", yz_path, (1, 2, 0), 2),
    ]

    volumes = {
        name: read_view(path, transpose_axes, crop_axis, padding)
        for name, path, transpose_axes, crop_axis in views
    }

    for view_name, volume in volumes.items():
        print(f"{view_name.upper()} aligned shape:", volume.shape)

    validate_shapes(volumes)

    fused = fuse_masks(list(volumes.values()), fusion_mode)
    tiff.imwrite(output_path, fused)
    print(f"Saved {fusion_mode} fusion to {output_path}")


def main(args: argparse.Namespace) -> int:
    merge_segmentations(
        xy_path=args.xy,
        xz_path=args.xz,
        yz_path=args.yz,
        output_path=args.output,
        padding=args.padding,
        fusion_mode=args.fusion_mode,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge XY/XZ/YZ segmentation masks into one 3D segmentation."
    )

    parser.add_argument(
        "--xy",
        required=True,
        help="Path to the XY segmentation TIFF, shaped as Z/Y/X.",
    )

    parser.add_argument(
        "--xz",
        required=True,
        help="Path to the XZ segmentation TIFF, shaped as Y/Z/X.",
    )

    parser.add_argument(
        "--yz",
        required=True,
        help="Path to the YZ segmentation TIFF, shaped as X/Z/Y.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for the merged output TIFF.",
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Padding to remove from each view axis before merging.",
    )
    
    parser.add_argument(
        "--fusion-mode",
        "--fusion_mode",
        dest="fusion_mode",
        choices=["union", "intersection", "voting"],
        default="voting",
        help="Method to combine the three masks.",
    )

    main(parser.parse_args())
