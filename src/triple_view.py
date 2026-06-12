#
# Script to create the three views (XY, XZ, YZ) of the original 3D image. The XY view is the original image.
# Additionally, add padding to the images (work in progress)
#
import argparse
import numpy as np
import tifffile as tiff

PADDING = True
PADDING_SIZE = 10


VIEW_AXES = {
    "xy": (0, 1, 2),
    "xz": (1, 0, 2),
    "yz": (2, 0, 1),
}


def pad_first_axis(volume: np.ndarray) -> np.ndarray:
    if not PADDING:
        return volume
    return np.pad(
        volume,
        ((PADDING_SIZE, PADDING_SIZE), (0, 0), (0, 0)),
        mode="reflect",
    )


def make_view(volume: np.ndarray, axes: tuple[int, int, int]) -> np.ndarray:
    return pad_first_axis(volume.transpose(axes))


def output_path(view_name: str) -> str:
    if PADDING:
        return f"{PADDING_SIZE}pad_img_{view_name}.tif"
    return f"img_{view_name}.tif"


def main(args) -> int:
    # Load original volume
    volume = tiff.imread(args.input)

    print("Typing of input image:", volume.dtype)
    print("Original shape (z,y,x):", volume.shape)

    views = {
        view_name: make_view(volume, axes)
        for view_name, axes in VIEW_AXES.items()
    }

    for view_name, view_volume in views.items():
        print(f"{view_name.upper()} shape:", view_volume.shape)
        tiff.imwrite(output_path(view_name), view_volume)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        '--input',
        type=str, 
        required=True, 
        help='Path to the input 3D image (TIFF format)'
    )

    args = parser.parse_args()
    main(args)
