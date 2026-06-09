#
# Script to create the three views (XY, XZ, YZ) of the original 3D image. The XY view is the original image.
# Additionally, add padding to the images (work in progress)
#
import numpy as np
import tifffile as tiff

padding = True
PADDING_SIZE = 50

# Load original volume
volume = tiff.imread("img.tif")

print("Typing of input image:", volume.dtype)
print("Original shape (z,y,x):", volume.shape)

if padding:
    volume_pad = np.pad(
        volume,
        ((PADDING_SIZE, PADDING_SIZE), (0, 0), (0, 0)),
        mode="reflect"
    )

# XZ view
vol_xz = volume.transpose(1, 0, 2)  # (y,z,x)
if padding:
    vol_xz = np.pad(
        vol_xz,
        ((PADDING_SIZE, PADDING_SIZE), (0, 0), (0, 0)),
        mode="reflect"
    )

# YZ view
vol_yz = volume.transpose(2, 0, 1)  # (x,z,y)
if padding:
    vol_yz = np.pad(
        vol_yz,
        ((PADDING_SIZE, PADDING_SIZE), (0, 0), (0, 0)),
        mode="reflect"
    )

print("XY shape:", volume_pad.shape)
print("XZ shape:", vol_xz.shape)
print("YZ shape:", vol_yz.shape)

# Save
tiff.imwrite(f"{PADDING_SIZE}pad_img_xy.tif" if padding else "img_xy.tif", volume_pad)
tiff.imwrite(f"{PADDING_SIZE}pad_img_xz.tif" if padding else "img_xz.tif", vol_xz)
tiff.imwrite(f"{PADDING_SIZE}pad_img_yz.tif" if padding else "img_yz.tif", vol_yz)
