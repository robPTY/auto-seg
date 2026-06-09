#
# Script to merge the segmentation masks of the three views (XY, XZ, YZ) into a single 3D segmentation mask.
#
import numpy as np
import tifffile as tiff

PAD = 50

def main() -> int:
    zyx = f"{PAD}pad_img_xy_seg.tif"
    yzx = f"{PAD}pad_img_zx_seg.tif"
    xzy = f"{PAD}pad_img_zy_seg.tif"

    # Load images and undo transpose
    vol_zyx = tiff.imread(zyx)  # (z,y,x)
    vol_yzx = tiff.imread(yzx).transpose(1, 0, 2)  # (y,z,x) -> (z,y,x)
    vol_xzy = tiff.imread(xzy).transpose(1, 2, 0)  # (x,z,y) -> (z,y,x)

    print("zyx:", vol_zyx.shape)
    print("yzx:", vol_yzx.shape)
    print("xzy:", vol_xzy.shape)


    # Crop to original size (remove padding)
    vol_zyx = vol_zyx[PAD:-PAD, :, :]
    vol_yzx = vol_yzx[:, PAD:-PAD, :]
    vol_xzy = vol_xzy[:, :, PAD:-PAD]

    print("After cropping:")
    print("zyx:", vol_zyx.shape)
    print("yzx:", vol_yzx.shape)
    print("xzy:", vol_xzy.shape)

    # Merge
    m1 = vol_zyx > 0
    m2 = vol_yzx > 0
    m3 = vol_xzy > 0

    votes = (
        m1.astype(np.uint8) +
        m2.astype(np.uint8) +
        m3.astype(np.uint8)
    )

    fused = votes >= 2

    tiff.imwrite(f"{PAD}pad_fused.tif", fused.astype(np.uint8))

    fused_union = (
        (m1 > 0) |
        (m2 > 0) |
        (m3 > 0)
    )

    tiff.imwrite(f"{PAD}pad_fused_union.tif", fused_union.astype(np.uint8))

    return 0

if __name__ == "__main__":
    main()
