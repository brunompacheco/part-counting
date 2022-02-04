from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d

from src.data.rgbd import load_rgbd


def load_pcd(rgbd_image: Union[Path, o3d.geometry.RGBDImage]
            ) -> o3d.geometry.PointCloud:
    """Load a PointCloud from an RGBD image.

    Args:
        rgbd_image: path to RGBD image or RGBD image already loaded (see
        `load_rgbd` function).

    Returns:
        pcd: pointcloud with one point per pixel in the input image.
    """
    f = 711.1111  # focal distance, at the center of the image
    c = 256  # image center for both axis (expecting a 512x512 image)
    scale = (0.5, 1.5)

    if isinstance(rgbd_image, Path):
        rgbd = load_rgbd(rgbd_image, (c,c), f, scale)
    else:
        rgbd = rgbd_image

    camera_params = o3d.camera.PinholeCameraIntrinsic(512, 512, f, f, c, c)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_params)

    # adjust Z orientation (it would be upside-down, otherwise)
    points = np.array(pcd.points)
    points = np.array([[p[0],p[1],1.5-p[2]] for p in points])

    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd
