import numpy as np
import open3d as o3d

from pathlib import Path


def load_rgbd(image_fpath: Path):
    image = o3d.io.read_image(str(image_fpath))
    image_data = np.asarray(image)

    color_image = o3d.geometry.Image(image_data[:,:,2].astype('uint8'))
    depth_image = o3d.geometry.Image(255-image_data[:,:,1].astype('uint8'))

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_scale=1
    )
