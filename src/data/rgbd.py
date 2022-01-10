import numpy as np
import open3d as o3d

from pathlib import Path


def load_rgbd(image_fpath: Path, depth_scale=711.1111111 / 0.05
    ) -> o3d.geometry.RGBDImage:
    """Load .png image and build Open3D's RGBDImage.

    Args:
        image_fpath: path to .png with grayscale in the blue channel and depth
        in the green channel.
        depth_scale: see `o3d.geometry.RGBDImage.create_from_color_and_depth`.
    
    Returns:
        rgbd: Open3D RGBD image.
    """
    image = o3d.io.read_image(str(image_fpath))
    image_data = np.asarray(image)

    color_image = o3d.geometry.Image((image_data[:,:,2] / 255).astype('float32'))
    # depth_image = o3d.geometry.Image((1.371349 - 0.928849 - (1.371349 - 0.928849) * np.power(image_data[:,:,1].astype(float), 2) / np.square(255)).astype('float32'))
    depth_image = o3d.geometry.Image((1.371349 - 0.928849 - (1.371349 - 0.928849) * image_data[:,:,1].astype(float) / 255).astype('float32'))

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_scale=depth_scale, depth_trunc=711.1111111 / 0.0005
    )
