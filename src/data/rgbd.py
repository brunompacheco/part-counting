import cv2
import numpy as np
import open3d as o3d

from pathlib import Path


def load_rgbd(image_fpath: Path, c=(256, 256), f=711.1111, scale=(0.5, 1.5)
             ) -> o3d.geometry.RGBDImage:
    """Load .png image and build Open3D's RGBDImage.

    Args:
        image_fpath: path to .exr with grayscale in the first channel and depth
        in the second channel.
        c: focal point of the image (in pixels).
        f: focal distance (in pixels).
        scale: offset and max of the depth, in meters.

    Returns:
        rgbd: Open3D RGBD image.
    """
    image_fpath = Path(image_fpath)

    image_data = cv2.imread(
        str(image_fpath),
        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    )
    color = image_data[:,:,0]
    distance = image_data[:,:,1]  # distance to the camera

    # fix scale
    # near = 0.928849
    # far = 1.371349
    near = scale[0]
    far = scale[1]
    distance = near + distance * (far - near)

    rows, cols = distance.shape
    x_px, y_px = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    x_px = x_px - c[0]
    y_px = y_px - c[1]

    # pixel distance to image center
    d_px = np.sqrt(np.square(x_px) + np.square(y_px))

    # pixel distance to camera
    z_px = np.sqrt(np.square(d_px) + np.square(f))

    # convert distance to camera to depth from camera level
    depth = distance * f / z_px

    color_image = o3d.geometry.Image(np.sqrt(color.astype('float32')))
    depth_image = o3d.geometry.Image(depth.astype('float32'))

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_scale=1, depth_trunc=far
    )
