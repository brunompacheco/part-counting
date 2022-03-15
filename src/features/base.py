from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from src.data.rgbd import load_rgbd
from src.data.pcd import load_pcd
from .cropping import mask_selection_volume, box_mask_from_rgbd


def preprocess_box_for_cv(img_fpath: Path) -> o3d.geometry.PointCloud:
    """Load and strip walls of box, keeping the interior. For CV-based models.

    Args:
        img_fpath: Filepath of the .exr image file. Must contain grayscale as
        the first channel and depth as second channel.
    
    Returns:
        box: Point cloud image of the interior of the box.
    """
    rgbd = load_rgbd(img_fpath)

    box_mask = box_mask_from_rgbd(rgbd)

    vol = mask_selection_volume(rgbd, box_mask)

    pcd = load_pcd(rgbd)

    box = vol.crop_point_cloud(pcd)

    return box

def load_part_model(part_fpath: Path, number_of_points=10000) -> o3d.geometry.PointCloud:
    """Load part model as a point cloud image in meters.

    Args:
        part_fpath: Filepath of the .stl model file.
        number_of_points: For the resulting point cloud, which is sampled
        uniformly.
    
    Returns:
        part: Point cloud of the part, sampled uniformly.
    """
    part_mesh = o3d.io.read_triangle_mesh(str(part_fpath), enable_post_processing=True)

    part_mesh.paint_uniform_color([1., 0., 0.,])

    part = part_mesh.sample_points_uniformly(number_of_points=number_of_points)

    part_points = np.array(part.points) / 1000  # mm to meter conversion
    part_points = part_points + np.array([0,0,0.3])
    part_points = o3d.utility.Vector3dVector(part_points)
    part.points = part_points

    return part

def preprocess_box_for_dl(img_fpath: Path, device: torch.device = None) -> torch.Tensor:
    """Load box picture and reshape it. For DL-based models.

    Args:
        img_fpath: Filepath of the .png image file.
        device: Torch device where to load the image.

    Returns:
        X: Image loaded in a batch-like format (batch with a single sample),
        proper for feeding to a model.
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data = np.array(o3d.io.read_image(str(img_fpath)))
    # swap axis so that channels is the first axis
    data = np.moveaxis(data[:,:,1:3], -1, 0)

    X = torch.from_numpy(data).unsqueeze(0)
    X = X.type(torch.FloatTensor)
    X = X.to(device)

    return X
