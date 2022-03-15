import open3d as o3d
from src.features.base import preprocess_box_for_cv

from src.features.voxels import voxel2np


def estimate_volume(box: o3d.geometry.PointCloud, voxel_size: float) -> float:
    """Estimate volume occupied inside box through voxelization.
    
    Args:
        box: Point cloud generated from RGBD image.
        voxel_size: Resolution of the voxel grid generated. Be careful so it is
        not higher than the point cloud resolution.
    
    Returns:
        vol: Estimated volume (in the same unit as the voxel size).
    """
    box_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(box, voxel_size)

    # get grid shape and position
    grid_pos = box_voxel.get_min_bound()
    grid_pos[-1] = 0.1088  # floor

    grid_shape = (box_voxel.get_max_bound() - grid_pos) / box_voxel.voxel_size
    grid_shape = (grid_shape + 0.5).astype(int)

    box_grid = voxel2np(box_voxel, grid_pos, grid_shape)

    return box_grid.sum() * (voxel_size ** 3)


if __name__ == '__main__':
    """Fit linear regression and polynomial models to the training data.
    """
    from joblib import dump
    from pathlib import Path

    import numpy as np

    from dotenv import find_dotenv
    from tqdm import tqdm

    from sklearn.linear_model import LinearRegression


    # find .env automagically by walking up directories until it's found
    dotenv_path = find_dotenv()
    project_dir = Path(dotenv_path).parent

    raw_data_dir = project_dir/'data/raw/render_results_imov_cam_mist_simple'

    voxel_size = 0.005

    vols = dict()
    for img_fpath in tqdm(list(raw_data_dir.glob('*/*.exr'))):
        if img_fpath == Path('/home/ctc_das/Desktop/part_counting/data/raw/render_results_imov_cam_mist_simple/simulacao120/simulacao120_0098.exr'):
            # there was a problem with this image
            continue

        box = preprocess_box_for_cv(img_fpath)
        
        vol = estimate_volume(box, voxel_size=voxel_size)
        
        vols[img_fpath.name] = vol

    y = np.array([int(y.split('_')[-1].split('.')[0]) for y in vols.keys()])
    X = np.array(list(vols.values()))

    # fit sklearn's linear regression
    lr = LinearRegression()
    lr = lr.fit(X.reshape(-1,1), y)

    with open(project_dir/'models/linear_regression_.pkl', 'wb') as f:
        dump(lr, f)

    # fit a polynomial of degree 2
    p = np.polyfit(X, y, 2)

    with open(project_dir/'models/polynomial_fit_.pkl', 'wb') as f:
        dump(p, f)
