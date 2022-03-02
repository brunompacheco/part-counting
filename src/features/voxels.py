import numpy as np
import open3d as o3d


def get_shape_pos(v_grid: o3d.geometry.VoxelGrid):
    # get grid shape and position
    grid_pos = v_grid.get_min_bound()
    # grid_pos[-1] = 0.0  # floor
    grid_pos[-1] = 0.1288  # floor

    grid_shape = (v_grid.get_max_bound() - grid_pos) / v_grid.voxel_size
    grid_shape = (grid_shape + 0.5).astype(int)

    return grid_shape, grid_pos

def voxel2np(v_grid: o3d.geometry.VoxelGrid, pos: np.array, shape: np.array,
             mode='shadow'):
    """Convert VoxelGrid to numpy array.

    Offsets the voxels in `v_grid` to match a grid at `pos` with `shape`.

    Args:
        v_grid: Open3D VoxelGrid to be converted.
        pos: Spatial positioning of the corner of the final grid, its minimum
        bound.
        shape: Shape of the target grid.
        mode: how to fill the voxel grid based on `v_grid` voxels. If `shadow`
        (default), fills everything below. If `reverse-shadow`, fills
        everything above. If `floating`, fills only the intersection between
        `shadow` and `reverse-shadow` grids, that is, the middle in the z axis.

    Returns:
        grid: Numpy voxel grid.
    """
    offset = (pos - v_grid.get_min_bound()) / v_grid.voxel_size
    offset = (offset + 0.5).astype(int)

    indices = [v.grid_index - offset for v in v_grid.get_voxels()]

    # image filled to the bottom
    grid_bottom = np.zeros(shape, dtype=bool)
    for i in indices:
        if (i > 0).all() and (i < shape).all():
            grid_bottom[i[0],i[1],0:i[2]] = True

    # image filled to the top
    grid_top = np.zeros(shape, dtype=bool)
    for i in indices:
        if (i > 0).all() and (i < shape).all():
            grid_top[i[0],i[1],0:i[2]] = True

    if mode == 'shadow':
        return grid_bottom
    elif mode == 'floating':
        return grid_bottom & grid_top
    elif mode == 'reverse-shadow':
        return grid_top
