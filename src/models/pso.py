import numpy as np
import open3d as o3d
import pyswarms as ps

from src.features.voxels import voxel2np, dig_box


def intersection_value(box_grid: np.ndarray, part_grid: np.ndarray):
    return (box_grid & part_grid).sum()

def surface_distance(box_grid: np.ndarray, part_grid: np.ndarray):
    part_surface = (np.indices(part_grid.shape)[-1] * part_grid).max(axis=-1)
    part_mask = part_grid.sum(axis=-1) > 0

    box_surface = (np.indices(box_grid.shape)[-1] * box_grid).max(axis=-1)

    distances = np.abs((box_surface * part_mask) - part_surface)

    return distances.sum()

def reward(box_grid, part_grid):
    return intersection_value(box_grid, part_grid) \
           - surface_distance(box_grid, part_grid)

class _PointCloudTransmissionFormat:  # see https://github.com/isl-org/Open3D/issues/218
    def __init__(self, pointcloud: o3d.geometry.PointCloud):
        self.points = np.array(pointcloud.points)
        self.colors = np.array(pointcloud.colors)
        self.normals = np.array(pointcloud.normals)

    def create_pointcloud(self) -> o3d.geometry.PointCloud:
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(self.points)
        pointcloud.colors = o3d.utility.Vector3dVector(self.colors)
        pointcloud.normals = o3d.utility.Vector3dVector(self.normals)
        return pointcloud

def eval_position(x: np.array, box_grid: np.ndarray,
                  part: _PointCloudTransmissionFormat, voxel_size: float,
                  grid_pos: np.array, grid_shape: np.array):
    tx, ty, tz, rx, ry, rz = x

    part_ = part.create_pointcloud()

    part_.translate(np.array([tx, ty, tz]))

    part_.rotate(
        o3d.geometry.get_rotation_matrix_from_xyz(np.array([rx, ry, rz]))
    )

    part_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(part_, voxel_size)
    part_grid = voxel2np(part_voxel, grid_pos, grid_shape, mode='floating')

    return - reward(box_grid, part_grid)

def objective_f(xs, box_grid, part, voxel_size, grid_pos, grid_shape):
    if xs.size > 0:
        return np.apply_along_axis(
            eval_position, -1, xs,
            box_grid=box_grid,
            part=part,
            voxel_size=voxel_size,
            grid_pos=grid_pos,
            grid_shape=grid_shape
        )
    else:
        return np.apply_along_axis(
            eval_position, -1, np.array([0.,0.,0.,0.,0.,0.]),
            box_grid=box_grid,
            part=part,
            voxel_size=voxel_size,
            grid_pos=grid_pos,
            grid_shape=grid_shape
        )

def dig_and_predict(box: o3d.geometry.PointCloud, part: o3d.geometry.PointCloud,
                    voxel_size: float) -> int:
    box_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(box, voxel_size)

    # get grid shape and position
    grid_pos = box_voxel.get_min_bound()
    grid_pos[-1] = 0.1288  # floor

    grid_shape = (box_voxel.get_max_bound() - grid_pos) / box_voxel.voxel_size
    grid_shape = (grid_shape + 0.5).astype(int)

    # create box grid
    box_grid = voxel2np(box_voxel, grid_pos, grid_shape)
    
    # optimization bounds
    min_bound = np.array([-0.5, -0.5, -0.5, -np.pi, -np.pi, -np.pi])
    bounds = (min_bound, -min_bound)

    ## PSO LOOP
    n = 0  # number of parts
    while True:
        optimizer = ps.single.LocalBestPSO(
            n_particles=18,
            dimensions=6,
            bounds=bounds,
            options={'c1': 0.5, 'c2': 0.05, 'w':0.9, 'k': 2, 'p': 2},
        )

        _, pos = optimizer.optimize(
            objective_f,
            iters=500,
            n_processes=6,
            box_grid=box_grid,
            part=_PointCloudTransmissionFormat(part),
            voxel_size=voxel_size,
            grid_pos=grid_pos,
            grid_shape=grid_shape,
            verbose=False,
        )

        # get part at optimal position
        tx, ty, tz, rx, ry, rz = pos

        part_ = o3d.geometry.PointCloud(part)

        part_.translate(np.array([tx, ty, tz]))

        part_.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz(np.array([rx, ry, rz]))
        )

        # conert part to voxel grid
        part_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(part_, voxel_size)
        part_grid = voxel2np(part_voxel, grid_pos, grid_shape, mode='floating')

        # find out how much of the part is immersed at the box content
        immersion_ratio = (part_grid & box_grid).sum() / part_grid.sum()

        if immersion_ratio > 0.60:
            box_grid = dig_box(box_grid, part, grid_pos, grid_shape, pos,
                               voxel_size)
            n += 1
        else:
            break

    return n
