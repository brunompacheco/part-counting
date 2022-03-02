import numpy as np
import open3d as o3d

from src.features.voxels import voxel2np


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
