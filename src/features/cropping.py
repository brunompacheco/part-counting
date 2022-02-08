import numpy as np
import open3d as o3d

from typing import Tuple

from skimage.feature import canny
from skimage.measure import label
from skimage.morphology import binary_dilation
from skimage.transform import hough_line, hough_line_peaks
from sklearn.cluster import DBSCAN

from src.data.pcd import load_pcd


def get_lines(image: np.ndarray, canny_sigma=1, angle_range=np.pi/8
    ) -> Tuple[Tuple[np.array,np.array],Tuple[np.array,np.array]]:
    """Find best horizontal and vertical lines that match the box contour.

    Args:
        image: numpy array that represents the grayscale image in 0-1 range.
        canny_sigma: sigma value passed to canny.
        angle_range: (half) of the range to search the lines, aroung 0
        (vertical lines) and pi/2 (horizontal lines).

    Returns:
        v_angles: angles of the vertical lines found (hough transform).
        v_dists: distances of the vertical lines found (hough transform).
        h_angles: angles of the horizontal lines found (hough transform).
        h_dists: distances of the horizontal lines found (hough transform).
    """
    # standardize image
    image_ = np.array(image*255, dtype='uint8')  # skimage-friendly

    # compute edges
    edges = canny(image_, sigma=canny_sigma)

    v_angles = np.linspace(-angle_range, angle_range, 90, endpoint=True)
    h_angles = np.linspace(np.pi/2-angle_range, np.pi/2+angle_range, 180, endpoint=True)

    h, theta, d = hough_line(edges, theta=v_angles)
    _, v_angles, v_dists = hough_line_peaks(h, theta, d)

    h, theta, d = hough_line(edges, theta=h_angles)
    _, h_angles, h_dists = hough_line_peaks(h, theta, d)

    return (v_angles, v_dists), (h_angles, h_dists)

def pick_similar_lines(angles: np.array, dists: np.array, n=4, eps=0.02
    ) -> Tuple[np.array, np.array]:
    """Pick `n` lines closest to being parallel to each other.

    Uses DBSCAN to cluster lines by angle (min_samples=n). Picks the cluster
    with the smallest internal angle variance. From this cluster, picks the `n`
    lines that are closest to the median.

    Args:
        angles: lines' angles (hough transform representation).
        dists: lines' distances (hough transform representation).
        n: number of lines to be selected.
        eps: see sklearn's DBSCAN.

    Returns:
        best_angles: angles of the `n` lines selected.
        best_dists: distances of the `n` lines selected.
    """
    labels = DBSCAN(min_samples=n, eps=eps).fit_predict(angles.reshape(-1,1))

    clusters = {l: angles[labels == l] for l in range(max(labels)+1)}

    # compute within-cluster variance
    clusters_variance = dict()
    for l, cluster in clusters.items():
        clusters_variance[l] = np.sum(np.abs(cluster - np.mean(cluster)))

    # pick cluster with smalles internal variance
    a = np.array(list(zip(clusters_variance.keys(),clusters_variance.values())))
    best_l, _ = np.sort(a, axis=0)[0]
    best_l = int(best_l)
    best_angles = angles[labels == best_l]

    # sort angles by distance to median (within-cluster similarity)
    median_angle = np.median(best_angles)
    angles_dists = np.abs(best_angles - np.median(best_angles))
    best_angles = np.sort(np.stack([angles_dists,best_angles], axis=1), axis=0)[:,1]

    # pick top 4
    best_i = np.isin(angles, best_angles[:4])
    best_angles = angles[best_i]
    best_dists = dists[best_i]

    return best_angles, best_dists

def get_box_mask(shape: tuple,
                 h_lines: Tuple[np.array, np.array],
                 v_lines: Tuple[np.array, np.array]) -> np.ndarray:
    """Generate mask of area delimited by given lines.

    One mask is generated for each line, partitioning the area in two. Each
    mask is such that the center point of the image is always in the positive
    area. The resulting mask is the combinantion (AND) of all masks.

    Args:
        shape: shape of the image for the mask.
        h_lines: horizontal lines that delimit the area of interest.
        v_lines: vertical lines that delimit the area of interest.

    Return:
        mask: binary mask of the region of interest.
    """
    idx = np.indices(shape)
    mask = np.ones(shape).astype(bool)

    for angle, dist in zip(*v_lines):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        slope = np.tan(angle + np.pi/2)

        # get line function
        f = lambda y: x0 + (y - y0) * (1/slope)

        # generate line-related mask
        if f(shape[1] / 2) < shape[0] / 2:
            mask = mask & (idx[1] > f(idx[0]))
        else:
            mask = mask & (idx[1] < f(idx[0]))

    for angle, dist in zip(*h_lines):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        slope = np.tan(angle + np.pi/2)

        # get line function
        f = lambda x: y0 + (x - x0)*slope

        # generate line-related mask
        if f(shape[0] / 2) < shape[1] / 2:
            mask = mask & (idx[0] > f(idx[1]))
        else:
            mask = mask & (idx[0] < f(idx[1]))

    return mask

def fill_binary(mask):
    """Works best if `mask` is convex.
    """
    x, y = np.indices(mask.shape)

    x_max = np.max(x, initial=0, where=mask, axis=0)
    x_min = np.min(x, initial=mask.shape[0], where=mask, axis=0)

    y_max = np.max(y, initial=0, where=mask, axis=1)
    y_min = np.min(y, initial=mask.shape[1], where=mask, axis=1)

    return   (x >= x_min) \
           & (x <= x_max) \
           & (y >= y_min.reshape((-1,1))) \
           & (y <= y_max.reshape((-1,1)))

def largest_connected_component(mask):
    """Return connected component with larges (filled) area.

    Fills connected components using `fill_binary` above before calculating
    size.
    """
    labeled, num = label(mask, background=0, return_num=True)

    labels_sizes = [np.sum(fill_binary(labeled == i)) for i in range(1,num+1)]

    biggest_label = np.argmax(labels_sizes) + 1

    return labeled == biggest_label

def box_mask_from_rgbd(rgbd: o3d.geometry.RGBDImage) -> np.ndarray:
    """Get interior box mask for RGBD image.

    It is expected that the depth channel of the RGBD image contains depths in
    the range of 0 to 1.5 meters.

    Returns:
        mask: binary mask of the interior of the box in the shape of the input
        image.
    """
    depth = np.asarray(rgbd.depth)

    top = (1.5 - depth) >= 0.5  # top of the box

    # remove everything (like parts) but the box border (and parts touching it)
    top = largest_connected_component(top)

    v_lines, h_lines = get_lines(top)
    
    v_lines = pick_similar_lines(v_lines[0], v_lines[1])
    h_lines = pick_similar_lines(h_lines[0], h_lines[1])
    
    return get_box_mask(top.shape, h_lines, v_lines)

def mask_selection_volume(rgbd: o3d.geometry.RGBDImage, mask: np.array,
                          border_size=20) -> o3d.visualization.SelectionPolygonVolume:
    """Get selection volume (3D) of the box mask.

    Args:
        rgbd: RGBD image to generate the point cloud from.
        mask: mask of the interior of the box. see `box_mask_from_rgbd`.
        border_size: number of pixels to dillate the mask as to capture enough
        of the border of the box.
    """
    depth = np.array(rgbd.depth)

    # get box border
    border = binary_dilation(mask, np.ones((border_size,border_size))) ^ mask

    border_depth = np.median(depth, where=border)

    # crate RGBD image of the mask
    box_pcd_depth = 2 * np.ones(mask.shape)
    box_pcd_depth[mask] = border_depth

    mask_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ones(box_pcd_depth.shape).astype('uint8') * 25),
        o3d.geometry.Image((255 * box_pcd_depth / 2).astype('uint8')),
        depth_scale=0.5,
        depth_trunc=2.0
    )

    # get selection volume vertices from the PCD generated from the mask RGBD
    mask_points = load_pcd(mask_rgbd).points

    main_diagonal = [p[0] + p[1] for p in mask_points]
    secondary_diagonal = [-p[0] + p[1] for p in mask_points]

    ne = mask_points[np.argmax(main_diagonal)]
    sw = mask_points[np.argmin(main_diagonal)]

    nw = mask_points[np.argmax(secondary_diagonal)]
    se = mask_points[np.argmin(secondary_diagonal)]

    polygon = np.array([ne, se, sw, nw])

    # create selection polygon
    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Z"
    vol.axis_min = 0
    vol.axis_max = 1.5
    vol.bounding_polygon = o3d.utility.Vector3dVector(polygon)

    return vol
