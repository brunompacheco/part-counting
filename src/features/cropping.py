import numpy as np

from typing import Tuple

from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from sklearn.cluster import DBSCAN


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

    v_angles = np.linspace(-angle_range, angle_range, 180, endpoint=True)
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
