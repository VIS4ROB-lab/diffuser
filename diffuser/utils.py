# Copyright (c) 2021, ETHZ V4RL. All rights reserved.
# Licensed under the BSD 3-Clause License.

import os

import numpy as np
from sklearn.neighbors import KDTree


def compute_perspective_projection(points, K, T_C_W):
    P = K.dot(T_C_W[:3, :])
    points_projected = points.dot(P.T)
    depth = points_projected[:, 2]
    px_coords = np.column_stack([
        points_projected[:, 0] / depth + 0.5,
        points_projected[:, 1] / depth + 0.5
    ]).astype(int)
    return px_coords, depth


def is_within_image_bounds(px_coords, img_size):
    visible = px_coords[:, 0] >= 0
    visible = np.logical_and(visible, px_coords[:, 1] >= 0)
    visible = np.logical_and(visible, px_coords[:, 0] < img_size[1])
    visible = np.logical_and(visible, px_coords[:, 1] < img_size[0])
    return visible


def check_consistency(values, px_coords, image, threshold=0.05):
    return np.abs(values - image[px_coords[:, 1], px_coords[:, 0]]) < threshold


def find_nearest_neighbors(points, num_neighbors):
    kdtree = KDTree(points)
    distances, neighbors = kdtree.query(points, k=num_neighbors + 1)
    return distances[:, 1:], neighbors[:, 1:]


def get_sorted_file_list(dir, file_ext=None):
    if file_ext is None:
        files = [file.path for file in os.scandir(dir) if os.path.isfile(file)]
    else:
        files = [
            file.path for file in os.scandir(dir)
            if file.name.endswith(file_ext)
        ]
    files.sort()
    return files