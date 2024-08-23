# Copyright (c) 2021, ETHZ V4RL. All rights reserved.
# Licensed under the BSD 3-Clause License.

import numpy as np
import open3d as o3d


def get_label_colors(labels, label_taxonomy):
    color_palette = np.asarray(label_taxonomy.PALETTE) / 255
    color_palette = np.concatenate((color_palette, np.array([[0, 0, 0]])),
                                   axis=0)
    return color_palette[labels]


def visualize_3d_segmentation(points,
                              labels,
                              label_taxonomy,
                              show=True,
                              save_as=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(
        get_label_colors(labels, label_taxonomy))
    if show:
        o3d.visualization.draw_geometries([pcd])
    if save_as is not None:
        o3d.io.write_point_cloud(save_as, pcd)
        print("Saved colored point cloud as '%s'" %(save_as))


def save_data_array(array, filename):
    if filename.endswith('.npy'):
        np.save(filename, array)
        print("Saved output as '%s'" %(filename))
    elif filename.endswith('.txt'):
        np.savetxt(filename, array)
        print("Saved output as '%s'" %(filename))
