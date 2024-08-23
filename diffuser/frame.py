# Copyright (c) 2021, ETHZ V4RL. All rights reserved.
# Licensed under the BSD 3-Clause License.

import numpy as np

from .utils import (check_consistency, compute_perspective_projection,
                    is_within_image_bounds)


class Frame(object):

    def __init__(self, labels, depth, intrinsics_rgb, intrinsics_depth,
                 extrinsics):
        self.labels = labels
        self.depth = depth
        self.T_C_W = extrinsics
        self.K_rgb = intrinsics_rgb
        self.K_depth = intrinsics_depth
        self.rgb_height, self.rgb_width = labels.shape
        self.depth_height, self.depth_width = depth.shape

    def project(self, points):
        num_points = points.shape[0]
        # Append 1 for homogeneous coordinates
        points = np.concatenate([points, np.ones((num_points, 1))], axis=1)

        # Project points
        points_projected, points_depth = compute_perspective_projection(
            points, self.K_rgb, self.T_C_W)

        # Compute visibility mask
        visible = points_depth > 0
        visible = np.logical_and(
            visible, is_within_image_bounds(points_projected,
                                            self.labels.shape))

        # Depth consistency check
        if np.array_equal(self.K_rgb, self.K_depth):
            points_projected_d = points_projected
        else:
            points_projected_d, _ = compute_perspective_projection(
                points, self.K_depth, self.T_C_W)
            visible = np.logical_and(
                visible,
                is_within_image_bounds(points_projected_d, self.depth.shape))

        depth_consistent = check_consistency(points_depth[visible],
                                             points_projected_d[visible, :],
                                             self.depth, 0.05)

        visible[visible] = depth_consistent
        points_projected[~visible, :] = -1000

        return points_projected, visible
