# Copyright (c) 2021, ETHZ V4RL. All rights reserved.
# Licensed under the BSD 3-Clause License.

import numpy as np
import open3d as o3d


class DataLoader(object):

    def __init__(self,
                 dataset,
                 img_step=1,
                 pcloud_voxel_size=-1,
                 pcloud_normals_radius=-1,
                 pcloud_normals_max_nn=0):
        self.dataset = dataset
        self.img_step = img_step
        self.pcloud_voxel_size = pcloud_voxel_size
        self.pcloud_normals_radius = pcloud_normals_radius
        self.pcloud_normals_max_nn = pcloud_normals_max_nn

    def frames(self):
        for idx in range(0, len(self.dataset.img_labels_info), self.img_step):
            yield self.dataset.load_frame(idx), idx

    def point_cloud(self):
        pcloud_o3d = self.dataset.load_point_cloud()
        assert (pcloud_o3d.has_points())

        if self.pcloud_voxel_size > 0:
            pcloud_o3d = pcloud_o3d.voxel_down_sample(
                voxel_size=self.pcloud_voxel_size)

        if self.pcloud_normals_radius > 0 and self.pcloud_normals_max_nn > 0:
            pcloud_o3d.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.pcloud_normals_radius,
                    max_nn=self.pcloud_normals_max_nn))

        assert (pcloud_o3d.has_normals())

        return np.asarray(pcloud_o3d.points), np.asarray(pcloud_o3d.normals)
