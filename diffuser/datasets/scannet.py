# Copyright (c) 2021, ETHZ V4RL. All rights reserved.
# Licensed under the BSD 3-Clause License.

import os

import numpy as np
from PIL import Image

from ..utils import get_sorted_file_list
from .base_dataset import BaseLabelTaxonomy, BaseScene
from .builder import (register_dataset, register_label_mapping,
                      register_label_taxonomy)


@register_label_taxonomy('scannet20')
class ScanNet20(BaseLabelTaxonomy):

    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')

    PALETTE = (
        (174, 199, 232),  # wall
        (152, 223, 138),  # floor
        (31, 119, 180),   # cabinet
        (255, 187, 120),  # bed
        (188, 189, 34),   # chair
        (140, 86, 75),    # sofa
        (255, 152, 150),  # table
        (214, 39, 40),    # door
        (197, 176, 213),  # window
        (148, 103, 189),  # bookshelf
        (196, 156, 148),  # picture
        (23, 190, 207),   # counter
        (247, 182, 210),  # desk
        (219, 219, 141),  # curtain
        (255, 127, 14),   # refrigerator
        (158, 218, 229),  # shower curtain
        (44, 160, 44),    # toilet
        (112, 128, 144),  # sink
        (227, 119, 194),  # bathtub
        (82, 84, 163)     # otherfurniture
    )

    def __init__(self):
        super().__init__()


@register_label_mapping('scannet40-to-scannet20')
def scannet40_to_scannet20_mapping():
    return [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], 
            [14], [16], [24], [28], [33], [34], [36], [39]]


@register_dataset('scannet')
class ScanNetScene(BaseScene):

    def __init__(self,
                 data_root,
                 img_labels_dir,
                 label_taxonomy='scannet20',
                 img_labels_suffix='.png',
                 img_labels_mapping=None):

        super().__init__(
            data_root,
            img_labels_dir,
            label_taxonomy,
            img_labels_suffix=img_labels_suffix,
            img_labels_mapping=img_labels_mapping)

    def _load_img_intrinsics_info(self, data_root):
        intrinsics_path = os.path.join(data_root, '_info.txt')
        intrinsics = {}
        with open(intrinsics_path, "r") as info_file:
            for line in info_file.readlines():
                s = line.split(" = ")
                if s[0] not in ('m_calibrationColorIntrinsic',
                                'm_calibrationDepthIntrinsic'):
                    continue
                else:
                    data = np.array([
                        np.float32(num_str) for num_str in s[1].split()
                    ]).reshape((4, 4))
                    intrinsics[s[0]] = data[:3, :3]
        return intrinsics

    def _load_img_extrinsics_info(self, data_root):
        poses_dir = os.path.join(data_root, 'poses')
        return get_sorted_file_list(poses_dir, file_ext='.txt')

    def _load_img_depths_info(self, data_root):
        depths_dir = os.path.join(data_root, 'depths')
        return get_sorted_file_list(depths_dir, file_ext='.pgm')

    def _load_pcloud_info(self, data_root):
        scene_name = os.path.basename(data_root)
        return os.path.join(data_root, scene_name + '_vh_clean_2.ply')

    def load_img_depth(self, idx):
        depth = np.array(Image.open(self.img_depths_info[idx]))
        return depth / 1000

    def load_img_intrinsics(self, idx):
        return self.img_intrinsics_info[
            'm_calibrationColorIntrinsic'], self.img_intrinsics_info[
                'm_calibrationDepthIntrinsic']

    def load_img_extrinsics(self, idx):
        pose_path = self.img_extrinsics_info[idx]
        with open(pose_path, "r") as pose_file:
            T_W_C = np.identity(4)
            for i in range(4):
                line = pose_file.readline().split()
                for j in range(4):
                    T_W_C[i, j] = line[j]
        return np.linalg.inv(T_W_C)
