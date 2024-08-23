# Copyright (c) 2021, ETHZ V4RL. All rights reserved.
# Licensed under the BSD 3-Clause License.

from abc import ABCMeta, abstractmethod

import numpy as np
import open3d as o3d
from PIL import Image

from ..frame import Frame
from ..utils import get_sorted_file_list
from .builder import build_label_taxonomy, get_label_mapping


class BaseLabelTaxonomy(object):

    CLASSES = ()
    PALETTE = ()

    def __init__(self):
        self.num_classes = len(self.CLASSES)


class BaseScene(object, metaclass=ABCMeta):

    def __init__(self,
                 data_root,
                 img_labels_dir,
                 label_taxonomy,
                 img_labels_suffix='.png',
                 img_labels_mapping=None):
        self.img_intrinsics_info = self._load_img_intrinsics_info(data_root)
        self.img_extrinsics_info = self._load_img_extrinsics_info(data_root)
        self.img_depths_info = self._load_img_depths_info(data_root)
        assert (len(self.img_extrinsics_info) == len(self.img_depths_info))
        self.img_labels_info = self._load_img_labels_info(img_labels_dir,
                                                     img_labels_suffix)
        assert (len(self.img_extrinsics_info) == len(self.img_labels_info))
        self.pcloud_info = self._load_pcloud_info(data_root)
        
        self.label_taxonomy = build_label_taxonomy(label_taxonomy)
        self.num_classes = self.label_taxonomy.num_classes
        self.img_labels_mapping = self._set_label_mapping(img_labels_mapping)

    @abstractmethod
    def _load_img_intrinsics_info(self, data_root):
        pass

    @abstractmethod
    def _load_img_extrinsics_info(self, data_root):
        pass

    @abstractmethod
    def _load_img_depths_info(self, data_root):
        pass

    def _load_img_labels_info(self, img_labels_dir, img_labels_suffix):
        return get_sorted_file_list(img_labels_dir, file_ext=img_labels_suffix)

    @abstractmethod
    def _load_pcloud_info(self, data_root):
        pass

    def load_frame(self, idx):
        labels = self.load_img_labels(idx)
        depth = self.load_img_depth(idx)
        intrinsics_rgb, intrinsics_depth = self.load_img_intrinsics(idx)
        extrinsics = self.load_img_extrinsics(idx)
        return Frame(labels, depth, intrinsics_rgb, intrinsics_depth,
                     extrinsics)

    def load_img_labels(self, idx):
        labels = np.array(Image.open(self.img_labels_info[idx]))
        if self.img_labels_mapping is not None:
            labels = self._remap_labels(labels, self.img_labels_mapping)
        return labels

    @abstractmethod
    def load_img_depth(self, idx):
        pass

    @abstractmethod
    def load_img_intrinsics(self, idx):
        pass

    @abstractmethod
    def load_img_extrinsics(self, idx):
        pass

    def load_point_cloud(self):
        return o3d.io.read_point_cloud(self.pcloud_info)

    def _set_label_mapping(self, label_mapping_name):
        label_mapping = None
        if label_mapping_name is not None:
            label_mapping = get_label_mapping(label_mapping_name)
            if isinstance(label_mapping, list):
                assert (len(label_mapping) == self.num_classes)
                label_mapping = {
                    src_id: tgt_id
                    for tgt_id, src_ids in enumerate(label_mapping)
                    for src_id in src_ids
                }
            elif isinstance(label_mapping, dict):
                tgt_ids = label_mapping.values()
                assert (len(set(tgt_ids)) == self.num_classes
                        and max(tgt_ids) == self.num_classes - 1)
        return label_mapping

    def _remap_labels(self, labels, label_mapping):
        labels_remapped = np.full(labels.shape, self.num_classes, 
                                  dtype=np.uint8)
        for k, v in label_mapping.items():
            labels_remapped[labels == k] = v
        return labels_remapped
