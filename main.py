# Copyright (c) 2021, ETHZ V4RL. All rights reserved.
# Licensed under the BSD 3-Clause License.

import argparse
import configparser
import os

import numpy as np

from diffuser.dataloader import DataLoader
from diffuser.datasets.builder import build_dataset
from diffuser.diffuser import Diffuser
from diffuser.output import save_data_array, visualize_3d_segmentation


def main():
    # Load config parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(args.config)

    # Initialization
    dataset_config = config['dataset']
    dataset = build_dataset(
        dataset_config.get('dataset_name'),
        dataset_config.get('dataset_dir'), 
        dataset_config.get('img_labels_dir'),
        dataset_config.get('label_taxonomy'),
        dataset_config.get('img_labels_suffix', fallback='.png'),
        dataset_config.get('img_labels_mapping', fallback=None))
    
    dataloader_config = config['dataloader']
    dataloader = DataLoader(
        dataset,
        dataloader_config.getint('img_step'),
        dataloader_config.getfloat('pcloud_voxel_size'),
        dataloader_config.getfloat('pcloud_normals_radius'),
        dataloader_config.getint('pcloud_normals_max_nn'))
    
    diffuser_config = config['diffuser']
    diffuser = Diffuser(
        diffuser_config.getint('num_pt_neighbors'),
        diffuser_config.getfloat('distance_mu'),
        diffuser_config.getfloat('normals_mu'),
        diffuser_config.getfloat('px_to_pt_weight'))

    # Execution
    points, normals = dataloader.point_cloud()
    frames = dataloader.frames()
    labels, _ = diffuser.run(
        points, normals, frames, dataset.num_classes, 
        max_iters=diffuser_config.getint('max_iters'))

    # Output
    experiment_config = config['experiment']
    output_dir = experiment_config.get('output_dir')
    experiment_name = experiment_config.get('experiment_name')
    output_npy = os.path.join(output_dir, experiment_name + ".npy")
    save_data_array(np.column_stack((points, labels)), output_npy)
    output_ply = os.path.join(output_dir, experiment_name + ".ply")
    visualize_3d_segmentation(points, labels, dataset.label_taxonomy, 
                              save_as=output_ply)


if __name__ == '__main__':
    main()
