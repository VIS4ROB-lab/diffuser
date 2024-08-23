# Copyright (c) 2021, ETHZ V4RL. All rights reserved.
# Licensed under the BSD 3-Clause License.

DATASETS = {}
LABEL_TAXONOMIES = {}
LABEL_MAPPINGS = {}

def register_dataset(dataset_name):
    def decorator(cls):
        DATASETS[dataset_name] = cls
        return cls     
    return decorator


def register_label_taxonomy(label_taxonomy_name):
    def decorator(cls):
        LABEL_TAXONOMIES[label_taxonomy_name] = cls
        return cls     
    return decorator


def register_label_mapping(label_mapping_name):
    def decorator(fn):
        LABEL_MAPPINGS[label_mapping_name] = fn
        return fn
    return decorator


def build_dataset(dataset_name, *args, **kwargs):
    return DATASETS[dataset_name](*args, **kwargs)


def build_label_taxonomy(taxonomy_name, *args, **kwargs):
    return LABEL_TAXONOMIES[taxonomy_name](*args, **kwargs)


def get_label_mapping(label_mapping_name):
    return LABEL_MAPPINGS[label_mapping_name]()