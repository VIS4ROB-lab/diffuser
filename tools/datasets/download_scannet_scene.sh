#!/bin/bash

# Copyright (c) 2021, ETHZ V4RL. All rights reserved.
# Licensed under the BSD 3-Clause License.
# Usage: ./download_scannet_scene.sh [scene_id] [scannet_data_dir] [scannet_repo_dir]

SCENE="$1"
SCANNET_DATA_DIR="$(realpath $2)"
SCANNET_REPO_DIR="$(realpath $3)"

# Check if scene is already downloaded
if [ -d "$SCANNET_DATA_DIR"/"$SCENE" ]; then
    echo "ERROR: Directory for this scene already exists!"
    exit 1
fi

ROOT="$(dirname $(dirname $(dirname $(realpath $0))))"

# Download scene
python "$ROOT"/tools/datasets/download-scannet.py -o "$SCANNET_DATA_DIR" --id "$SCENE"

cd "$SCANNET_DATA_DIR"
mv scans/"$SCENE" .
[ "$(ls -A scans)" ] || rm -d scans
cd "$SCENE"

# Extract raw RGB-D data
mkdir raw 
"$SCANNET_REPO_DIR"/SensReader/c++/sens "$SCENE".sens raw

# Organize extracted data
mkdir poses
mkdir images
mkdir depths
mv raw/*.color.jpg images
mv raw/*.depth.pgm depths
mv raw/*.pose.txt poses
mv raw/_info.txt .
rm -r raw