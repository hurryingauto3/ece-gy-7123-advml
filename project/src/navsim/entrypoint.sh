#!/bin/bash
set -e

# Check if the dataset is already present.
# Here we check if the 'trainval' folder under navsim_logs exists and is non-empty.
if [ ! -d "/navsim_workspace/dataset/navsim_logs/trainval" ] || [ -z "$(ls -A /navsim_workspace/dataset/navsim_logs/trainval)" ]; then
    echo "Dataset not found in /navsim_workspace/dataset. Starting download..."
    cd /navsim_workspace/navsim/download
    ./download_maps
    ./download_mini
    ./download_trainval
    ./download_test
    ./download_private_test_e2e
    ./download_warmup_synthetic_scenes
    ./download_navtrain
else
    echo "Dataset already exists. Skipping download."
fi

# Execute the container's command (default is bash)
exec "$@"