#!/bin/bash

set -e

# Ensure conda libs are prioritized for GLIBCXX compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CONFIG_PATH=${1:-configs/hotpot_e5_sage.yml}

echo "Training GRAFT with config: $CONFIG_PATH"

python -m graft.train.train "$CONFIG_PATH"

echo "Training complete!"
