#!/bin/bash

set -e

CONFIG_PATH=${1:-configs/hotpot_e5_sage.yml}

echo "Training GRAFT with config: $CONFIG_PATH"

python -m graft.train.train "$CONFIG_PATH"

echo "Training complete!"
