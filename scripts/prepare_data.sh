#!/bin/bash

set -e

# Ensure conda libs are prioritized for GLIBCXX compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CONFIG=${1:-configs/hotpot_e5_sage.yml}
SPLIT=${2:-train}

echo "Preparing HotpotQA data from $SPLIT split..."
python -m graft.data.prepare "$CONFIG" "$SPLIT"
