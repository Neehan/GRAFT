#!/bin/bash

set -e

CONFIG=${1:-configs/hotpot_e5_sage.yml}
SPLIT=${2:-train}

echo "Preparing HotpotQA data from $SPLIT split..."
python -m graft.data.prepare "$CONFIG" "$SPLIT"
