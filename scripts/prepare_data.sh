#!/bin/bash

set -e

OUTPUT_DIR=${1:-./datasets/hotpot}
SPLIT=${2:-train}

echo "Preparing HotpotQA data from $SPLIT split..."
python -m graft.data.prepare "$OUTPUT_DIR" "$SPLIT"
