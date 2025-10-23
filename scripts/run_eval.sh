#!/bin/bash

set -e

ENCODER_PATH=$1
CONFIG_PATH=$2
OUTPUT_DIR=$3
SPLIT=${4:-validation}

echo "Running full evaluation pipeline..."
python -m graft.eval.run_eval "$ENCODER_PATH" "$CONFIG_PATH" "$OUTPUT_DIR" "$SPLIT"
