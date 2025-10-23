#!/bin/bash

set -e

ENCODER_PATH=$1
INDEX_PATH=$2
CONFIG_PATH=$3
OUTPUT_PATH=$4

echo "Evaluating retriever..."

python -c "
import logging, yaml
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from graft.eval.eval_retriever import evaluate_retriever
with open('$CONFIG_PATH') as f:
    config = yaml.safe_load(f)
evaluate_retriever('$ENCODER_PATH', '$INDEX_PATH', config, '$OUTPUT_PATH')
"

echo "Evaluation complete: $OUTPUT_PATH"
