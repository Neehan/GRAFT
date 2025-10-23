#!/bin/bash

set -e

ENCODER_PATH=$1
CONFIG_PATH=$2
OUTPUT_PATH=$3

echo "Embedding corpus with encoder: $ENCODER_PATH"

python -c "
import logging, yaml
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from graft.eval.embed_corpus import embed_corpus
with open('$CONFIG_PATH') as f:
    config = yaml.safe_load(f)
embed_corpus('$ENCODER_PATH', config, '$OUTPUT_PATH')
"

echo "Corpus embedded: $OUTPUT_PATH"
