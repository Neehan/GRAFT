#!/bin/bash

set -e

OUTPUT_DIR=${1:-./data/hotpot}

mkdir -p "$OUTPUT_DIR"

echo "Building graph from HotpotQA train split..."
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from graft.data.build_graph import build_hotpot_graph
build_hotpot_graph('$OUTPUT_DIR/page_graph.pt', 'train')
"

echo "Data preparation complete! Graph saved to $OUTPUT_DIR/page_graph.pt"
