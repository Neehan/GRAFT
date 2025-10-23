#!/bin/bash
# Augment graph with kNN edges for better GNN connectivity

set -e

# Ensure conda libs are prioritized for GLIBCXX compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

INPUT_GRAPH=${1}
CONFIG=${2}
K=${3:-5}
KNN_ONLY=${4:-false}

if [ -z "$INPUT_GRAPH" ] || [ -z "$CONFIG" ]; then
    echo "Usage: bash scripts/augment_graph.sh INPUT_GRAPH CONFIG [K] [knn-only]"
    echo ""
    echo "Examples:"
    echo "  # Hybrid: structural + kNN edges"
    echo "  bash scripts/augment_graph.sh datasets/hotpot/page_graph.pt configs/hotpot_e5_sage.yml 5"
    echo ""
    echo "  # Ablation: kNN only"
    echo "  bash scripts/augment_graph.sh datasets/hotpot/page_graph.pt configs/hotpot_e5_sage.yml 5 knn-only"
    exit 1
fi

if [ "$KNN_ONLY" = "knn-only" ]; then
    OUTPUT_GRAPH="${INPUT_GRAPH%.pt}_knn_only${K}.pt"
    KNN_FLAG="--knn-only"
else
    OUTPUT_GRAPH="${INPUT_GRAPH%.pt}_knn${K}.pt"
    KNN_FLAG=""
fi

echo "=== kNN Graph Augmentation ==="
echo "Input:  $INPUT_GRAPH"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_GRAPH"
echo "k:      $K nearest neighbors"
echo "Mode:   $([ "$KNN_ONLY" = "knn-only" ] && echo "kNN-only (ablation)" || echo "Hybrid")"
echo ""

python -m graft.data.augment_graph \
    "$INPUT_GRAPH" \
    "$OUTPUT_GRAPH" \
    "$CONFIG" \
    --k "$K" \
    $KNN_FLAG

echo ""
echo "=== Augmentation complete! ==="
echo "Output: $OUTPUT_GRAPH"
echo ""
echo "Update your config:"
echo "  data.graph_path: $OUTPUT_GRAPH"
