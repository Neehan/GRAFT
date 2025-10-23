#!/bin/bash
# Evaluate GRAFT: embed corpus → build FAISS → evaluate

set -e

ENCODER_PATH=$1
CONFIG_PATH=$2
OUTPUT_DIR=$3
SPLIT=${4:-validation}

if [ -z "$ENCODER_PATH" ] || [ -z "$CONFIG_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: bash scripts/run_eval_graft.sh ENCODER_PATH CONFIG_PATH OUTPUT_DIR [SPLIT]"
    echo "Example: bash scripts/run_eval_graft.sh outputs/graft_hotpot_e5_sage/encoder_best.pt configs/hotpot_e5_sage.yml outputs/eval_graft validation"
    exit 1
fi

echo "=== GRAFT Evaluation Pipeline ==="
echo "Encoder: $ENCODER_PATH"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "Split: $SPLIT"
echo ""

mkdir -p "$OUTPUT_DIR"

EMBEDDINGS_PATH="$OUTPUT_DIR/embeddings.npy"
INDEX_PATH="$OUTPUT_DIR/index.faiss"
RESULTS_PATH="$OUTPUT_DIR/results.json"

# Step 1: Embed corpus with trained encoder
echo "Step 1/3: Embedding corpus..."
python -m graft.eval.embed_corpus "$ENCODER_PATH" "$CONFIG_PATH" "$EMBEDDINGS_PATH"
echo ""

# Step 2: Build FAISS index
echo "Step 2/3: Building FAISS index..."
python -m graft.eval.build_faiss "$EMBEDDINGS_PATH" "$CONFIG_PATH" "$INDEX_PATH"
echo ""

# Step 3: Evaluate
echo "Step 3/3: Evaluating GRAFT..."
python -m graft.eval.evaluate \
    --method graft \
    --encoder-path "$ENCODER_PATH" \
    --faiss-index "$INDEX_PATH" \
    --config "$CONFIG_PATH" \
    --output "$RESULTS_PATH" \
    --split "$SPLIT"
echo ""

echo "=== Evaluation complete! ==="
echo "Results: $RESULTS_PATH"
cat "$RESULTS_PATH"

