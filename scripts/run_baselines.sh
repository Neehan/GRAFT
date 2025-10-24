#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -A mit_general
#SBATCH --job-name=graft_baselines
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # More CPUs for faster FAISS index building
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=128GB  # Needed for embedding 5.2M corpus
#SBATCH -t 48:00:00

module load miniforge/24.3.0-0
conda activate graft

set -e

PROJECT_ROOT="/home/notadib/projects/GRAFT"
cd ${PROJECT_ROOT}

# Ensure conda libs are prioritized for GLIBCXX compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Ensure all GPUs are visible (SLURM might restrict)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

# Use all available CPUs for FAISS index building
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

CONFIG_PATH=${1:-configs/hotpot_e5_sage.yml}
OUTPUT_DIR=${2:-outputs/baselines_test}
SPLIT=${3:-test}

echo "=== Running Baselines ==="
echo "Config: $CONFIG_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Split: $SPLIT"
echo "GPUs available: $(nvidia-smi --list-gpus | wc -l)"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

mkdir -p "$OUTPUT_DIR"

# 1. Zero-shot E5 (main baseline: same encoder as GRAFT, but untrained)
echo "=== 1/2: Running Zero-shot E5 ==="

ZERO_SHOT_INDEX="$OUTPUT_DIR/zero_shot_index.faiss"

echo "  Step 1/2: Embedding corpus and building FAISS index..."
python -m graft.eval.embed_corpus \
    intfloat/e5-base-v2 \
    "$CONFIG_PATH" \
    "dummy" \
    --index-path "$ZERO_SHOT_INDEX"

echo "  Step 2/2: Evaluating zero-shot E5..."
python -m graft.eval.evaluate \
    --method zero-shot \
    --model-name intfloat/e5-base-v2 \
    --faiss-index "$ZERO_SHOT_INDEX" \
    --config "$CONFIG_PATH" \
    --output "$OUTPUT_DIR/zero_shot_e5_results.json" \
    --split "$SPLIT"
echo ""

# 2. BM25 baseline (sparse retrieval, no FAISS needed)
echo "=== 2/2: Running BM25 ==="
python -m graft.eval.evaluate \
    --method bm25 \
    --config "$CONFIG_PATH" \
    --output "$OUTPUT_DIR/bm25_results.json" \
    --split "$SPLIT"
echo ""

# Optional: Add other dense baselines (uncomment to run)
# echo "=== Running Zero-shot Contriever ==="
# CONTRIEVER_EMBEDDINGS="$OUTPUT_DIR/contriever_embeddings.npy"
# CONTRIEVER_INDEX="$OUTPUT_DIR/contriever_index.faiss"
# python -m graft.eval.embed_corpus facebook/contriever "$CONFIG_PATH" "$CONTRIEVER_EMBEDDINGS"
# python -m graft.eval.build_faiss "$CONTRIEVER_EMBEDDINGS" "$CONFIG_PATH" "$CONTRIEVER_INDEX"
# python -m graft.eval.evaluate \
#     --method zero-shot \
#     --model-name facebook/contriever \
#     --faiss-index "$CONTRIEVER_INDEX" \
#     --config "$CONFIG_PATH" \
#     --output "$OUTPUT_DIR/zero_shot_contriever_results.json" \
#     --split "$SPLIT"

echo "=== Baselines complete! ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary:"
for file in "$OUTPUT_DIR"/*_results.json; do
    echo ""
    echo "$(basename "$file"):"
    cat "$file"
done
