#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -A mit_general
#SBATCH --job-name=graft_prepare
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=256GB  # Required for 5.2M corpus with chunking (~10-15M nodes)
#SBATCH -t 24:00:00

module load miniforge/24.3.0-0
conda activate graft

set -e

PROJECT_ROOT="/home/notadib/projects/GRAFT"
cd ${PROJECT_ROOT}

# Ensure conda libs are prioritized for GLIBCXX compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Ensure GPUs are visible (needed for embedding corpus if using kNN augmentation)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

# Use all available CPUs for FAISS index building (multi-threaded)
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

CONFIG=${1:-configs/hotpot_e5_sage.yml}
SPLIT=${2:-train}

echo "=== Preparing HotpotQA Data ==="
echo "Config: $CONFIG"
echo "Split: $SPLIT"
echo "GPUs available: $(nvidia-smi --list-gpus | wc -l)"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

echo "Preparing HotpotQA data from $SPLIT split..."
python -m graft.data.prepare "$CONFIG" "$SPLIT"

echo ""
echo "=== Data preparation complete! ==="
