#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -A mit_general
#SBATCH --job-name=graft_eval
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Helps with preprocessing and evaluation overhead
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

# Use all available CPUs for preprocessing helpers that can leverage OpenMP
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

CONFIG_PATH=$1

if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: bash scripts/run_eval_graft.sh CONFIG_PATH"
    echo "Example: bash scripts/run_eval_graft.sh configs/hotpot_e5_sage.yml"
    echo ""
    echo "Config controls: checkpoint (best/final), split (train/dev/test), output_dir, etc."
    exit 1
fi

echo "=== GRAFT Evaluation Pipeline ==="
echo "Config: $CONFIG_PATH"
echo "GPUs available: $(nvidia-smi --list-gpus | wc -l)"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# Python script determines encoder path, output path, split from config
python -m graft.eval.evaluate --config "$CONFIG_PATH"
