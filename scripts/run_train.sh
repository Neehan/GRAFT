#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -A mit_general
#SBATCH --job-name=graft_train
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=128GB
#SBATCH -t 48:00:00

module load miniforge/24.3.0-0
conda activate graft

set -e

PROJECT_ROOT="/home/notadib/projects/GRAFT"
cd ${PROJECT_ROOT}

# Ensure conda libs are prioritized for GLIBCXX compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CONFIG_PATH=${1:-configs/hotpot_e5_sage.yml}

# Debug GPU allocation
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs via nvidia-smi:"
nvidia-smi --list-gpus

# Get actual number of visible GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi

echo "Training GRAFT with config: $CONFIG_PATH"
echo "Using $NUM_GPUS GPUs for training"

export PYTHONWARNINGS="ignore::UserWarning"

accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --dynamo_backend no \
    --multi_gpu \
    --mixed_precision bf16 \
    graft/train/train.py "$CONFIG_PATH"

echo "Training complete!"
