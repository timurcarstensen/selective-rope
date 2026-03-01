#!/bin/bash
#SBATCH --job-name=zoology
#SBATCH --output=/home/carstent/r1/ultra-delta/logs/zoology/slurm_logs/zoology_%A_%a.out
#SBATCH --error=/home/carstent/r1/ultra-delta/logs/zoology/slurm_logs/zoology_%A_%a.err
#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:8,localtmp:50

# Create logs directory if it doesn't exist
mkdir -p "$TMPDIR"/.triton/autotune

export TRITON_CACHE_DIR=$TMPDIR/.triton/autotune

# Define arrays of parameters
cd zoology || exit 1

uv run --frozen --no-sync python -m zoology.launch zoology/experiments/iclr24_zoology_figure2/configs.py -p
