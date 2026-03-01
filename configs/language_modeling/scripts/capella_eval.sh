#!/bin/bash
#SBATCH --job-name=gla_eval
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=90G
#SBATCH --cpus-per-gpu=6
#SBATCH --time=12:00:00
#SBATCH --output={logs_dir}/gla_eval_%j_out.log
#SBATCH --error={logs_dir}/gla_eval_%j_err.log
#SBATCH --exclude=c4,c154,c17
#SBATCH --licenses=cat

set -euo pipefail

project_home={project_home}
code_dir={code_dir}

cd $code_dir

# Use worktree's flash-linear-attention instead of editable install from main repo
export PYTHONPATH=$code_dir/flash-linear-attention:${{PYTHONPATH:-}}

ml release/24.10 || true
ml CUDA/12.8.0 || true

mkdir -p /tmp/.triton/autotune
export TRITON_CACHE_DIR="/tmp/.triton/autotune"

echo "Starting evaluation for run: {output_dir}"

uv run --frozen --no-sync --project $project_home python evals/eval.py \
    --run-dir "{output_dir}" \
    --eval-samples 1000 \
    --run-lm-eval \
    --run-length-extrap \
    --length-extrap-max-len 4096 \
    --length-extrap-dataset codeparrot \
    --log-to-wandb

echo "Evaluation complete. Results saved to {output_dir}/eval_*.json"
