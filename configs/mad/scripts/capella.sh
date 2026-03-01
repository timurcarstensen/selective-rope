#!/bin/bash
#SBATCH --job-name=mad
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=01-00:00:00
#SBATCH --output={logs_dir}/mad_%j_out.log
#SBATCH --error={logs_dir}/mad_%j_err.log
#SBATCH --licenses=cat

set -euxo pipefail

project_home={project_home}
code_dir={code_dir}

cd $code_dir

ml release/24.10 || true
ml CUDA/12.8.0 || true

mkdir -p /tmp/.triton/autotune
export TRITON_CACHE_DIR=/tmp/.triton/autotune
export PYTHONIOENCODING=utf-8

export WANDB_RESUME={wandb_resume}
export WANDB_PROJECT={wandb_project}
export WANDB_ENTITY={wandb_entity}
export WANDB_RUN_GROUP={wandb_group}
export WANDB_DIR={wandb_dir}
export WANDB_NAME={wandb_run_name}

export UV_PROJECT_ENVIRONMENT=$project_home/.venv

uv run --frozen --no-sync \
    python synthetic_tasks/mad/src/mad/benchmark.py \
    --config-path {config_path} \
    --config-name config \
    data_path=$project_home/data/mad \
    log_base_path={results_dir}
