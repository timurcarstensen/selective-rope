#!/bin/bash
#SBATCH --job-name=copying
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=06:00:00
#SBATCH --output={logs_dir}/copying_%j_out.log
#SBATCH --error={logs_dir}/copying_%j_err.log
#SBATCH --licenses=cat

set -euxo pipefail

project_home={project_home}
code_dir={code_dir}

cd $code_dir

ml release/24.10 || true
ml CUDA/12.8.0 || true

mkdir -p /tmp/.triton/autotune
export TRITON_CACHE_DIR=/tmp/.triton/autotune

export WANDB_OFFLINE=true
export WANDB_RESUME={wandb_resume}
export WANDB_PROJECT={wandb_project}
export WANDB_ENTITY={wandb_entity}
export WANDB_RUN_GROUP={wandb_group}
export WANDB_DIR={wandb_dir}
export WANDB_NAME={wandb_run_name}

uv run --frozen --no-sync --project $project_home \
    python -m copying.main --cfg {trainer_config_path}
