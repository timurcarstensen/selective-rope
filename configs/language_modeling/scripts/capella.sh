#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --mem-per-gpu={mem_per_gpu}
#SBATCH --cpus-per-gpu={cpus_per_gpu}
#SBATCH --time={time_limit}
#SBATCH --output={logs_dir}/{job_name}_%j_out.log
#SBATCH --error={logs_dir}/{job_name}_%j_err.log
#SBATCH --exclude={exclude}
#SBATCH --mail-user={mail_user}
#SBATCH --mail-type={mail_type}
#SBATCH --licenses={licenses}

set -euxo pipefail

project_home={project_home}
code_dir={code_dir}

cd $code_dir

ml release/24.10 || true
ml CUDA/12.8.0 || true

mkdir -p /tmp/.triton/autotune

export TRITON_CACHE_DIR=/tmp/.triton/autotune


export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export NCCL_DEBUG=INFO
export NCCL_IB_RETRY_CNT=10
export NCCL_MIN_NCHANNELS=11
export NCCL_TREE_THRESHOLD=4294967296
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_DISTRIBUTED_TIMEOUT=300  # 30 minutes in seconds
export TORCHELASTIC_MAX_FAILED_CONNECTIONS=60
export TORCH_DISTRIBUTED_HEARTBEAT_TIMEOUT=300
export TORCH_DISTRIBUTED_COODINATOR_TIMEOUT=300
export OMP_NUM_THREADS=18

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${{nodes_array[0]}}

export RDZV_HOST=$head_node
export RDZV_PORT=29400

echo "head_node=$head_node"

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

echo NPROC_PER_NODE=$NPROC_PER_NODE

export WANDB_OFFLINE=true
export WANDB_RESUME={wandb_resume}
export WANDB_PROJECT={wandb_project}
export WANDB_ENTITY={wandb_entity}
export WANDB_RUN_GROUP={wandb_group}
export WANDB_DIR={wandb_dir}
export WANDB_NAME={wandb_run_name}

unset SLURM_CPUS_PER_TASK

srun uv run --frozen --no-sync --project $project_home torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    train.py --cfg {trainer_config_path}
