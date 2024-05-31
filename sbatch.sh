#!/bin/bash
#SBATCH --partition=learn
#SBATCH --time=1:00:00
#SBATCH --job-name=AVA256_TRAIN
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=200G
#SBATCH --gpus-per-task=8
#SBATCH --output=/home/%u/rsc/ava-256/logs/slurm-%j.out
#SBATCH --error=/home/%u/rsc/ava-256/logs/slurm-%j.err

source /uca/conda-envs/activate-latest

export GLOG_minloglevel=2
export NCCL_ASYNC_ERROR_HANDLING=1
export DB_CACHE_DIR=/shared/airstore_index/avatar_index_cache

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# export NCCL_DEBUG=INFO

cd /home/$USER/rsc/ava-256/ || exit 1
srun python ddp-train.py --masteraddress ${MASTER_ADDR} --masterport ${MASTER_PORT} --config configs/config-256.yaml
