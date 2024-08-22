#!/bin/bash
#SBATCH --array=1-2
#SBATCH --partition=learn
#SBATCH --time=7-00:00:00
#SBATCH --job-name=AVA256_TRAIN
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=220G
#SBATCH --gpus-per-task=8
#SBATCH --output=/home/%u/rsc/ava-256/logs/slurm/%A_%a.out
#SBATCH --error=/home/%u/rsc/ava-256/logs/slurm/%A_%a.err
#SBATCH --qos=urgent_deadline

source /uca/conda-envs/activate-latest

export GLOG_minloglevel=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export DB_CACHE_DIR=/shared/airstore_index/avatar_index_cache

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

cd /home/$USER/rsc/ava-256/ || exit 1
srun python ddp-train.py configs/config-240.yaml \
--masteraddress ${MASTER_ADDR} \
--masterport ${MASTER_PORT} \
--progress.output_path run-240/${SLURM_ARRAY_TASK_ID}/ \
--progress.tensorboard.output_path run-240/${SLURM_ARRAY_TASK_ID}/tb/
