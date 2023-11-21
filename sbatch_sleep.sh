#!/bin/sh
#SBATCH --partition=learn
#SBATCH --time=7-00:00:00
#SBATCH --job-name=DEV
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=225G
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --error=/home/%u/logs/job.%J.err
#SBATCH --output=/home/%u/logs/job.%J.out

srun sleep 7d
