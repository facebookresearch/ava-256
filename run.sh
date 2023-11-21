#!/bin/bash

# $1 number of io workers per GPU
# $2 CONFIG_FILE

#SBATCH
#SBATCH --gpus-per-node 8
#SBATCH --ntasks-per-node 8
#SBATCH --partition learn
#SBATCH --time 7-00:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

echo " ARG2: number of io workers " $1
echo " ARG3: CONFIG FILE  " $2

srun bash srun-wrapper.sh $1 $2
