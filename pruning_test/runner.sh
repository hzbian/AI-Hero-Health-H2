#!/bin/bash

#SBATCH --job-name=AI-HERO_health_baseline_training
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=20:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/im9193-H2/AI-Hero-Health-H2/pruning_test/output/output.txt

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=8

group_workspace=/hkfs/work/workspace/scratch/im9193-H2

source ${group_workspace}/health_baseline_env_dm/bin/activate
python ${group_workspace}/AI-Hero-Health-H2/pruning_test/finetune.py

