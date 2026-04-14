#!/bin/bash
#SBATCH -J pgdr_C2_pure_sysid_seed0
#SBATCH -o pgdr/tacc/jobs/logs/C2_pure_sysid_seed0_%j.out
#SBATCH -e pgdr/tacc/jobs/logs/C2_pure_sysid_seed0_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A al62479

# TACC Lonestar6 module setup
module load gcc/11.2.0
module load cuda/12.0
module load python/3.12.11

# Activate environment
source $WORK/pgdr_env/bin/activate

# Set JAX to use GPU
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$TACC_CUDA_DIR"
export JAX_PLATFORMS="gpu"

cd $WORK/posterior-guided-dr

python -m pgdr.train_all_conditions train \
    --model-xml t1 \
    --config pgdr/config/train_config.yaml \
    --results-dir pgdr/results/20260409_162150_friction/ \
    --checkpoint-dir pgdr/checkpoints \
    --conditions C2_pure_sysid \
    --seeds 0
