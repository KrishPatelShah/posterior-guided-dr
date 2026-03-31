#!/bin/bash
# Usage: sbatch --export=CONDITION=C1_uniform_dr,SEED=0 pgdr/tacc/2_train.sh
#SBATCH -J pgdr_train
#SBATCH -o pgdr/results/logs/train_%x_%j.out
#SBATCH -e pgdr/results/logs/train_%x_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A IRI26006

set -euo pipefail

module load gcc/11.2.0
module load cuda/12.0
module load python3/3.9.7

export PYTHONPATH=$WORK/python_packages:$PYTHONPATH
export MUJOCO_PATH=$WORK/mujoco
export LD_LIBRARY_PATH=$WORK/mujoco/lib:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$TACC_CUDA_DIR"
export JAX_PLATFORMS="gpu"

cd /home1/11386/mohammada/posterior-guided-dr
MODEL_XML="/home1/11386/mohammada/posterior-guided-dr/mujoco_menagerie/booster_t1/t1.xml"

echo "Training ${CONDITION} seed=${SEED} — $(date)"

python -m pgdr.train_all_conditions train \
    --model-xml "$MODEL_XML" \
    --config pgdr/config/train_config.yaml \
    --results-dir pgdr/results \
    --checkpoint-dir pgdr/checkpoints \
    --conditions "$CONDITION" \
    --seeds "$SEED"

echo "Training ${CONDITION} seed=${SEED} complete — $(date)"
