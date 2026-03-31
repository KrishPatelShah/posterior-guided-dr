#!/bin/bash
#SBATCH -J pgdr_eval
#SBATCH -o pgdr/results/logs/eval_%j.out
#SBATCH -e pgdr/results/logs/eval_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A IRI26006

set -eo pipefail

module load gcc/11.2.0
module load cuda/12.0
module load python3/3.9.7

export PYTHONPATH=$WORK/python_packages:${PYTHONPATH:-}
export MUJOCO_PATH=$WORK/mujoco
export LD_LIBRARY_PATH=$WORK/mujoco/lib:${LD_LIBRARY_PATH:-}
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$TACC_CUDA_DIR"
export JAX_PLATFORMS=""

cd /home1/11386/mohammada/posterior-guided-dr
MODEL_XML="/home1/11386/mohammada/posterior-guided-dr/mujoco_menagerie/booster_t1/t1.xml"
RESULTS_DIR="pgdr/results"
DATA_DIR="pgdr/data"
CHECKPOINTS_DIR="pgdr/checkpoints"

echo ">>> Covariance calibration"
python -m pgdr.evaluate covariance-calibration \
    --model-xml "$MODEL_XML" \
    --p-star "$RESULTS_DIR/p_star.npy" \
    --sigma "$RESULTS_DIR/Sigma.npy" \
    --p-true "$DATA_DIR/sim_a_params.npy" \
    --output "$RESULTS_DIR/calibration.json"

echo ">>> Sim-to-sim evaluation (nominal)"
python -m pgdr.evaluate sim2sim \
    --model-xml "$MODEL_XML" \
    --p-star "$RESULTS_DIR/p_star.npy" \
    --sigma "$RESULTS_DIR/Sigma.npy" \
    --p-true "$DATA_DIR/sim_a_params.npy" \
    --checkpoints "$CHECKPOINTS_DIR" \
    --output "$RESULTS_DIR/eval_results.json"

echo ">>> Sim-to-sim evaluation (perturbed)"
python -m pgdr.evaluate sim2sim \
    --model-xml "$MODEL_XML" \
    --p-star "$RESULTS_DIR/p_star.npy" \
    --sigma "$RESULTS_DIR/Sigma.npy" \
    --p-true "$DATA_DIR/sim_a_params.npy" \
    --checkpoints "$CHECKPOINTS_DIR" \
    --perturbation both \
    --output "$RESULTS_DIR/eval_results_perturbed.json"

echo ">>> Generating figures"
python -m pgdr.plotting --results "$RESULTS_DIR"

echo "Evaluation complete — $(date)"
