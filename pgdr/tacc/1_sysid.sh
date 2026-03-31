#!/bin/bash
#SBATCH -J pgdr_sysid
#SBATCH -o pgdr/results/logs/sysid_%j.out
#SBATCH -e pgdr/results/logs/sysid_%j.err
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

# Ensure all Python deps are installed
pip install --no-cache-dir --target=$WORK/python_packages \
    optax flax 2>/dev/null || true

cd /home1/11386/mohammada/posterior-guided-dr
MODEL_XML="/home1/11386/mohammada/posterior-guided-dr/mujoco_menagerie/booster_t1/scene.xml"
RESULTS_DIR="pgdr/results"
DATA_DIR="pgdr/data"
CONFIG="pgdr/config/sysid_config.yaml"

mkdir -p "$RESULTS_DIR/logs" "$DATA_DIR"

echo ">>> STEP 1: Sensitivity Analysis"
python -m pgdr.sensitivity \
    --model-xml "$MODEL_XML" \
    --perturbation 0.2 \
    --output "$RESULTS_DIR/sensitivity_ranking.json" \
    --keep-top-k 40
echo "    DONE."

echo ">>> STEP 2: Create Sim A ground truth"
python -m pgdr.sysid create-sim-a \
    --model-xml "$MODEL_XML" \
    --perturbation 0.15 \
    --output "$DATA_DIR/sim_a_params.npy"
echo "    DONE."

echo ">>> STEP 3: Collect reference trajectories"
python -m pgdr.sysid collect-reference \
    --model-xml "$MODEL_XML" \
    --sim-a-params "$DATA_DIR/sim_a_params.npy" \
    --config "$CONFIG" \
    --output "$DATA_DIR/sim_a_reference.npz"
echo "    DONE."

echo ">>> STEP 4: CMA-ES System Identification"
python -m pgdr.sysid identify \
    --model-xml "$MODEL_XML" \
    --reference "$DATA_DIR/sim_a_reference.npz" \
    --config "$CONFIG" \
    --output-p-star "$RESULTS_DIR/p_star.npy" \
    --output-sigma "$RESULTS_DIR/Sigma.npy"
echo "    DONE."

echo "Sysid complete — $(date)"
