#!/bin/bash
# =============================================================================
# PGDR Full Pipeline — Run Everything End-to-End
# =============================================================================
# Usage on TACC Lonestar6:
#   sbatch pgdr/run_all.sh
#
# Or interactively (for debugging):
#   bash pgdr/run_all.sh
#
# All results go to pgdr/results/ and pgdr/checkpoints/
# =============================================================================

#SBATCH -J pgdr_full_pipeline
#SBATCH -o pgdr/results/logs/pipeline_%j.out
#SBATCH -e pgdr/results/logs/pipeline_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A IRI26006

set -euo pipefail

# ---- TACC module setup ----
module load gcc/11.2.0
module load cuda/12.0
module load python3/3.9.7

# ---- Environment ----
export PYTHONPATH=$WORK/python_packages:$PYTHONPATH
export MUJOCO_PATH=$WORK/mujoco
export LD_LIBRARY_PATH=$WORK/mujoco/lib:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$TACC_CUDA_DIR"
export JAX_PLATFORMS="gpu"

# ---- Paths ----
cd /home1/11386/mohammada/posterior-guided-dr
MODEL_XML="/home1/11386/mohammada/posterior-guided-dr/mujoco_menagerie/booster_t1/t1.xml"

RESULTS_DIR="pgdr/results"
CHECKPOINTS_DIR="pgdr/checkpoints"
DATA_DIR="pgdr/data"
CONFIG="pgdr/config/sysid_config.yaml"
TRAIN_CONFIG="pgdr/config/train_config.yaml"

# Create output directories
mkdir -p "$RESULTS_DIR/logs" "$CHECKPOINTS_DIR" "$DATA_DIR"

echo "============================================================"
echo "PGDR Full Pipeline"
echo "Model XML:    $MODEL_XML"
echo "Results dir:  $RESULTS_DIR"
echo "Started:      $(date)"
echo "============================================================"

# =============================================================================
# STEP 1: Sensitivity Analysis
# =============================================================================
echo ""
echo ">>> STEP 1/6: Sensitivity Analysis"
echo "    Output: $RESULTS_DIR/sensitivity_ranking.json"

python -m pgdr.sensitivity \
    --model-xml "$MODEL_XML" \
    --perturbation 0.2 \
    --output "$RESULTS_DIR/sensitivity_ranking.json" \
    --keep-top-k 40

echo "    DONE."

# =============================================================================
# STEP 2: Create Sim A (perturbed ground truth for sim-to-sim validation)
# =============================================================================
echo ""
echo ">>> STEP 2/6: Create Sim A ground truth"
echo "    Output: $DATA_DIR/sim_a_params.npy"

python -m pgdr.sysid create-sim-a \
    --model-xml "$MODEL_XML" \
    --perturbation 0.15 \
    --output "$DATA_DIR/sim_a_params.npy"

echo "    DONE."

# =============================================================================
# STEP 3: Collect reference trajectories from Sim A
# =============================================================================
echo ""
echo ">>> STEP 3/6: Collect reference trajectories"
echo "    Output: $DATA_DIR/sim_a_reference.npz"

python -m pgdr.sysid collect-reference \
    --model-xml "$MODEL_XML" \
    --sim-a-params "$DATA_DIR/sim_a_params.npy" \
    --config "$CONFIG" \
    --output "$DATA_DIR/sim_a_reference.npz"

echo "    DONE."

# =============================================================================
# STEP 4: Run CMA-ES identification (produces p* and Σ)
# =============================================================================
echo ""
echo ">>> STEP 4/6: CMA-ES System Identification"
echo "    Output: $RESULTS_DIR/p_star.npy, $RESULTS_DIR/Sigma.npy"

python -m pgdr.sysid identify \
    --model-xml "$MODEL_XML" \
    --reference "$DATA_DIR/sim_a_reference.npz" \
    --config "$CONFIG" \
    --output-p-star "$RESULTS_DIR/p_star.npy" \
    --output-sigma "$RESULTS_DIR/Sigma.npy"

echo "    DONE."

# =============================================================================
# STEP 5: Train policies under all conditions (C1-C4)
# =============================================================================
echo ""
echo ">>> STEP 5/6: Train all conditions (C1 uniform, C2 sysid, C3 isotropic, C4 PGDR)"
echo "    Output: $CHECKPOINTS_DIR/<condition>_seed<N>/"

python -m pgdr.train_all_conditions train \
    --model-xml "$MODEL_XML" \
    --config "$TRAIN_CONFIG" \
    --results-dir "$RESULTS_DIR" \
    --checkpoint-dir "$CHECKPOINTS_DIR"

echo "    DONE."

# =============================================================================
# STEP 6: Evaluate all conditions + generate plots
# =============================================================================
echo ""
echo ">>> STEP 6/6: Evaluation and plotting"

# 6a: Covariance calibration (does Σ reflect real uncertainty?)
echo "    6a: Covariance calibration"
echo "    Output: $RESULTS_DIR/calibration.json"

python -m pgdr.evaluate covariance-calibration \
    --model-xml "$MODEL_XML" \
    --p-star "$RESULTS_DIR/p_star.npy" \
    --sigma "$RESULTS_DIR/Sigma.npy" \
    --p-true "$DATA_DIR/sim_a_params.npy" \
    --output "$RESULTS_DIR/calibration.json"

# 6b: Sim-to-sim evaluation (nominal)
echo "    6b: Sim-to-sim evaluation (nominal)"
echo "    Output: $RESULTS_DIR/eval_results.json"

python -m pgdr.evaluate sim2sim \
    --model-xml "$MODEL_XML" \
    --p-star "$RESULTS_DIR/p_star.npy" \
    --sigma "$RESULTS_DIR/Sigma.npy" \
    --p-true "$DATA_DIR/sim_a_params.npy" \
    --checkpoints "$CHECKPOINTS_DIR" \
    --output "$RESULTS_DIR/eval_results.json"

# 6c: Sim-to-sim evaluation (with perturbations)
echo "    6c: Sim-to-sim evaluation (payload + friction perturbation)"
echo "    Output: $RESULTS_DIR/eval_results_perturbed.json"

python -m pgdr.evaluate sim2sim \
    --model-xml "$MODEL_XML" \
    --p-star "$RESULTS_DIR/p_star.npy" \
    --sigma "$RESULTS_DIR/Sigma.npy" \
    --p-true "$DATA_DIR/sim_a_params.npy" \
    --checkpoints "$CHECKPOINTS_DIR" \
    --perturbation both \
    --output "$RESULTS_DIR/eval_results_perturbed.json"

# 6d: Generate all plots
echo "    6d: Generating figures"
echo "    Output: $RESULTS_DIR/*.png"

python -m pgdr.plotting --results "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  $RESULTS_DIR/"
echo "    ├── sensitivity_ranking.json   (Step 1)"
echo "    ├── p_star.npy                 (Step 4: identified params)"
echo "    ├── Sigma.npy                  (Step 4: covariance matrix)"
echo "    ├── calibration.json           (Step 6a: does Σ work?)"
echo "    ├── eval_results.json          (Step 6b: C1-C4 comparison)"
echo "    ├── eval_results_perturbed.json(Step 6c: robustness test)"
echo "    └── *.png                      (Step 6d: paper figures)"
echo ""
echo "  $CHECKPOINTS_DIR/"
echo "    ├── C1_uniform_dr_seed*/       (trained policies)"
echo "    ├── C2_pure_sysid_seed*/"
echo "    ├── C3_isotropic_seed*/"
echo "    └── C4_pgdr_*_seed*/"
echo ""
echo "  $DATA_DIR/"
echo "    ├── sim_a_params.npy           (ground truth for validation)"
echo "    └── sim_a_reference.npz        (reference trajectories)"
