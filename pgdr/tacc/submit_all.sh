#!/bin/bash
# =============================================================================
# PGDR Parallel Pipeline Launcher
# =============================================================================
# Submits the full pipeline as chained SLURM jobs:
#   1) Sysid (1 node, ~30 min)
#   2) 18 training jobs in parallel (18 nodes, ~2-6 hrs each)
#      — automatically start after sysid finishes
#   3) Evaluation (1 node, ~30 min)
#      — automatically starts after ALL training jobs finish
#
# Usage:
#   bash pgdr/tacc/submit_all.sh
# =============================================================================

set -euo pipefail
cd /home1/11386/mohammada/posterior-guided-dr
mkdir -p pgdr/results/logs pgdr/checkpoints pgdr/data

CONDITIONS="C1_uniform_dr C2_pure_sysid C3_isotropic C4_pgdr_0.5 C4_pgdr_1.0 C4_pgdr_2.0"
SEEDS="0 1 2"

# --- Step 1: Submit sysid job ---
SYSID_JOB=$(sbatch --parsable pgdr/tacc/1_sysid.sh)
echo "Submitted sysid job: $SYSID_JOB"

# --- Step 2: Submit 18 training jobs, each depends on sysid ---
TRAIN_JOBS=""
for cond in $CONDITIONS; do
    for seed in $SEEDS; do
        JOB=$(sbatch --parsable \
            --dependency=afterok:${SYSID_JOB} \
            --export=ALL,CONDITION=${cond},SEED=${seed} \
            --job-name="pgdr_${cond}_s${seed}" \
            pgdr/tacc/2_train.sh)
        TRAIN_JOBS="${TRAIN_JOBS}:${JOB}"
        echo "  Submitted training: ${cond} seed=${seed} -> job $JOB (after $SYSID_JOB)"
    done
done

# Remove leading colon
TRAIN_JOBS="${TRAIN_JOBS#:}"

# --- Step 3: Submit eval job, depends on ALL training jobs ---
EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:${TRAIN_JOBS} \
    pgdr/tacc/3_eval.sh)
echo "Submitted eval job: $EVAL_JOB (after all training)"

echo ""
echo "============================================================"
echo "Pipeline submitted!"
echo "  Sysid:    $SYSID_JOB"
echo "  Training: 18 jobs (${CONDITIONS// /, } x seeds ${SEEDS// /, })"
echo "  Eval:     $EVAL_JOB"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "============================================================"
