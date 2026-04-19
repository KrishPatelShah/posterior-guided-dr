#!/bin/bash
# Run this once on TACC to set up the pgdr_env virtual environment.
# Usage: bash pgdr/tacc/setup_env.sh

set -e

module load gcc/11.2.0
module load cuda/12.0
module load python/3.12.11

python -m venv $WORK/pgdr_env
source $WORK/pgdr_env/bin/activate

# Install all pip-installable dependencies
pip install -r requirements_tacc.txt

# mujoco_playground requires submodules (assets not bundled in pip install)
if [ ! -d "$WORK/mujoco_playground/.git" ]; then
    git clone --recurse-submodules https://github.com/google-deepmind/mujoco_playground.git $WORK/mujoco_playground
fi
pip install -e $WORK/mujoco_playground

# Install this package
pip install -e $WORK/posterior-guided-dr

echo "Environment setup complete."
