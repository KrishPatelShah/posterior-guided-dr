#!/bin/bash
# Submit all PGDR training jobs

mkdir -p pgdr/tacc/jobs/logs

sbatch pgdr/tacc/jobs/train_C1_uniform_dr_seed0.sh
sbatch pgdr/tacc/jobs/train_C1_uniform_dr_seed1.sh
sbatch pgdr/tacc/jobs/train_C1_uniform_dr_seed2.sh
sbatch pgdr/tacc/jobs/train_C2_pure_sysid_seed0.sh
sbatch pgdr/tacc/jobs/train_C2_pure_sysid_seed1.sh
sbatch pgdr/tacc/jobs/train_C2_pure_sysid_seed2.sh
sbatch pgdr/tacc/jobs/train_C3_isotropic_seed0.sh
sbatch pgdr/tacc/jobs/train_C3_isotropic_seed1.sh
sbatch pgdr/tacc/jobs/train_C3_isotropic_seed2.sh
sbatch pgdr/tacc/jobs/train_C4_pgdr_0.5_seed0.sh
sbatch pgdr/tacc/jobs/train_C4_pgdr_0.5_seed1.sh
sbatch pgdr/tacc/jobs/train_C4_pgdr_0.5_seed2.sh
sbatch pgdr/tacc/jobs/train_C4_pgdr_1.0_seed0.sh
sbatch pgdr/tacc/jobs/train_C4_pgdr_1.0_seed1.sh
sbatch pgdr/tacc/jobs/train_C4_pgdr_1.0_seed2.sh
sbatch pgdr/tacc/jobs/train_C4_pgdr_2.0_seed0.sh
sbatch pgdr/tacc/jobs/train_C4_pgdr_2.0_seed1.sh
sbatch pgdr/tacc/jobs/train_C4_pgdr_2.0_seed2.sh

echo 'Submitted 18 jobs'
