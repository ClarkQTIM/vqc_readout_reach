#!/bin/bash
# Reproduce the 4-qubit pilot exactly (4,860 jobs). Results do not depend on the number of
# shards: every job is seeded from its own (circuit, readout, task, seed) key.
#
#   mkdir -p logs
#   N_SHARDS=20 sbatch --array=0-19 slurm/pilot.sbatch
#   # when all tasks finish:
#   bash scripts/reproduce_pilot.sh --analyze-only

#SBATCH --job-name=pilot_reach
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME

set -euo pipefail
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate CHANGE_ME
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python scripts/selection_rule.py \
  --layers 1 2 4 --seeds 0 1 2 \
  --readouts z0 linear parity \
  --tasks sign magnitude iris_versicolor_virginica \
  --out results/raw/pilot \
  --shard "${SLURM_ARRAY_TASK_ID}" --n-shards "${N_SHARDS:-20}"
