#!/bin/bash
# Selection-rule experiment as a SLURM job array. Each array task runs one shard; shards resume.
#
#   mkdir -p logs
#   sbatch --array=0-49 slurm/selection_rule.sbatch                       # 4 qubits, defaults
#   NQ=6 RUN=full_6q sbatch --array=0-199 slurm/selection_rule.sbatch      # 6 qubits
#
# Set N_SHARDS to the array size (default 50). Edit the partition/account/env lines for your cluster.

#SBATCH --job-name=readout_reach
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME

set -euo pipefail
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate CHANGE_ME
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NQ="${NQ:-4}"
RUN="${RUN:-full_${NQ}q}"
N_SHARDS="${N_SHARDS:-50}"

python scripts/selection_rule.py \
  --n-qubits "${NQ}" \
  --layers 1 2 4 8 \
  --readouts z0 linear parity \
  --tasks sign magnitude iris_versicolor_virginica \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --epochs 150 \
  --out "results/raw/${RUN}" \
  --shard "${SLURM_ARRAY_TASK_ID}" --n-shards "${N_SHARDS}"
