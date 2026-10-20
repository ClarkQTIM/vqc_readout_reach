#!/bin/bash
# Reproduce the 4-qubit pilot from scratch and compare it with the reference run.
#
#   bash scripts/reproduce_pilot.sh            # everything, locally (about 1 CPU-hour)
#   bash scripts/reproduce_pilot.sh --analyze-only   # after running slurm/pilot.sbatch
#
# Your run is the one that counts: compare_runs.py treats it as the truth and checks the
# reference run (results/reference/) against it.

set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

if [[ "${1:-}" != "--analyze-only" ]]; then
  echo "== tests"
  python -m pytest -q
  echo "== training runs (2 local shards)"
  for s in 0 1; do
    python scripts/selection_rule.py --layers 1 2 4 --seeds 0 1 2 \
      --readouts z0 linear parity --tasks sign magnitude iris_versicolor_virginica \
      --out results/raw/pilot --shard "$s" --n-shards 2 &
  done
  wait
fi

echo "== circuit metrics (KL, entanglement, Lie dimension, reach)"
python scripts/circuit_metrics.py --out results/circuit_metrics_4q.csv

echo "== compare circuit metrics with the reference"
python - <<'EOF'
import pandas as pd, numpy as np
a = pd.read_csv("results/circuit_metrics_4q.csv"); b = pd.read_csv("results/reference/circuit_metrics_4q.csv")
keys = ["n_qubits", "num_layers", "single_qubit_op", "entanglement_op", "topology"]
m = a.merge(b, on=keys, suffixes=("", "_ref"))
print(f"circuits: {len(m)} matched of {len(a)} / {len(b)}")
for c in ["n_params", "dla_dim", "reach_z0", "reach_linear", "reach_parity"]:
    print(f"  {c:14s} mismatches: {(m[c].astype(str) != m[c + '_ref'].astype(str)).sum()}")
for c in ["kl", "meyer_wallach", "vn_entropy_bits"]:
    print(f"  {c:14s} max |diff| = {np.abs(m[c] - m[c + '_ref']).max():.2e}")
EOF

echo "== analysis"
python scripts/analyze_selection_rule.py results/raw/pilot --metrics results/circuit_metrics_4q.csv
python scripts/plot_selection_rule.py results/pilot_runs.csv results/figures/selection_rule_pilot.png

echo "== compare training runs with the reference (yours = truth)"
python scripts/compare_runs.py results/raw/pilot results/reference/claude_pilot_runs.csv
