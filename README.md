# vqc_readout_reach

What can the readout qubit see? Readout reachability, Lie algebras and performance of
variational quantum classifiers.

## Layout

| Path | What it holds |
| --- | --- |
| `vqcx/pauli.py` | Pauli strings as bit masks: products, commutation, CNOT/CZ conjugation, statevector action |
| `vqcx/circuits.py` | Layered circuit specs (RX/RY/RZ/U3 × none/CNOT/CZ/CRX/CRY/Ising × ring/all-to-all/line) |
| `vqcx/simulate.py` | Batched statevector simulator (matches PennyLane on the whole grid) |
| `vqcx/reach.py` | Which input features reach the measured observable, and whether only even functions of them do |
| `vqcx/lie.py` | Dynamical Lie algebra dimension, including fixed CNOT/CZ layers and tied CRX/CRY generators |
| `vqcx/metrics.py` | Sim et al. KL expressibility (exact Haar bins, U[0, 2π] init), Meyer–Wallach, entanglement entropy in bits |
| `vqcx/tasks.py` | Synthetic sign / magnitude tasks on one feature; Iris class pairs |
| `vqcx/train.py` | Classifier with z0, parity or linear readout; BCE + Adam; exact adjoint gradients |
| `scripts/selection_rule.py` | The training experiment, sharded for SLURM arrays and resumable |
| `scripts/circuit_metrics.py` | KL, entanglement, Lie dimension and reach for every circuit in a grid |
| `scripts/analyze_selection_rule.py` | Summary tables for a run |
| `scripts/grid_table.py` | 4-qubit grid: structure joined with the original sweep's AUCs |
| `slurm/selection_rule.sbatch` | Job-array template for the server |
| `reference/` | Original code and result tables, unchanged, with a list of known issues |

## Quick start

```bash
pip install -e ".[test]"
pytest                                   # ~2 min; simulator tests also need `pip install pennylane`
python scripts/circuit_metrics.py --out results/circuit_metrics_4q.csv
python scripts/selection_rule.py --layers 1 2 --seeds 0 --out results/raw/smoke
python scripts/analyze_selection_rule.py results/raw/smoke --metrics results/circuit_metrics_4q.csv
```

## Reproducing a run

Every job is seeded from its own key (circuit, readout, task, seed), so results do not depend on
run order or on how many shards a run is split into. Each shard also writes a manifest with the
git commit and library versions.

```bash
bash scripts/reproduce_pilot.sh                     # tests + full pilot locally, then compare
# or on SLURM:
N_SHARDS=20 sbatch --array=0-19 slurm/pilot.sbatch
bash scripts/reproduce_pilot.sh --analyze-only
```

`results/reference/` holds the reference outputs to check against. The server's run is the
one that counts; `scripts/compare_runs.py <truth> <other>` reports every job that disagrees.

## The prediction being tested

Each input x_q enters once, through RY(x_q), so the readout's expectation value is multilinear in
(cos x_q, sin x_q). `vqcx/reach.py` finds, from the gate list alone, which of those two parts
reach the measured observable for each feature. If a task's label is unchanged by x_q -> -x_q
(the magnitude task) and only the sin part reaches the readout, or the label flips with x_q (the
sign task) and only the cos part does, the model's outputs for the two classes are mirror
images, so the expected test AUC is exactly 0.5, however expressive or entangling the circuit is.
