# vqc-readout

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
| `scripts/grid_table.py` | 4-qubit grid: structure joined with the original sweep's AUCs |
| `reference/` | Original code and result tables, unchanged, with a list of known issues |

## Quick start

```bash
pip install -e ".[test]"
pytest                       # ~1 min; the simulator test also needs `pip install pennylane`
python scripts/grid_table.py # writes results/grid_4q.csv
```
