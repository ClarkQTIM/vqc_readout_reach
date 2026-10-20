"""Task-independent metrics for every circuit in a grid: KL expressibility, entanglement,
Lie algebra dimension, and readout reach for each readout.

Usage:  python scripts/circuit_metrics.py --n-qubits 4 --layers 1 2 4 --out results/circuit_metrics_4q.csv
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from vqcx.circuits import ENTANGLERS, ROTATIONS, CircuitSpec, build_circuit
from vqcx.lie import lie_algebra_dimension
from vqcx.metrics import entanglement, expressibility_kl, haar_references, sample_states

from selection_rule import code, reach_parts  # same reach summary as the training runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--topologies", nargs="+", default=["ring", "all_to_all"])
    ap.add_argument("--n-pairs", type=int, default=5000)
    ap.add_argument("--n-ent", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    n = args.n_qubits
    rows, dla = [], {}
    for L, rot, ent, topo in itertools.product(args.layers, ROTATIONS, ENTANGLERS, args.topologies):
        if ent == "none" and topo != args.topologies[0]:
            continue
        circ = build_circuit(CircuitSpec(n, L, rot, ent, topo))
        key = (rot, ent, topo)
        if key not in dla:
            dla[key] = lie_algebra_dimension(build_circuit(CircuitSpec(n, 1, rot, ent, topo))) if n <= 6 else None
        rng = np.random.default_rng([args.seed, L, ROTATIONS.index(rot), ENTANGLERS.index(ent), args.topologies.index(topo)])
        states = sample_states(circ, 2 * args.n_pairs, rng)
        e = entanglement(circ, n_samples=args.n_ent, states=states)
        rows.append(dict(
            n_qubits=n, num_layers=L, single_qubit_op=rot, entanglement_op=ent,
            topology="none" if ent == "none" else topo, n_params=circ.n_params, dla_dim=dla[key],
            kl=expressibility_kl(circ, n_pairs=args.n_pairs, states=states),
            meyer_wallach=e["meyer_wallach"], vn_entropy_bits=e["vn_entropy_bits"],
            **{f"reach_{ro}": code(reach_parts(circ, ro)) for ro in ("z0", "linear", "parity")},
        ))
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    ref = haar_references(n)
    print(f"wrote {args.out} ({len(df)} circuits). Haar: MW {ref['meyer_wallach']:.3f}, "
          f"vN {ref['vn_entropy_bits']:.3f} bits")


if __name__ == "__main__":
    main()
