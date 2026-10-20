"""Structural properties of the 4-qubit grid, joined with the original sweep's results.

For every circuit: parameter count, Lie algebra dimension, and which input features reach the
qubit-0 readout (none / even / odd). Joined with the original Iris and RetinaMNIST AUCs and the
original KL / entanglement values, so we can ask what predicts performance.

Usage:  python scripts/grid_table.py  ->  results/grid_4q.csv + a printed summary
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from vqcx.circuits import ENTANGLERS, ROTATIONS, CircuitSpec, build_circuit, readout_pauli
from vqcx.lie import lie_algebra_dimension
from vqcx.reach import readout_reach

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "original_results"
OUT = ROOT / "results" / "grid_4q.csv"
LAYERS = (1, 2, 4)
TOPOS = ("ring", "all_to_all")


def iris_feature_aucs() -> np.ndarray:
    """Best single-feature AUC for setosa vs versicolor, after the original standardization."""
    d = load_iris()
    keep = d.target < 2
    X = StandardScaler().fit_transform(d.data[keep])
    y = d.target[keep]
    return np.array([max(a, 1 - a) for a in (roc_auc_score(y, X[:, i]) for i in range(4))])


def structural_table() -> pd.DataFrame:
    rows, dla_cache = [], {}
    for L, rot, ent, topo in itertools.product(LAYERS, ROTATIONS, ENTANGLERS, TOPOS):
        if ent == "none" and topo != "ring":
            continue
        spec = CircuitSpec(4, L, rot, ent, topo)
        circ = build_circuit(spec)
        key = (rot, ent, topo)
        if key not in dla_cache:
            dla_cache[key] = lie_algebra_dimension(build_circuit(CircuitSpec(4, 1, rot, ent, topo)))
        r0 = readout_reach(circ, readout_pauli(4, "Z0"))
        rp = readout_reach(circ, readout_pauli(4, "parity"))
        rows.append(dict(
            num_layers=L, single_qubit_op=rot, entanglement_op=ent,
            topology="none" if ent == "none" else topo,
            n_params=circ.n_params, dla_dim=dla_cache[key],
            reach_z0="".join(s[0].upper() for s in r0.feature_status),   # e.g. 'NOOE'
            n_odd_z0=len(r0.odd_features), n_reached_z0=len(r0.reached_features),
            reach_parity="".join(s[0].upper() for s in rp.feature_status),
            n_odd_parity=len(rp.odd_features),
        ))
    return pd.DataFrame(rows)


def main():
    df = structural_table()
    keys = ["num_layers", "single_qubit_op", "entanglement_op", "topology"]

    runs = pd.read_csv(REF / "performance_runs_reparsed.csv").rename(
        columns={"L": "num_layers", "sq": "single_qubit_op", "ent": "entanglement_op", "topo": "topology"})
    perf = runs.groupby(["ds"] + keys).auc.agg(["mean", "std"]).reset_index()
    for ds in ("iris", "retina"):
        p = perf[perf.ds == ds].drop(columns="ds").rename(columns={"mean": f"auc_{ds}", "std": f"auc_{ds}_std"})
        df = df.merge(p, on=keys, how="left")

    for kind, col in (("expressibility", "kl_divergence"), ("entanglement", "avg_entropy")):
        old = pd.read_csv(REF / "individual_qubits/4_qubit" / f"{kind}_results.csv")
        old["entanglement_op"] = old["entanglement_op"].fillna("none")
        old = old.groupby(keys)[col].mean().reset_index().rename(columns={col: f"old_{col}"})
        df = df.merge(old, on=keys, how="left")

    fa = iris_feature_aucs()
    df["iris_pred_ceiling"] = [
        max([fa[i] for i, s in enumerate(r) if s == "O"], default=0.5) for r in df.reach_z0
    ]
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Iris single-feature AUCs (setosa vs versicolor, standardized): {np.round(fa, 3)}")
    print(f"\nWrote {OUT.relative_to(ROOT)}  ({len(df)} circuits)\n")
    print("Spearman correlation with Iris AUC (mean of 3 runs):")
    for col in ("iris_pred_ceiling", "n_odd_z0", "n_reached_z0", "old_kl_divergence",
                "old_avg_entropy", "dla_dim", "n_params"):
        rho, pval = spearmanr(df[col], df.auc_iris)
        print(f"  {col:20s} rho = {rho:+.2f}   (p = {pval:.1e})")
    print("\nSpearman correlation with RetinaMNIST AUC:")
    for col in ("n_odd_z0", "old_kl_divergence", "old_avg_entropy", "dla_dim", "n_params"):
        rho, pval = spearmanr(df[col], df.auc_retina)
        print(f"  {col:20s} rho = {rho:+.2f}   (p = {pval:.1e})")
    print("\nIris AUC by readout-reach pattern (features 0-3: N=none, E=even only, O=odd):")
    g = df.groupby("reach_z0").agg(n=("auc_iris", "size"), auc_mean=("auc_iris", "mean"),
                                   auc_min=("auc_iris", "min"), auc_max=("auc_iris", "max"))
    print(g.sort_values("auc_mean").round(3).to_string())


if __name__ == "__main__":
    main()
