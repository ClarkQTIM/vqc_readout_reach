"""Summarize a selection-rule run: does readout reach predict what each circuit can learn?

Usage:  python scripts/analyze_selection_rule.py results/raw/pilot --metrics results/circuit_metrics_4q.csv
Writes  results/<run>_runs.csv (all runs, one row each) and prints the summary tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

KEYS = ["n_qubits", "num_layers", "single_qubit_op", "entanglement_op", "topology"]


def load(run_dir: Path) -> pd.DataFrame:
    rows = [json.loads(l) for f in sorted(run_dir.glob("shard_*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--metrics", type=Path, default=None)
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    df = load(args.run_dir)
    if args.metrics is not None:
        m = pd.read_csv(args.metrics).drop(columns=["n_params", "dla_dim"], errors="ignore")
        df = df.merge(m[KEYS + ["kl", "meyer_wallach", "vn_entropy_bits"]], on=KEYS, how="left")
    out = Path("results") / f"{args.run_dir.name}_runs.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} runs -> {out}\n")

    syn = df[df.task.isin(["sign", "magnitude"])].copy()
    if len(syn):
        print("=== 1. Prediction vs outcome on the synthetic tasks (test AUC; n_test per run in the config) ===")
        g = syn.groupby(["task", "readout", "predicted_solvable"]).auc_test
        t = pd.DataFrame({"runs": g.size(), "mean": g.mean(), "p5": g.quantile(0.05), "p95": g.quantile(0.95),
                          "frac_above_0.6": g.apply(lambda s: (s > 0.6).mean())})
        print(t.round(3).to_string(), "\n")

        bad = syn[~syn.predicted_solvable]
        print(f"Predicted unsolvable: {len(bad)} runs, mean AUC {bad.auc_test.mean():.3f}, "
              f"largest |AUC - 0.5| = {np.abs(bad.auc_test - 0.5).max():.3f}")
        if "kl" in bad:
            hi = bad[bad.kl < 0.05]
            print(f"  of which near-Haar KL (< 0.05): {len(hi)} runs, mean AUC {hi.auc_test.mean():.3f}")
            show = (hi.groupby(KEYS[1:] + ["readout", "task"]).agg(kl=("kl", "first"),
                    vn=("vn_entropy_bits", "first"), dla=("dla_dim", "first"), auc=("auc_test", "mean"))
                    .sort_values("kl").head(8))
            print(show.round(3).to_string(), "\n")

        ok = syn[syn.predicted_solvable]
        print("=== 2. Among predicted-solvable runs, what tracks AUC? (Spearman rho) ===")
        cols = [c for c in ["kl", "meyer_wallach", "vn_entropy_bits", "dla_dim", "n_params", "num_layers"] if c in ok]
        rows = []
        for (task, ro), s in ok.groupby(["task", "readout"]):
            rows.append({"task": task, "readout": ro, "runs": len(s),
                         **{c: spearmanr(s[c], s.auc_test)[0] for c in cols}})
        print(pd.DataFrame(rows).round(2).to_string(index=False), "\n")

    iris = df[df.task.str.startswith("iris")]
    if len(iris):
        print("=== 3. Iris (versicolor vs virginica): AUC by number of features with an odd part ===")
        iris = iris.assign(n_odd=iris.reach.apply(lambda r: sum(c in "OB" for c in r)))
        print(iris.groupby(["readout", "n_odd"]).auc_test.agg(["size", "mean", "min", "max"]).round(3).to_string(), "\n")

    print("=== 4. Matched pairs for C: same circuit and parameter count, different Lie algebra ===")
    pair = df[df.entanglement_op.isin(["ising_xx", "ising_yy", "ising_zz"]) & df.single_qubit_op.isin(["rx", "ry", "rz"])]
    pair = pair[pair.task.isin(["sign", "magnitude"]) & pair.predicted_solvable.fillna(False)]
    agg = (pair.groupby(["single_qubit_op", "entanglement_op", "topology", "readout"])
           .agg(dla=("dla_dim", "first"), n_params=("n_params", "first"), runs=("auc_test", "size"),
                auc=("auc_test", "mean"), loss_final=("loss_final", "mean")))
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
