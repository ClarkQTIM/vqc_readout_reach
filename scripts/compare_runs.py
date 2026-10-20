"""Compare two runs of the same experiment job by job (e.g. the server's run vs a reference run).

Each argument is either a run directory of shard_*.jsonl files or a *_runs.csv file.

Usage:
  python scripts/compare_runs.py results/raw/pilot results/reference/claude_pilot_runs.csv
Exit code 0 if every shared job agrees within --tol on every compared column, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

JOB = ["n_qubits", "num_layers", "single_qubit_op", "entanglement_op", "topology", "readout", "task", "seed"]
COMPARE = ["auc_test", "auc_train", "loss_init", "loss_final", "grad_norm_init"]
EXACT = ["n_params", "dla_dim", "reach", "predicted_solvable"]


def load(p: Path) -> pd.DataFrame:
    if p.is_dir():
        rows = [json.loads(l) for f in sorted(p.glob("shard_*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
        return pd.DataFrame(rows)
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth", type=Path, help="the run that counts (your server's)")
    ap.add_argument("other", type=Path, help="the run to check against it")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    a, b = load(args.truth), load(args.other)
    for df in (a, b):
        df["dla_dim"] = df["dla_dim"].astype("float")
        if "predicted_solvable" in df:
            df["predicted_solvable"] = df["predicted_solvable"].map(
                {True: "True", False: "False", "True": "True", "False": "False"}).fillna("")
    m = a.merge(b, on=JOB, suffixes=("", "_other"), how="outer", indicator=True)
    only_a, only_b = (m._merge == "left_only").sum(), (m._merge == "right_only").sum()
    both = m[m._merge == "both"]
    print(f"jobs: {len(both)} in both, {only_a} only in truth, {only_b} only in other")

    ok = True
    for c in EXACT:
        if c in both and f"{c}_other" in both:
            bad = both[both[c].astype(str) != both[f"{c}_other"].astype(str)]
            print(f"  {c:16s} mismatches: {len(bad)}")
            ok &= len(bad) == 0
    for c in COMPARE:
        if c in both and f"{c}_other" in both:
            d = np.abs(both[c].astype(float) - both[f"{c}_other"].astype(float))
            print(f"  {c:16s} max |diff| = {d.max():.2e}   jobs over tol: {(d > args.tol).sum()}")
            ok &= bool((d <= args.tol).all())
    if not ok:
        d = np.abs(both.auc_test - both.auc_test_other)
        print("\nLargest auc_test differences:")
        print(both.assign(diff=d).nlargest(10, "diff")[JOB + ["auc_test", "auc_test_other", "diff"]].to_string(index=False))
    print("\nAGREE" if ok and only_a == 0 and only_b == 0 else "\nDIFFERENT (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
