"""Headline figure: test AUC vs KL expressibility, split by the reach rule's prediction.

Usage:  python scripts/plot_selection_rule.py results/pilot_runs.csv results/figures/selection_rule_pilot.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOLVABLE = "#2a78d6"     # categorical slot 1 (validated pair with slot 2)
UNSOLVABLE = "#eb6834"   # categorical slot 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main(csv: Path, out: Path):
    df = pd.read_csv(csv)
    df = df[df.task.isin(["sign", "magnitude"])].copy()
    df["predicted_solvable"] = df.predicted_solvable.astype(bool)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    plt.rcParams.update({"font.size": 10})
    titles = {"sign": "Label = sign of x₃  (needs sin x₃)", "magnitude": "Label = size of |x₃|  (needs cos x₃)"}
    for ax, task in zip(axes, ["sign", "magnitude"]):
        s = df[df.task == task]
        ax.axvspan(1e-3, 0.05, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.text(1.2e-3, 0.06, "near-Haar KL", color=INK2, fontsize=8.5, va="bottom")
        ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        for flag, color in ((True, SOLVABLE), (False, UNSOLVABLE)):
            t = s[s.predicted_solvable == flag]
            jitter = 10 ** rng.normal(0, 0.03, len(t))
            ax.scatter(np.clip(t.kl, 1e-3, None) * jitter, t.auc_test, s=9, color=color, alpha=0.35,
                       lw=0, zorder=2 if flag else 3)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 150)
        ax.set_ylim(0, 1.03)
        ax.set_title(titles[task], fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("KL divergence from Haar (lower = more expressive)", color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK2)
        n_bad = (~s.predicted_solvable).sum()
        ax.text(150, 0.43, f"predicted unsolvable ({n_bad} runs)", color=UNSOLVABLE, fontsize=9, ha="right")
        ax.text(150, 1.0, f"predicted solvable ({len(s) - n_bad} runs)", color=SOLVABLE, fontsize=9, ha="right", va="top")
    axes[0].set_ylabel("Test AUC", color=INK2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l, markersize=6)
               for c, l in ((SOLVABLE, "predicted solvable"), (UNSOLVABLE, "predicted unsolvable"))]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
