"""The selection-rule experiment: does readout reach predict which tasks a circuit can learn?

Every job = (circuit, readout, task, seed). Jobs are enumerated in a fixed order and split into
shards (job i goes to shard i % n_shards), so a SLURM array can run shard $SLURM_ARRAY_TASK_ID.
Each shard appends one JSON line per finished job and skips jobs already written, so it resumes.

Data are seeded by ``seed`` alone (every circuit sees the same data for a given seed); parameter
initialization is seeded by (seed, circuit, readout, task).

Example (pilot):
  python scripts/selection_rule.py --layers 1 2 --seeds 0 1 2 --readouts z0 linear \
      --tasks sign magnitude --out results/raw/pilot --shard 0 --n-shards 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
import zlib
from pathlib import Path

import numpy as np

from vqcx.circuits import ENTANGLERS, ROTATIONS, CircuitSpec, build_circuit
from vqcx.lie import lie_algebra_dimension
from vqcx.reach import readout_reach
from vqcx.tasks import make_task
from vqcx.train import Classifier, auc, readout_observables


def jobs(args):
    for L, rot, ent, topo, ro, task, seed in itertools.product(
            args.layers, args.rotations, args.entanglers, args.topologies,
            args.readouts, args.tasks, args.seeds):
        if ent == "none" and topo != args.topologies[0]:
            continue  # topology is irrelevant without entanglers
        yield dict(n_qubits=args.n_qubits, num_layers=L, single_qubit_op=rot, entanglement_op=ent,
                   topology="none" if ent == "none" else topo, readout=ro, task=task, seed=seed)


def job_key(j) -> str:
    return "|".join(str(j[k]) for k in ("n_qubits", "num_layers", "single_qubit_op", "entanglement_op",
                                        "topology", "readout", "task", "seed"))


def reach_parts(circ, readout):
    """Per feature, the union of {even, odd} parts over the readout's observables."""
    n = circ.n_qubits
    parts = [set() for _ in range(n)]
    for obs in readout_observables(n, readout):
        r = readout_reach(circ, obs)
        for i, s in enumerate(r.feature_parts):
            parts[i] |= s
    return parts


def code(parts) -> str:
    """'N' none, 'E' even only, 'O' odd only, 'B' both, per feature."""
    return "".join({frozenset(): "N", frozenset({"even"}): "E", frozenset({"odd"}): "O"}
                   .get(frozenset(p), "B") for p in parts)


def run_job(j, args, dla_cache):
    topo = j["topology"] if j["topology"] != "none" else "ring"
    spec = CircuitSpec(j["n_qubits"], j["num_layers"], j["single_qubit_op"], j["entanglement_op"], topo)
    circ = build_circuit(spec)
    key = (j["n_qubits"], j["single_qubit_op"], j["entanglement_op"], topo)
    if key not in dla_cache:
        one_layer = build_circuit(CircuitSpec(j["n_qubits"], 1, *key[1:]))
        dla_cache[key] = lie_algebra_dimension(one_layer) if j["n_qubits"] <= args.max_dla_qubits else None
    parts = reach_parts(circ, j["readout"])

    data = make_task(j["task"], n_features=j["n_qubits"], feature=args.feature,
                     n_train=args.n_train, n_test=args.n_test, rng=np.random.default_rng(j["seed"]))
    init_seed = np.random.SeedSequence([j["seed"], zlib.crc32(job_key(j).encode())])
    rng = np.random.default_rng(init_seed)
    clf = Classifier(circ, j["readout"]).init(rng)
    t0 = time.time()
    clf.fit(data.x_train, data.y_train, epochs=args.epochs, lr=args.lr, rng=rng)
    rec = dict(j)
    rec.update(
        n_params=circ.n_params, dla_dim=dla_cache[key], reach=code(parts),
        feature=data.feature,
        auc_test=auc(data.y_test, clf.predict_proba(data.x_test)),
        auc_train=auc(data.y_train, clf.predict_proba(data.x_train)),
        loss_init=clf.history["loss"][0], loss_final=clf.history["loss"][-1],
        grad_norm_init=clf.history["grad_norm_theta"][0],
        epochs=args.epochs, lr=args.lr, n_train=args.n_train, seconds=round(time.time() - t0, 3),
    )
    if data.feature is not None:
        p = parts[data.feature]
        rec["predicted_solvable"] = ("odd" in p) if j["task"] == "sign" else ("even" in p)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--rotations", nargs="+", default=list(ROTATIONS))
    ap.add_argument("--entanglers", nargs="+", default=list(ENTANGLERS))
    ap.add_argument("--topologies", nargs="+", default=["ring", "all_to_all"])
    ap.add_argument("--readouts", nargs="+", default=["z0", "linear", "parity"])
    ap.add_argument("--tasks", nargs="+", default=["sign", "magnitude", "iris_versicolor_virginica"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--feature", type=int, default=None, help="label feature for synthetic tasks (default: last qubit)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--max-dla-qubits", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    if args.feature is None:
        args.feature = args.n_qubits - 1

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"shard_{args.shard:04d}.jsonl"
    done = set()
    if path.exists():
        done = {job_key(json.loads(line)) for line in path.read_text().splitlines() if line.strip()}
    todo = [j for i, j in enumerate(jobs(args)) if i % args.n_shards == args.shard and job_key(j) not in done]
    print(f"shard {args.shard}/{args.n_shards}: {len(todo)} jobs to run ({len(done)} already done)", flush=True)
    dla_cache: dict = {}
    t0 = time.time()
    with path.open("a") as fh:
        for k, j in enumerate(todo, 1):
            fh.write(json.dumps(run_job(j, args, dla_cache)) + "\n")
            fh.flush()
            if k % 50 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} jobs, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
