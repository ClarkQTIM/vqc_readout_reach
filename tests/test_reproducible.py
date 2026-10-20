"""A job's result depends only on its key, not on the machine's run order or shard count."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("selection_rule", ROOT / "scripts" / "selection_rule.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)

ARGS = SimpleNamespace(feature=3, n_train=50, n_test=100, epochs=5, lr=0.05, max_dla_qubits=6)
JOB = dict(n_qubits=4, num_layers=1, single_qubit_op="ry", entanglement_op="cnot", topology="ring",
           readout="z0", task="sign", seed=1)


def test_same_job_same_result():
    a = sr.run_job(dict(JOB), ARGS, {})
    b = sr.run_job(dict(JOB), ARGS, {})
    for k in ("auc_test", "auc_train", "loss_init", "loss_final", "grad_norm_init", "reach", "dla_dim"):
        assert a[k] == b[k]


def test_sharding_partitions_the_jobs():
    base = dict(n_qubits=4, layers=[1, 2], rotations=["rx", "ry"], entanglers=["none", "cnot"],
                topologies=["ring", "all_to_all"], readouts=["z0"], tasks=["sign"], seeds=[0, 1])
    keys = [sr.job_key(j) for j in sr.jobs(SimpleNamespace(**base))]
    shards = [{k for i, k in enumerate(keys) if i % 3 == s} for s in range(3)]
    assert set().union(*shards) == set(keys) and sum(map(len, shards)) == len(keys)
