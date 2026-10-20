"""The structural readout-reach rule must agree with direct simulation on the whole grid."""

import itertools

import numpy as np
import pytest

from vqcx.circuits import ENTANGLERS, ROTATIONS, CircuitSpec, build_circuit, readout_pauli
from vqcx.reach import readout_reach
from vqcx.simulate import expectation


def numeric_status(circ, obs, rng, n_theta=6, n_x=6, h=1e-3, tol=1e-7):
    n = circ.n_qubits
    status = ["none"] * n
    for _ in range(n_theta):
        theta = rng.uniform(0, 2 * np.pi, circ.n_params)
        x = rng.uniform(-2, 2, size=(n_x, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = h
            d = expectation(circ, x + e, theta, obs) - expectation(circ, x - e, theta, obs)
            flipped = x.copy()
            flipped[:, i] *= -1
            odd = expectation(circ, x, theta, obs) - expectation(circ, flipped, theta, obs)
            if np.max(np.abs(odd)) > tol:
                status[i] = "odd"
            elif np.max(np.abs(d)) / (2 * h) > tol and status[i] == "none":
                status[i] = "even"
    return status


CASES = [
    (r, e, t, L, ro)
    for r, e, t in itertools.product(ROTATIONS, ENTANGLERS, ["ring", "all_to_all", "line"])
    if not (e == "none" and t != "ring")
    for L in (1, 2, 3)
    for ro in ("Z0", "parity")
]


@pytest.mark.parametrize("rotation,entangler,topology,layers,readout", CASES)
def test_rule_matches_simulation(rotation, entangler, topology, layers, readout):
    spec = CircuitSpec(4, layers, rotation, entangler, topology)
    circ = build_circuit(spec)
    obs = readout_pauli(4, readout)
    rule = readout_reach(circ, obs).feature_status
    rng = np.random.default_rng(len(CASES))
    assert numeric_status(circ, obs, rng) == rule


def test_ring_cnot_examples():
    """One layer of RX vs RY with a CNOT ring, read out on qubit 0 (the old Iris puzzle)."""
    obs = readout_pauli(4, "Z0")
    rx = readout_reach(build_circuit(CircuitSpec(4, 1, "rx", "cnot", "ring")), obs)
    ry = readout_reach(build_circuit(CircuitSpec(4, 1, "ry", "cnot", "ring")), obs)
    assert rx.feature_status == ["none", "even", "even", "even"]
    assert ry.feature_status == ["none", "odd", "odd", "odd"]
