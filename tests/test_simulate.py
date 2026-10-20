"""The simulator must reproduce PennyLane's gate conventions, as used in the original sweep."""

import itertools
import zlib

import numpy as np
import pytest

from vqcx import pauli as pl
from vqcx.circuits import ENTANGLERS, ROTATIONS, CircuitSpec, build_circuit, pairs
from vqcx.simulate import expectation, run

qml = pytest.importorskip("pennylane")


def pennylane_state(spec: CircuitSpec, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    n = spec.n_qubits
    dev = qml.device("default.qubit", wires=n)
    edges = [] if spec.entangler == "none" else pairs(n, spec.topology)

    @qml.qnode(dev)
    def circuit():
        for q in range(n):
            qml.RY(x[q], wires=q)
        k = 0
        for _ in range(spec.n_layers):
            for q in range(n):
                if spec.rotation == "u3":
                    qml.U3(theta[k], theta[k + 1], theta[k + 2], wires=q)
                    k += 3
                else:
                    {"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ}[spec.rotation](theta[k], wires=q)
                    k += 1
            for c, t in edges:
                e = spec.entangler
                if e == "cnot":
                    qml.CNOT(wires=[c, t])
                elif e == "cz":
                    qml.CZ(wires=[c, t])
                else:
                    gate = {"crx": qml.CRX, "cry": qml.CRY, "ising_xx": qml.IsingXX,
                            "ising_yy": qml.IsingYY, "ising_zz": qml.IsingZZ}[e]
                    gate(theta[k], wires=[c, t])
                    k += 1
        return qml.state()

    return np.asarray(circuit())


GRID = list(itertools.product(ROTATIONS, ENTANGLERS, ["ring", "all_to_all", "line"]))


@pytest.mark.parametrize("rotation,entangler,topology", GRID)
def test_matches_pennylane(rotation, entangler, topology):
    rng = np.random.default_rng(zlib.crc32(f"{rotation}-{entangler}-{topology}".encode()))
    spec = CircuitSpec(3, 2, rotation, entangler, topology)
    circ = build_circuit(spec)
    x = rng.uniform(-2, 2, size=3)
    theta = rng.uniform(0, 2 * np.pi, size=circ.n_params)
    ours = run(circ, x[None, :], theta)[0]
    ref = pennylane_state(spec, x, theta)
    assert abs(abs(np.vdot(ref, ours)) - 1) < 1e-10  # equal up to a global phase (U3)
    for label in ["ZII", "ZZZ", "XIY"]:
        p = pl.from_label(label)
        assert np.isclose(expectation(circ, x[None, :], theta, p)[0], np.real(np.vdot(ref, pl.apply(ref, p, 3))))


def test_parameter_counts():
    assert build_circuit(CircuitSpec(4, 2, "rx", "none")).n_params == 8
    assert build_circuit(CircuitSpec(4, 2, "u3", "cnot", "ring")).n_params == 24
    assert build_circuit(CircuitSpec(4, 1, "rx", "ising_zz", "all_to_all")).n_params == 4 + 6
    assert build_circuit(CircuitSpec(4, 1, "ry", "crx", "ring")).n_params == 4 + 4
