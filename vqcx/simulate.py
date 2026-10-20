"""Batched statevector simulation of a Circuit."""

from __future__ import annotations

import numpy as np

from . import pauli as pl
from .circuits import Circuit, Clifford, Encode, Rot


def _bit(n: int, q: int) -> int:
    return 1 << (n - 1 - q)


def _apply_clifford(state: np.ndarray, op: Clifford, n: int) -> np.ndarray:
    idx = np.arange(2**n)
    c, t = _bit(n, op.control), _bit(n, op.target)
    if op.gate == "cnot":
        src = np.where(idx & c, idx ^ t, idx)
        return state[..., src]
    if op.gate == "cz":
        sign = np.where(((idx & c) != 0) & ((idx & t) != 0), -1.0, 1.0)
        return state * sign
    raise ValueError(op.gate)


def _rotate(state: np.ndarray, p: pl.Pauli, angle: np.ndarray, n: int) -> np.ndarray:
    """exp(-i angle/2 P)|psi>; ``angle`` broadcasts over the batch axis."""
    a = np.asarray(angle, dtype=float)[..., None]
    return np.cos(a / 2) * state - 1j * np.sin(a / 2) * pl.apply(state, p, n)


def run(circuit: Circuit, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Final states for inputs ``x`` of shape (B, n_features) and one parameter vector ``theta``.

    Returns an array of shape (B, 2**n).
    """
    n = circuit.n_qubits
    x = np.atleast_2d(np.asarray(x, dtype=float))
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (circuit.n_params,):
        raise ValueError(f"theta must have shape ({circuit.n_params},), got {theta.shape}")
    state = np.zeros((x.shape[0], 2**n), dtype=complex)
    state[:, 0] = 1.0
    for op in circuit.ops:
        if isinstance(op, Encode):
            state = _rotate(state, pl.single(n, op.qubit, "Y"), x[:, op.feature], n)
        elif isinstance(op, Rot):
            state = _rotate(state, op.pauli, op.coeff * theta[op.param], n)
        else:
            state = _apply_clifford(state, op, n)
    return state


def expectation(circuit: Circuit, x: np.ndarray, theta: np.ndarray, observable: pl.Pauli) -> np.ndarray:
    """<O>(x; theta) for each row of x."""
    return pl.expval(run(circuit, x, theta), observable, circuit.n_qubits)
