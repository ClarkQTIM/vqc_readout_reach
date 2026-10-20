"""Training variational classifiers with exact (adjoint) gradients.

Readouts
--------
* ``z0``     : p(class 1) = (1 - <Z_0>) / 2, i.e. the probability of measuring 1 on qubit 0
               (the original sweep's readout).
* ``parity`` : p = (1 - <Z_0 Z_1 ... Z_n-1>) / 2.
* ``linear`` : p = sigmoid(w . z + b) with z_q = <Z_q> on every qubit and a trainable (w, b).

Gradients use the adjoint method: one forward pass, then one backward pass that undoes each
gate while accumulating d<O>/d(angle) = Im <lambda| P |psi> for every Pauli rotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

from . import pauli as pl
from .circuits import Circuit, Clifford, Encode, Rot, readout_pauli
from .simulate import _apply_clifford, _rotate, run

READOUTS = ("z0", "parity", "linear")
EPS = 1e-9


def readout_observables(n: int, readout: str) -> list[pl.Pauli]:
    if readout == "z0":
        return [readout_pauli(n, "Z0")]
    if readout == "parity":
        return [readout_pauli(n, "parity")]
    if readout == "linear":
        return [pl.single(n, q, "Z") for q in range(n)]
    raise ValueError(readout)


def expectations_and_vjp(circuit: Circuit, x: np.ndarray, theta: np.ndarray,
                         observables: list[pl.Pauli]):
    """Return (f, vjp): f has shape (B, n_obs); vjp(g) = sum_{b,o} g[b,o] * d f[b,o] / d theta."""
    n = circuit.n_qubits
    psi = run(circuit, x, theta)
    f = np.stack([pl.expval(psi, o, n) for o in observables], axis=1)

    def vjp(g: np.ndarray) -> np.ndarray:
        state = psi.copy()
        lam = sum(g[:, [k]] * pl.apply(state, o, n) for k, o in enumerate(observables))
        grad = np.zeros(circuit.n_params)
        for op in reversed(circuit.ops):
            if isinstance(op, Rot):
                # d f / d angle = Im <lam| P |psi>, summed over the batch (g already folded into lam)
                grad[op.param] += op.coeff * np.sum(np.imag(np.conj(lam) * pl.apply(state, op.pauli, n)))
                a = -op.coeff * theta[op.param]
                state = _rotate(state, op.pauli, a, n)
                lam = _rotate(lam, op.pauli, a, n)
            elif isinstance(op, Clifford):
                state = _apply_clifford(state, op, n)
                lam = _apply_clifford(lam, op, n)
            elif isinstance(op, Encode):
                continue  # encodings sit at the start; nothing trainable before them
        return grad

    return f, vjp


@dataclass
class Classifier:
    circuit: Circuit
    readout: str = "z0"
    theta: np.ndarray | None = None
    w: np.ndarray | None = None
    b: float = 0.0
    history: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}")
        self.observables = readout_observables(self.circuit.n_qubits, self.readout)

    def init(self, rng: np.random.Generator, theta_range: float = 2 * np.pi):
        self.theta = rng.uniform(0, theta_range, self.circuit.n_params)
        if self.readout == "linear":
            self.w = rng.normal(0, 1, len(self.observables))
            self.b = 0.0
        return self

    # ------------------------------------------------------------------ forward
    def _prob_from_f(self, f: np.ndarray) -> np.ndarray:
        if self.readout == "linear":
            return 1.0 / (1.0 + np.exp(-(f @ self.w + self.b)))
        return (1.0 - f[:, 0]) / 2.0

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        f, _ = expectations_and_vjp(self.circuit, x, self.theta, self.observables)
        return self._prob_from_f(f)

    # ------------------------------------------------------------------ loss + gradients
    def loss_and_grads(self, x: np.ndarray, y: np.ndarray):
        f, vjp = expectations_and_vjp(self.circuit, x, self.theta, self.observables)
        p = np.clip(self._prob_from_f(f), EPS, 1 - EPS)
        B = len(y)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        grads = {}
        if self.readout == "linear":
            dlogit = (p - y) / B                       # dL/dlogit for sigmoid + BCE
            grads["w"] = f.T @ dlogit
            grads["b"] = np.sum(dlogit)
            g = dlogit[:, None] * self.w[None, :]      # dL/dz
        else:
            dp = (p - y) / (p * (1 - p)) / B            # dL/dp
            g = (-0.5 * dp)[:, None]                    # dp/df = -1/2
        grads["theta"] = vjp(g)
        return loss, grads

    # ------------------------------------------------------------------ training
    def fit(self, x, y, epochs=100, lr=0.05, rng=None, x_val=None, y_val=None,
            beta1=0.9, beta2=0.999):
        rng = rng or np.random.default_rng()
        if self.theta is None:
            self.init(rng)
        params = {"theta": self.theta}
        if self.readout == "linear":
            params["w"] = self.w
            params["b"] = np.array(self.b)
        m = {k: np.zeros_like(v, dtype=float) for k, v in params.items()}
        v = {k: np.zeros_like(val, dtype=float) for k, val in params.items()}
        hist = {"loss": [], "grad_norm_theta": [], "val_auc": []}
        for t in range(1, epochs + 1):
            loss, grads = self.loss_and_grads(x, y)
            hist["loss"].append(float(loss))
            hist["grad_norm_theta"].append(float(np.linalg.norm(grads["theta"])))
            for k in params:
                gk = np.asarray(grads[k], dtype=float)
                m[k] = beta1 * m[k] + (1 - beta1) * gk
                v[k] = beta2 * v[k] + (1 - beta2) * gk**2
                mh, vh = m[k] / (1 - beta1**t), v[k] / (1 - beta2**t)
                params[k] = params[k] - lr * mh / (np.sqrt(vh) + 1e-8)
            self.theta = params["theta"]
            if self.readout == "linear":
                self.w, self.b = params["w"], float(params["b"])
            if x_val is not None and (t == epochs or t % 25 == 0):
                hist["val_auc"].append((t, auc(y_val, self.predict_proba(x_val))))
        self.history = hist
        return self


def auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))
