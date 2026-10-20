"""Circuit-level metrics, computed on the circuit's output states from |0...0> (encoding off).

* ``expressibility_kl``: Sim et al.'s KL divergence between the fidelity distribution of pairs
  of independently sampled states and the Haar one, P(F) = (N - 1)(1 - F)^(N - 2). Unlike the
  original code, Haar bin probabilities are exact (integrated over each bin), pairs are
  independent, and parameters are drawn from U[0, 2pi]. Restricted circuits whose fidelities
  pile up near F = 1 get very large KL, since Haar puts almost no mass there.
* ``entanglement``: Meyer-Wallach Q and the mean von Neumann entropy over balanced
  bipartitions, in bits, with their Haar values for reference.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .circuits import Circuit
from .simulate import run_batch


def sample_states(circuit: Circuit, n: int, rng: np.random.Generator, theta_range: float = 2 * np.pi,
                  chunk: int = 2000) -> np.ndarray:
    out = []
    zeros = np.zeros((1, circuit.n_qubits))
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        thetas = rng.uniform(0, theta_range, size=(m, circuit.n_params))
        out.append(run_batch(circuit, zeros, thetas))
    return np.concatenate(out)


def haar_bin_probs(dim: int, n_bins: int) -> np.ndarray:
    edges = np.linspace(0, 1, n_bins + 1)
    cdf_tail = (1 - edges) ** (dim - 1)          # P(F > e)
    return cdf_tail[:-1] - cdf_tail[1:]


def expressibility_kl(circuit: Circuit, n_pairs: int = 5000, n_bins: int = 75,
                      rng: np.random.Generator | None = None, states: np.ndarray | None = None) -> float:
    rng = rng or np.random.default_rng()
    if states is None:
        states = sample_states(circuit, 2 * n_pairs, rng)
    a, b = states[:n_pairs], states[n_pairs:2 * n_pairs]
    fid = np.abs(np.sum(np.conj(a) * b, axis=1)) ** 2
    counts, _ = np.histogram(np.clip(fid, 0, 1), bins=n_bins, range=(0, 1))
    p = counts / counts.sum()
    q = haar_bin_probs(2**circuit.n_qubits, n_bins)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], 1e-300))))


def _entropy_bits(psi: np.ndarray, n: int, part: tuple[int, ...]) -> float:
    rest = [q for q in range(n) if q not in part]
    m = psi.reshape([2] * n).transpose(list(part) + rest).reshape(2 ** len(part), -1)
    s = np.linalg.svd(m, compute_uv=False) ** 2
    s = s[s > 1e-14]
    return float(-(s * np.log2(s)).sum())


def meyer_wallach(psi: np.ndarray, n: int) -> float:
    purities = []
    for q in range(n):
        rest = [k for k in range(n) if k != q]
        m = psi.reshape([2] * n).transpose([q] + rest).reshape(2, -1)
        rho = m @ m.conj().T
        purities.append(np.real(np.trace(rho @ rho)))
    return float(2 * (1 - np.mean(purities)))


def balanced_cuts(n: int) -> list[tuple[int, ...]]:
    cuts = list(combinations(range(n), n // 2))
    return cuts[: len(cuts) // 2] if n % 2 == 0 else cuts


def entanglement(circuit: Circuit, n_samples: int = 1000, rng: np.random.Generator | None = None,
                 states: np.ndarray | None = None) -> dict:
    rng = rng or np.random.default_rng()
    n = circuit.n_qubits
    if states is None:
        states = sample_states(circuit, n_samples, rng)
    states = states[:n_samples]
    cuts = balanced_cuts(n)
    vn = [np.mean([_entropy_bits(s, n, c) for c in cuts]) for s in states]
    mw = [meyer_wallach(s, n) for s in states]
    return {"meyer_wallach": float(np.mean(mw)), "vn_entropy_bits": float(np.mean(vn))}


def haar_references(n: int) -> dict:
    """Haar averages: Meyer-Wallach (2^n - 2)/(2^n + 1); Page entropy for an n//2 | n - n//2 cut."""
    N = 2**n
    a, b = 2 ** (n // 2), 2 ** (n - n // 2)
    page_nats = sum(1.0 / k for k in range(b + 1, a * b + 1)) - (a - 1) / (2 * b)
    return {"meyer_wallach": (N - 2) / (N + 1), "vn_entropy_bits": page_nats / np.log(2)}
