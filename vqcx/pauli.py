"""Pauli strings as bit masks.

Conventions
-----------
* A Pauli string on n qubits is a pair of ints ``(x, z)``. Qubit ``q`` is bit ``n - 1 - q``,
  so qubit 0 is the most significant bit (the PennyLane / big-endian convention).
* The Hermitian Pauli string is ``P(x, z) = i**popcount(x & z) * X**x Z**z``, so a qubit with
  both bits set is Y = iXZ.
"""

from __future__ import annotations

import numpy as np

Pauli = tuple[int, int]

_CHARS = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}
_BITS = {v: k for k, v in _CHARS.items()}


def popcount(v: int) -> int:
    return bin(v).count("1")


def from_label(label: str) -> Pauli:
    """'XIZY' -> (x, z). Character k acts on qubit k."""
    n = len(label)
    x = z = 0
    for q, ch in enumerate(label.upper()):
        bx, bz = _BITS[ch]
        bit = 1 << (n - 1 - q)
        x |= bit * bx
        z |= bit * bz
    return x, z


def to_label(p: Pauli, n: int) -> str:
    x, z = p
    return "".join(_CHARS[((x >> (n - 1 - q)) & 1, (z >> (n - 1 - q)) & 1)] for q in range(n))


def single(n: int, qubit: int, ch: str) -> Pauli:
    """Pauli ``ch`` on one qubit."""
    bx, bz = _BITS[ch.upper()]
    bit = 1 << (n - 1 - qubit)
    return bit * bx, bit * bz


def on(n: int, ops: dict[int, str]) -> Pauli:
    """Pauli string from {qubit: 'X'|'Y'|'Z'}."""
    x = z = 0
    for q, ch in ops.items():
        px, pz = single(n, q, ch)
        x |= px
        z |= pz
    return x, z


def letter(p: Pauli, n: int, qubit: int) -> str:
    x, z = p
    return _CHARS[((x >> (n - 1 - qubit)) & 1, (z >> (n - 1 - qubit)) & 1)]


def commutes(p: Pauli, q: Pauli) -> bool:
    return (popcount(p[0] & q[1]) + popcount(p[1] & q[0])) % 2 == 0


def product(p: Pauli, q: Pauli) -> tuple[int, Pauli]:
    """P Q = i**k R. Returns (k mod 4, R)."""
    x1, z1 = p
    x2, z2 = q
    x3, z3 = x1 ^ x2, z1 ^ z2
    k = popcount(x1 & z1) + popcount(x2 & z2) + 2 * popcount(z1 & x2) - popcount(x3 & z3)
    return k % 4, (x3, z3)


# ---------------------------------------------------------------- Clifford conjugation
def _images(gate: str, n: int, c: int, t: int) -> dict[Pauli, tuple[int, Pauli]]:
    """G X_q G and G Z_q G for the self-inverse two-qubit Cliffords we use."""
    X = lambda q: single(n, q, "X")
    Z = lambda q: single(n, q, "Z")
    mul = lambda a, b: (a[0] ^ b[0], a[1] ^ b[1])  # X/Z-only strings: phases are trivial here
    if gate == "cnot":
        return {X(c): (0, mul(X(c), X(t))), Z(t): (0, mul(Z(c), Z(t)))}
    if gate == "cz":
        return {X(c): (0, mul(X(c), Z(t))), X(t): (0, mul(Z(c), X(t)))}
    raise ValueError(gate)


def conjugate(p: Pauli, gate: str, n: int, c: int, t: int) -> tuple[int, Pauli]:
    """G P G for G in {cnot, cz} (both self-inverse). Returns (k, R) with G P G = i**k R, k in {0, 2}."""
    imgs = _images(gate, n, c, t)
    x, z = p
    k_total = popcount(x & z)  # P = i^a X^x Z^z
    acc: Pauli = (0, 0)
    for q in range(n):  # X^x part
        bit = 1 << (n - 1 - q)
        if x & bit:
            k_img, img = imgs.get((bit, 0), (0, (bit, 0)))
            k, acc = product(acc, img)
            k_total += k + k_img
    for q in range(n):  # Z^z part
        bit = 1 << (n - 1 - q)
        if z & bit:
            k_img, img = imgs.get((0, bit), (0, (0, bit)))
            k, acc = product(acc, img)
            k_total += k + k_img
    return k_total % 4, acc


# ---------------------------------------------------------------- statevector actions
def _parity_table(n: int) -> np.ndarray:
    idx = np.arange(2**n)
    counts = np.zeros(2**n, dtype=np.int64)
    for b in range(n):
        counts += (idx >> b) & 1
    return counts


_PARITY_CACHE: dict[int, np.ndarray] = {}


def _popcounts(n: int) -> np.ndarray:
    if n not in _PARITY_CACHE:
        _PARITY_CACHE[n] = _parity_table(n)
    return _PARITY_CACHE[n]


def apply(state: np.ndarray, p: Pauli, n: int) -> np.ndarray:
    """P|psi> for states of shape (..., 2**n)."""
    x, z = p
    idx = np.arange(2**n)
    src = idx ^ x
    sign = 1 - 2 * (_popcounts(n)[src & z] & 1)
    phase = (1j) ** (popcount(x & z) % 4)
    return phase * sign * state[..., src]


def expval(state: np.ndarray, p: Pauli, n: int) -> np.ndarray:
    """<psi|P|psi> (real) for states of shape (..., 2**n)."""
    return np.real(np.sum(np.conj(state) * apply(state, p, n), axis=-1))


def matrix(p: Pauli, n: int) -> np.ndarray:
    """Dense matrix of P (for tests)."""
    return apply(np.eye(2**n, dtype=complex), p, n).T
