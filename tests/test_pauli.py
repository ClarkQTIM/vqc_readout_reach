import itertools

import numpy as np
import pytest

from vqcx import pauli as pl

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
M = {"I": I2, "X": X, "Y": Y, "Z": Z}


def dense(label: str) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for ch in label:
        out = np.kron(out, M[ch])
    return out


LABELS3 = ["".join(t) for t in itertools.product("IXYZ", repeat=3)]


@pytest.mark.parametrize("label", LABELS3)
def test_matrix_matches_kron(label):
    assert np.allclose(pl.matrix(pl.from_label(label), 3), dense(label))


def test_label_roundtrip():
    for label in LABELS3:
        assert pl.to_label(pl.from_label(label), 3) == label


def test_product_and_commutation():
    rng = np.random.default_rng(0)
    for _ in range(300):
        a, b = rng.choice(LABELS3, 2)
        k, r = pl.product(pl.from_label(a), pl.from_label(b))
        assert np.allclose(dense(a) @ dense(b), (1j) ** k * pl.matrix(r, 3))
        comm = np.allclose(dense(a) @ dense(b), dense(b) @ dense(a))
        assert comm == pl.commutes(pl.from_label(a), pl.from_label(b))


def cnot(n, c, t):
    U = np.zeros((2**n, 2**n))
    for i in range(2**n):
        bits = [(i >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[c]:
            bits[t] ^= 1
        j = sum(b << (n - 1 - q) for q, b in enumerate(bits))
        U[j, i] = 1
    return U


def cz(n, c, t):
    d = []
    for i in range(2**n):
        d.append(-1.0 if ((i >> (n - 1 - c)) & 1) and ((i >> (n - 1 - t)) & 1) else 1.0)
    return np.diag(d)


@pytest.mark.parametrize("gate,fn", [("cnot", cnot), ("cz", cz)])
def test_clifford_conjugation(gate, fn):
    n = 3
    for c, t in [(0, 1), (2, 0), (1, 2)]:
        G = fn(n, c, t)
        for label in LABELS3:
            k, r = pl.conjugate(pl.from_label(label), gate, n, c, t)
            assert k in (0, 2)
            assert np.allclose(G @ dense(label) @ G, (1j) ** k * pl.matrix(r, n))


def test_apply_and_expval_random_states():
    rng = np.random.default_rng(1)
    psi = rng.normal(size=(5, 8)) + 1j * rng.normal(size=(5, 8))
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    for label in ["XYZ", "ZII", "IYI", "XXX"]:
        P = dense(label)
        assert np.allclose(pl.apply(psi, pl.from_label(label), 3), psi @ P.T)
        ev = np.einsum("bi,ij,bj->b", psi.conj(), P, psi).real
        assert np.allclose(pl.expval(psi, pl.from_label(label), 3), ev)
