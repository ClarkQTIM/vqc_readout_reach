"""Layered circuit specs and their gate lists.

A circuit is: angle encoding RY(x_q) on every qubit q, then ``n_layers`` layers of
(single-qubit rotation on every qubit) followed by (entangler on every pair of the topology).
These conventions match the original cnn_vqc_expressivity sweep (PennyLane gate definitions):

* RX(a) = exp(-i a X / 2), likewise RY, RZ.
* U3(theta, phi, lam) = RZ(phi) RY(theta) RZ(lam), up to a global phase.
* CRX(a) = |0><0| x I + |1><1| x RX(a) = exp(-i a/4 X_t) exp(+i a/4 Z_c X_t); CRY likewise.
* IsingXX(a) = exp(-i a XX / 2); likewise YY, ZZ.
* Ring pairs are (i, i+1 mod n); all_to_all pairs are (i, j) for i < j; line pairs are (i, i+1).

Every parameterized gate is expressed as Pauli rotations exp(-i c * theta / 2 * P), so the
same gate list drives the simulator, the readout-reach rule and the Lie algebra computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import pauli as pl

ROTATIONS = ("rx", "ry", "rz", "u3")
ENTANGLERS = ("none", "cnot", "cz", "crx", "cry", "ising_xx", "ising_yy", "ising_zz")
TOPOLOGIES = ("ring", "all_to_all", "line")


@dataclass(frozen=True)
class Encode:
    """RY(x[feature]) on ``qubit``."""
    qubit: int
    feature: int


@dataclass(frozen=True)
class Rot:
    """exp(-i * coeff * theta[param] / 2 * P)."""
    pauli: pl.Pauli
    param: int
    coeff: float = 1.0


@dataclass(frozen=True)
class Clifford:
    gate: str  # 'cnot' | 'cz'
    control: int
    target: int


Op = Encode | Rot | Clifford


@dataclass(frozen=True)
class CircuitSpec:
    n_qubits: int
    n_layers: int
    rotation: str = "ry"
    entangler: str = "none"
    topology: str = "ring"

    def __post_init__(self):
        if self.rotation not in ROTATIONS:
            raise ValueError(f"rotation must be one of {ROTATIONS}")
        if self.entangler not in ENTANGLERS:
            raise ValueError(f"entangler must be one of {ENTANGLERS}")
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"topology must be one of {TOPOLOGIES}")

    @property
    def name(self) -> str:
        topo = "none" if self.entangler == "none" else self.topology
        return f"q{self.n_qubits}_l{self.n_layers}_{self.rotation}_{self.entangler}_{topo}"


@dataclass
class Circuit:
    spec: CircuitSpec
    ops: list[Op]
    n_params: int
    # Lie-algebra generator of each parameter: {pauli: coeff}
    generators: dict[int, dict[pl.Pauli, float]] = field(default_factory=dict)

    @property
    def n_qubits(self) -> int:
        return self.spec.n_qubits


def pairs(n: int, topology: str) -> list[tuple[int, int]]:
    if topology == "ring":
        return [(i, (i + 1) % n) for i in range(n)] if n > 1 else []
    if topology == "all_to_all":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if topology == "line":
        return [(i, i + 1) for i in range(n - 1)]
    raise ValueError(topology)


def build_circuit(spec: CircuitSpec) -> Circuit:
    n = spec.n_qubits
    ops: list[Op] = [Encode(q, q) for q in range(n)]
    gens: dict[int, dict[pl.Pauli, float]] = {}
    k = 0  # next parameter index

    def rot(p: pl.Pauli, coeff: float = 1.0, new_param: bool = True):
        nonlocal k
        if new_param:
            k += 1
        idx = k - 1
        ops.append(Rot(p, idx, coeff))
        gens.setdefault(idx, {})
        gens[idx][p] = gens[idx].get(p, 0.0) + coeff

    edge_list = [] if spec.entangler == "none" else pairs(n, spec.topology)
    for _ in range(spec.n_layers):
        for q in range(n):
            if spec.rotation == "u3":
                # parameters laid out (theta, phi, lam) per qubit; time order RZ(lam), RY(theta), RZ(phi)
                th, ph, lm = k, k + 1, k + 2
                k += 3
                for p_idx, ch in ((lm, "Z"), (th, "Y"), (ph, "Z")):
                    p = pl.single(n, q, ch)
                    ops.append(Rot(p, p_idx, 1.0))
                    gens.setdefault(p_idx, {})[p] = 1.0
            else:
                rot(pl.single(n, q, spec.rotation[1].upper()))
        for c, t in edge_list:
            e = spec.entangler
            if e in ("cnot", "cz"):
                ops.append(Clifford(e, c, t))
            elif e in ("crx", "cry"):
                a = e[2].upper()
                rot(pl.single(n, t, a), 0.5)
                rot(pl.on(n, {c: "Z", t: a}), -0.5, new_param=False)
            else:  # ising_xx / yy / zz
                a = e[-1].upper()
                rot(pl.on(n, {c: a, t: a}))
    return Circuit(spec, ops, k, gens)


def readout_pauli(n: int, readout: str) -> pl.Pauli:
    """'Z0' (one qubit), 'parity' (Z on every qubit) or an explicit label like 'ZIIZ'."""
    if readout == "parity":
        return pl.on(n, {q: "Z" for q in range(n)})
    if len(readout) >= 2 and readout[0] in "XYZ" and readout[1:].isdigit():
        return pl.single(n, int(readout[1:]), readout[0])
    if len(readout) == n:
        return pl.from_label(readout)
    raise ValueError(f"unknown readout {readout!r}")
