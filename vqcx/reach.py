"""What can the readout see?  Heisenberg-picture Pauli propagation of the measured observable.

Pushing the observable O back through the variational layers (U^dag O U) expands it into a set of
Pauli strings. A rotation exp(-i a P / 2) keeps a string Q that commutes with P and, when Q
anticommutes with P, adds the string P Q. A Clifford maps each string to one string.

The surviving strings are then evaluated on the encoded product state, prod_q RY(x_q)|0>:
per qubit I -> 1, Z -> cos x_q, X -> sin x_q, Y -> 0. So

* a string containing any Y contributes nothing;
* each other string contributes the monomial prod_{Z} cos x_q * prod_{X} sin x_q.

A feature is *odd* if some surviving string has X on its qubit (the model can use the sign of
x_q), *even* if it only ever appears with Z (the model sees only cos x_q, which cannot tell
x_q from -x_q), and *none* if it never appears. ``feature_parts`` keeps both pieces: a feature
whose parts are {'odd'} only gives an output that is odd in x_q, which cannot see |x_q|.

Since each x_q enters once, <O> is multilinear in (cos x_q, sin x_q). So if the label is
invariant under x_q -> -x_q and only odd parts reach the readout (or the label flips sign and
only even parts reach it), the classes are mirror images under the model and the expected AUC
is exactly 0.5, whatever the training, depth or expressibility.

Assumption: coefficients are generic, i.e. no two paths cancel exactly. ``tests/test_reach.py``
checks the rule against direct simulation over the whole grid.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import pauli as pl
from .circuits import Circuit, Clifford, Encode, Rot


@dataclass
class Reach:
    n_qubits: int
    strings: set[pl.Pauli]          # surviving strings at the encoding layer (no Y)
    feature_status: list[str]       # per feature: 'none' | 'even' | 'odd'
    n_propagated: int               # strings before dropping those with Y
    feature_parts: list[frozenset[str]]  # per feature: subset of {'even', 'odd'} that appears

    def can_use_sign(self, feature: int) -> bool:
        """Some term is odd in x_feature, so the model can tell x from -x."""
        return "odd" in self.feature_parts[feature]

    def can_use_magnitude(self, feature: int) -> bool:
        """Some term is even and non-constant in x_feature (cos x), so |x| is visible."""
        return "even" in self.feature_parts[feature]

    @property
    def odd_features(self) -> list[int]:
        return [i for i, s in enumerate(self.feature_status) if s == "odd"]

    @property
    def reached_features(self) -> list[int]:
        return [i for i, s in enumerate(self.feature_status) if s != "none"]

    def labels(self) -> list[str]:
        return sorted(pl.to_label(p, self.n_qubits) for p in self.strings)


def propagate(circuit: Circuit, observable: pl.Pauli) -> set[pl.Pauli]:
    """Support of U^dag O U in the Pauli basis, assuming generic angles."""
    n = circuit.n_qubits
    strings = {observable}
    for op in reversed(circuit.ops):
        if isinstance(op, Encode):
            continue
        if isinstance(op, Rot):
            new = set(strings)
            for q in strings:
                if not pl.commutes(op.pauli, q):
                    new.add(pl.product(op.pauli, q)[1])
            strings = new
        elif isinstance(op, Clifford):
            strings = {pl.conjugate(q, op.gate, n, op.control, op.target)[1] for q in strings}
    return strings


def readout_reach(circuit: Circuit, observable: pl.Pauli) -> Reach:
    n = circuit.n_qubits
    feature_of = {op.qubit: op.feature for op in circuit.ops if isinstance(op, Encode)}
    n_features = max(feature_of.values()) + 1 if feature_of else 0
    if sorted(feature_of) != list(range(n)) or any(
        not isinstance(op, Encode) for op in circuit.ops[:n]
    ):
        raise ValueError("readout_reach assumes one RY encoding per qubit before all other gates")

    propagated = propagate(circuit, observable)
    surviving = {p for p in propagated if "Y" not in pl.to_label(p, n)}
    status = ["none"] * n_features
    parts: list[set[str]] = [set() for _ in range(n_features)]
    for p in surviving:
        for q in range(n):
            ch = pl.letter(p, n, q)
            f = feature_of[q]
            if ch == "X":
                status[f] = "odd"
                parts[f].add("odd")
            elif ch == "Z":
                parts[f].add("even")
                if status[f] == "none":
                    status[f] = "even"
    return Reach(n, surviving, status, len(propagated), [frozenset(s) for s in parts])
