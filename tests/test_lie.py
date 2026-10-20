import pytest

from vqcx.circuits import CircuitSpec, build_circuit
from vqcx.lie import _closure_dense, _closure_strings, layer_generators, lie_algebra_dimension


def dim(n, rot, ent, topo="ring"):
    return lie_algebra_dimension(build_circuit(CircuitSpec(n, 1, rot, ent, topo)))


# Values computed independently with dense matrices at the start of the project.
@pytest.mark.parametrize(
    "rot,ent,expected",
    [
        ("rx", "ising_xx", (7, 8, 10)),
        ("rx", "ising_yy", (28, 56, 126)),
        ("rx", "ising_zz", (28, 56, 126)),
        ("ry", "ising_yy", (7, 8, 10)),
        ("rz", "ising_zz", (7, 8, 10)),
        ("u3", "ising_xx", (255, 255, 255)),
        ("u3", "ising_zz", (255, 255, 255)),
    ],
)
def test_ising_table_four_qubits(rot, ent, expected):
    assert tuple(dim(4, rot, ent, t) for t in ("line", "ring", "all_to_all")) == expected


def test_two_qubits():
    assert dim(2, "rx", "ising_xx", "all_to_all") == 3
    assert dim(2, "rx", "ising_zz", "all_to_all") == 6
    assert dim(2, "u3", "ising_zz", "all_to_all") == 15


def test_no_entangler_is_local():
    assert dim(4, "rx", "none") == 4          # four commuting X_i
    assert dim(4, "u3", "none") == 4 * 3      # su(2) on each qubit


def test_cnot_with_single_axis_x_is_abelian():
    # CNOT maps X-strings to X-strings, so RX + CNOT stays inside the X-string algebra (dim <= 15).
    d = dim(4, "rx", "cnot", "ring")
    assert 4 <= d <= 15


def test_dense_and_string_closures_agree():
    for rot, ent in [("rx", "ising_zz"), ("ry", "cnot"), ("rz", "cz")]:
        circ = build_circuit(CircuitSpec(3, 1, rot, ent, "ring"))
        gens = layer_generators(circ)
        assert _closure_strings({next(iter(g)) for g in gens}) == _closure_dense(gens, 3)


def _unitary(circ, theta):
    import numpy as np

    from vqcx import pauli as pl
    from vqcx.circuits import Clifford, Encode, Rot
    from vqcx.simulate import _apply_clifford, _rotate

    n = circ.n_qubits
    state = np.eye(2**n, dtype=complex)  # rows = basis inputs
    for op in circ.ops:
        if isinstance(op, Encode):
            continue
        if isinstance(op, Rot):
            state = _rotate(state, op.pauli, np.full(2**n, op.coeff * theta[op.param]), n)
        elif isinstance(op, Clifford):
            state = _apply_clifford(state, op, n)
    return state.T  # columns = U|b>


@pytest.mark.parametrize(
    "rot,ent,topo",
    [("rx", "cnot", "ring"), ("ry", "cz", "ring"), ("rz", "cnot", "line"),
     ("rx", "crx", "ring"), ("ry", "cry", "line"), ("rx", "ising_zz", "ring")],
)
def test_jacobian_rank_at_depth_matches_dimension(rot, ent, topo):
    """At large depth the circuit's tangent space at a random point spans the whole algebra."""
    import numpy as np

    n, layers = 3, 14
    circ = build_circuit(CircuitSpec(n, layers, rot, ent, topo))
    rng = np.random.default_rng(3)
    theta = rng.uniform(0, 2 * np.pi, circ.n_params)
    U = _unitary(circ, theta)
    h = 1e-6
    tangents = []
    for k in range(circ.n_params):
        e = np.zeros(circ.n_params)
        e[k] = h
        dU = (_unitary(circ, theta + e) - _unitary(circ, theta - e)) / (2 * h)
        A = U.conj().T @ dU  # anti-Hermitian element of the algebra
        tangents.append(np.concatenate([A.real.ravel(), A.imag.ravel()]))
    s = np.linalg.svd(np.array(tangents), compute_uv=False)
    rank = int(np.sum(s > 1e-6 * s[0]))
    single_layer = build_circuit(CircuitSpec(n, 1, rot, ent, topo))
    assert rank == lie_algebra_dimension(single_layer)


def test_tied_generators_never_exceed_split_ones():
    for rot in ("rx", "ry", "rz"):
        for ent in ("crx", "cry"):
            circ = build_circuit(CircuitSpec(3, 1, rot, ent, "ring"))
            gens = layer_generators(circ)
            split = {p for g in gens for p in g}
            assert lie_algebra_dimension(circ) <= _closure_strings(split)
