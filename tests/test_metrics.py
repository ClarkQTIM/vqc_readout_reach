import numpy as np

from vqcx.circuits import CircuitSpec, build_circuit
from vqcx.metrics import entanglement, expressibility_kl, haar_references
from vqcx.simulate import run, run_batch


def haar_states(n_states, n, rng):
    v = rng.normal(size=(n_states, 2**n)) + 1j * rng.normal(size=(n_states, 2**n))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_run_batch_matches_run():
    rng = np.random.default_rng(0)
    circ = build_circuit(CircuitSpec(3, 2, "u3", "crx", "ring"))
    thetas = rng.uniform(0, 2 * np.pi, (4, circ.n_params))
    x = rng.uniform(-1, 1, (4, 3))
    batch = run_batch(circ, x, thetas)
    for b in range(4):
        assert np.allclose(batch[b], run(circ, x[b:b + 1], thetas[b])[0])


def test_haar_states_have_near_zero_kl_and_haar_entanglement():
    rng = np.random.default_rng(1)
    circ = build_circuit(CircuitSpec(4, 1, "u3", "none"))  # only used for n_qubits here
    states = haar_states(10000, 4, rng)
    assert expressibility_kl(circ, n_pairs=5000, states=states) < 0.03
    ent = entanglement(circ, n_samples=2000, states=states)
    ref = haar_references(4)
    assert abs(ent["meyer_wallach"] - ref["meyer_wallach"]) < 0.01
    assert abs(ent["vn_entropy_bits"] - ref["vn_entropy_bits"]) < 0.02


def test_reference_values():
    ref = haar_references(2)
    assert np.isclose(ref["vn_entropy_bits"], (1 / 3) / np.log(2))   # Page: 1/3 nat for 2 qubits
    assert np.isclose(haar_references(4)["meyer_wallach"], 14 / 17)


def test_rz_only_never_moves_and_product_circuits_have_no_entanglement():
    rng = np.random.default_rng(2)
    rz = build_circuit(CircuitSpec(4, 3, "rz", "cz", "ring"))
    assert expressibility_kl(rz, n_pairs=500, rng=rng) > 20
    ent = entanglement(rz, n_samples=50, rng=rng)
    assert ent["meyer_wallach"] < 1e-10 and ent["vn_entropy_bits"] < 1e-10
    local = build_circuit(CircuitSpec(4, 3, "u3", "none"))
    assert entanglement(local, n_samples=50, rng=rng)["meyer_wallach"] < 1e-10
