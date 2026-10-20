import itertools

import numpy as np
import pytest

from vqcx.circuits import ENTANGLERS, ROTATIONS, CircuitSpec, build_circuit
from vqcx.tasks import make_task
from vqcx.train import READOUTS, Classifier


CASES = list(itertools.product(ROTATIONS, ENTANGLERS, READOUTS))


@pytest.mark.parametrize("rotation,entangler,readout", CASES)
def test_gradients_match_finite_differences(rotation, entangler, readout):
    rng = np.random.default_rng(7)
    circ = build_circuit(CircuitSpec(3, 2, rotation, entangler, "ring"))
    clf = Classifier(circ, readout).init(rng)
    x = rng.uniform(-1.5, 1.5, size=(5, 3))
    y = rng.integers(0, 2, size=5).astype(float)
    _, grads = clf.loss_and_grads(x, y)
    h = 1e-6
    theta0 = clf.theta.copy()
    for k in range(circ.n_params):
        clf.theta = theta0.copy()
        clf.theta[k] += h
        lp, _ = clf.loss_and_grads(x, y)
        clf.theta[k] -= 2 * h
        lm, _ = clf.loss_and_grads(x, y)
        assert np.isclose(grads["theta"][k], (lp - lm) / (2 * h), atol=1e-6)
    clf.theta = theta0
    if readout == "linear":
        w0 = clf.w.copy()
        for k in range(len(w0)):
            clf.w = w0.copy(); clf.w[k] += h
            lp, _ = clf.loss_and_grads(x, y)
            clf.w = w0.copy(); clf.w[k] -= h
            lm, _ = clf.loss_and_grads(x, y)
            assert np.isclose(grads["w"][k], (lp - lm) / (2 * h), atol=1e-6)
        clf.w = w0


def test_training_solves_an_allowed_task():
    """RY + CNOT ring, one layer: feature 3 reaches qubit 0 with an odd part, so sign(x_3) is learnable."""
    rng = np.random.default_rng(0)
    data = make_task("sign", n_features=4, feature=3, n_train=200, n_test=400, rng=rng)
    circ = build_circuit(CircuitSpec(4, 1, "ry", "cnot", "ring"))
    clf = Classifier(circ, "z0").init(rng)
    clf.fit(data.x_train, data.y_train, epochs=80, lr=0.1, rng=rng)
    assert clf.history["loss"][-1] < clf.history["loss"][0]
    from vqcx.train import auc
    assert auc(data.y_test, clf.predict_proba(data.x_test)) > 0.9


def test_forbidden_task_stays_at_chance():
    """RX + CNOT ring, one layer: feature 3 reaches qubit 0 only as cos x_3, so sign(x_3) is invisible."""
    rng = np.random.default_rng(1)
    data = make_task("sign", n_features=4, feature=3, n_train=200, n_test=2000, rng=rng)
    circ = build_circuit(CircuitSpec(4, 1, "rx", "cnot", "ring"))
    clf = Classifier(circ, "z0").init(rng)
    clf.fit(data.x_train, data.y_train, epochs=80, lr=0.1, rng=rng)
    from vqcx.train import auc
    assert abs(auc(data.y_test, clf.predict_proba(data.x_test)) - 0.5) < 0.05
