"""Classification tasks.

Synthetic tasks put the label on one feature, x_k, with every feature drawn from U[-pi/2, pi/2]
(where sin is monotone in x and cos is monotone in |x|):

* ``sign``      : y = [x_k > 0]           -- needs an odd function of x_k (sin x_k).
* ``magnitude`` : y = [|x_k| > pi/4]      -- needs an even function of x_k (cos x_k).

Both are balanced, and every other feature is pure noise, so the readout-reach rule predicts
exactly which circuits can solve which task.

Real tasks: Iris class pairs, standardized as in the original sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HALF_PI = np.pi / 2
SYNTHETIC = ("sign", "magnitude")
IRIS_PAIRS = {"iris_setosa_versicolor": (0, 1), "iris_versicolor_virginica": (1, 2)}


@dataclass
class Data:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature: int | None = None  # the label-carrying feature (synthetic tasks)


def _synthetic(kind: str, n_features: int, feature: int, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(-HALF_PI, HALF_PI, size=(n, n_features))
    if kind == "sign":
        y = (x[:, feature] > 0).astype(float)
    elif kind == "magnitude":
        y = (np.abs(x[:, feature]) > HALF_PI / 2).astype(float)
    else:
        raise ValueError(kind)
    return x, y


def make_task(name: str, n_features: int = 4, feature: int = 3, n_train: int = 200,
              n_test: int = 1000, rng: np.random.Generator | None = None,
              test_fraction: float = 0.3) -> Data:
    rng = rng or np.random.default_rng()
    if name in SYNTHETIC:
        xtr, ytr = _synthetic(name, n_features, feature, n_train, rng)
        xte, yte = _synthetic(name, n_features, feature, n_test, rng)
        return Data(name, xtr, ytr, xte, yte, feature)
    if name in IRIS_PAIRS:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        if n_features != 4:
            raise ValueError("Iris has 4 features")
        d = load_iris()
        a, b = IRIS_PAIRS[name]
        keep = np.isin(d.target, [a, b])
        x, y = d.data[keep], (d.target[keep] == b).astype(float)
        xtr, xte, ytr, yte = train_test_split(
            x, y, test_size=test_fraction, stratify=y, random_state=int(rng.integers(2**31)))
        scaler = StandardScaler().fit(xtr)  # fit on train only
        return Data(name, scaler.transform(xtr), ytr, scaler.transform(xte), yte)
    raise ValueError(f"unknown task {name!r}")
