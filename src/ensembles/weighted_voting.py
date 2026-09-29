"""Validation-tuned weighted voting ensemble."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from .base import _Base


class WeightedVoting(_Base):
    """B - weights minimise validation log-loss on the probability simplex."""

    def fit(self, X_train, y_train, X_val, y_val, **_):
        self._fit_bases(X_train, y_train)
        val_probas = self._base_probas(X_val)
        m = len(self.names)

        def loss(w):
            p = np.clip(np.tensordot(w, val_probas, axes=1), 1e-12, 1.0)
            p /= p.sum(axis=1, keepdims=True)
            return log_loss(y_val, p, labels=self.classes_)

        res = minimize(
            loss, x0=np.full(m, 1.0 / m), method="SLSQP",
            bounds=[(0.0, 1.0)] * m,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        )
        self.weights_ = {n: round(float(w), 4) for n, w in zip(self.names, res.x)}
        self._w = res.x
        return self

    def predict_proba(self, X):
        return np.tensordot(self._w, self._base_probas(X), axes=1)
