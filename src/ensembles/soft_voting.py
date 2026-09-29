"""Mean-probability ensemble strategy."""
from __future__ import annotations

import numpy as np

from .base import _Base


class SoftVoting(_Base):
    """A - equal-weight mean of the five probability vectors."""

    def fit(self, X, y, **_):
        self._fit_bases(X, y)
        return self

    def predict_proba(self, X):
        return np.mean(self._base_probas(X), axis=0)
