"""Shared base helpers for ensemble models."""
from __future__ import annotations

import numpy as np


class _Base:
    def __init__(self, learners: dict):
        self.learners = learners
        self.names = list(learners)
        self.fitted_: dict = {}
        self.classes_ = None

    def _fit_bases(self, X, y):
        self.fitted_ = {n: m.fit(X, y) for n, m in self.learners.items()}
        self.classes_ = np.unique(y)

    def _base_probas(self, X) -> list[np.ndarray]:
        return [self.fitted_[n].predict_proba(X) for n in self.names]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
