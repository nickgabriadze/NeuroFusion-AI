"""Logistic-regression stacking over out-of-fold base predictions."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from .base import _Base


class StackingEnsemble(_Base):
    """C - LR meta-learner trained on out-of-fold base probabilities.

    OOF folds are grouped by participant (groups=RID) so no participant's
    visits are split between the fold that trains and the fold that predicts.
    """

    def __init__(self, learners: dict, n_splits: int = 5, seed: int = 42):
        super().__init__(learners)
        self.n_splits, self.seed = n_splits, seed

    def fit(self, X, y, groups, **_):
        cv = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed)
        oof = [
            cross_val_predict(clone(m), X, y, groups=groups, cv=cv,
                              method="predict_proba")
            for m in self.learners.values()
        ]
        Z = np.hstack(oof)
        self.meta_ = LogisticRegression(max_iter=2000, random_state=self.seed).fit(Z, y)
        self._fit_bases(X, y)
        return self

    def predict_proba(self, X):
        return self.meta_.predict_proba(np.hstack(self._base_probas(X)))
