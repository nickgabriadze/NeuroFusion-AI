"""Three ensemble strategies over the five base learners.

A. SoftVoting      - mean of probability vectors
B. WeightedVoting  - weights w_m >= 0, sum w_m = 1, chosen on VALIDATION data
C. StackingEnsemble- logistic-regression meta-learner on OUT-OF-FOLD probabilities

The test set is never touched by any of them.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict


class _Base:
    def __init__(self, learners: dict):
        self.learners = learners
        self.names = list(learners)
        self.fitted_: dict = {}
        self.classes_ = None

    def _fit_bases(self, X, y):
        self.fitted_ = {n: clone(m).fit(X, y) for n, m in self.learners.items()}
        self.classes_ = np.unique(y)

    def _base_probas(self, X) -> list[np.ndarray]:
        return [self.fitted_[n].predict_proba(X) for n in self.names]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class SoftVoting(_Base):
    """A - equal-weight mean of the five probability vectors."""

    def fit(self, X, y, **_):
        self._fit_bases(X, y)
        return self

    def predict_proba(self, X):
        return np.mean(self._base_probas(X), axis=0)


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
        Z = np.hstack(oof)                       # [P_LR | P_SVM | P_RF | P_XGB | P_MLP]
        self.meta_ = LogisticRegression(max_iter=2000, random_state=self.seed).fit(Z, y)
        self._fit_bases(X, y)                    # refit bases on all training data
        return self

    def predict_proba(self, X):
        return self.meta_.predict_proba(np.hstack(self._base_probas(X)))
