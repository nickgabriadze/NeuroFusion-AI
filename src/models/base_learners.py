"""Five base learners (M1-M5) for NeuroFusion-AI.

Every learner is an sklearn Pipeline:
    median imputation (+ missingness indicators) -> standardisation -> classifier
so imputation and scaling are fitted ONLY on whatever data .fit() receives
(i.e. the training fold). All learners expose fit / predict_proba.
Labels must be integers 0..K-1 (e.g. CN=0, MCI=1, AD=2).
"""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # fallback keeps the code runnable without xgboost
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False


def _preprocess() -> list:
    return [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]


def make_logistic(seed: int = 42) -> Pipeline:
    """M1 - multinomial logistic regression (transparent linear baseline)."""
    clf = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=seed
    )  # lbfgs solver -> multinomial for >2 classes
    return Pipeline(_preprocess() + [("clf", clf)])


def make_svm(seed: int = 42) -> Pipeline:
    """M2 - RBF SVM. probability=True applies internal Platt scaling (5-fold)."""
    clf = SVC(
        kernel="rbf", C=1.0, gamma="scale", probability=True,
        class_weight="balanced", random_state=seed,
    )
    return Pipeline(_preprocess() + [("clf", clf)])


def make_random_forest(seed: int = 42) -> Pipeline:
    """M3 - random forest (bagged trees)."""
    clf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed,
    )
    return Pipeline(_preprocess() + [("clf", clf)])


def make_xgboost(seed: int = 42) -> Pipeline:
    """M4 - gradient-boosted trees (XGBoost; HistGradientBoosting if unavailable)."""
    if HAS_XGB:
        clf = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="multi:softprob",
            eval_metric="mlogloss", tree_method="hist", n_jobs=-1,
            random_state=seed,
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, random_state=seed
        )
    return Pipeline(_preprocess() + [("clf", clf)])


def make_mlp(seed: int = 42) -> Pipeline:
    """M5 - small feed-forward neural network."""
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
        validation_fraction=0.15, max_iter=500, random_state=seed,
    )
    return Pipeline(_preprocess() + [("clf", clf)])


def get_base_learners(seed: int = 42) -> dict[str, Pipeline]:
    """Return the five base learners keyed by short name."""
    return {
        "LR": make_logistic(seed),
        "SVM": make_svm(seed),
        "RF": make_random_forest(seed),
        "XGB": make_xgboost(seed),
        "MLP": make_mlp(seed),
    }
