# Methods

## 1. Base learners

The project targets five classical ML baselines with consistent preprocessing.

| ID | Model | Purpose |
|---|---|---|
| M1 | Logistic Regression | transparent linear baseline |
| M2 | SVM (RBF) | nonlinear margin classifier |
| M3 | Random Forest | robust tabular baseline |
| M4 | XGBoost / LightGBM | high-performance gradient boosting |
| M5 | MLP | neural tabular representation |

All learners are trained under the same participant-level split and use probability outputs.

## 2. Ensembles

The ensemble set is:

- Soft voting
- Weighted soft voting
- Stacking with out-of-fold probability features

The test set is never used to select ensemble weights or the stacking meta-model.

## 3. Multimodal fusion

Three design patterns are planned:

- early fusion
- late fusion
- learned hybrid fusion

## 4. Longitudinal modelling

Temporal models compare baseline-only prediction against the sequence of patient observations, including trajectory and change features where appropriate.

## 5. Missing data and missing modalities

The project distinguishes:

- feature-level missingness
- modality absence

Proposed handling includes imputation, missingness indicators, modality masks, and robust training-time masking.

## Related pages

- [Overview](../overview/README.md)
- [Data & cohort](../data/README.md)
- [Evaluation & trustworthiness](../evaluation/README.md)
- [Implementation & roadmap](../project/README.md)
