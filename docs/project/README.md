# Implementation & roadmap

## 1. Current status

| Component | Status |
|---|---|
| Five base learners (LR, SVM, RF, XGBoost, MLP) | ✅ Implemented |
| Soft voting, weighted voting, stacking | ✅ Implemented |
| Participant-level split with leakage assertion | ✅ Implemented |
| Metrics: macro-F1, OvR AUROC, Brier, ECE | ✅ Implemented |
| Participant-level bootstrap CIs | ✅ Implemented |
| YAML configuration, tests, synthetic-data demo | ✅ Implemented |
| ADNI loader and harmonization | ⬜ Planned |
| Cohort builder and timeline engine | ⬜ Planned |
| Fusion engine | ⬜ Planned |
| Progression models and temporal encoders | ⬜ Planned |
| Calibration and uncertainty module | ⬜ Planned |
| Explainability and dashboard | ⬜ Planned |

## 2. Project structure

```text
NeuroFusion-AI/
├── configs/
├── data/
├── docs/
├── experiments/
├── src/
│   ├── data_import/
│   ├── models/
│   ├── ensembles/
│   └── experiments/
├── models/
├── reports/
├── notebooks/
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

## 3. Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.run_task_a
pytest -q
```

## 4. Roadmap

| Phase | Focus |
|---|---|
| 0 | Setup and governance |
| 1 | Cohort construction |
| 2 | Five base learners |
| 3 | Multimodal feature space |
| 4 | Ensemble methodology |
| 5 | MCI→AD progression |
| 6 | Calibration, uncertainty, XAI |
| 7 | Robustness and longitudinal analysis |
| 8 | Research dashboard |

## Related pages

- [Overview](../overview/README.md)
- [Data & cohort](../data/README.md)
- [Methods](../methods/README.md)
- [Evaluation & trustworthiness](../evaluation/README.md)
