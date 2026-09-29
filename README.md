# NeuroFusion-AI

**Explainable multimodal machine learning for Alzheimer's disease classification and longitudinal progression prediction using ADNI.**

NeuroFusion-AI is a Python research platform investigating how clinical, cognitive, MRI/PET, biofluid, and genetic information can be combined to study Alzheimer's disease. It focuses on current diagnosis and future MCI-to-AD progression, with reproducibility, participant-level evaluation, calibration, uncertainty, and interpretability built into the research design.

> **Research use only.** Model outputs are probabilistic estimates, not clinical diagnoses or treatment advice. This is not a diagnostic device.

## At a glance

| Area | What the project covers |
|---|---|
| Research question | Whether multimodal and longitudinal data improve disease-state and progression prediction over conventional baselines. |
| Prediction tasks | Classify current status as CN, MCI, or AD; estimate whether MCI progresses to AD within a defined horizon. |
| Data | Longitudinal ADNI participant visits with clinical, cognitive, MRI/PET, biofluid, and genetic features. Data are provided manually by approved researchers. |
| Models | Five tabular learners—logistic regression, SVM, random forest, XGBoost (or fallback), and MLP—plus soft voting, weighted voting, and stacking. |
| Fusion and time | The research plan compares early, late, and learned multimodal fusion, as well as baseline and longitudinal prediction. |
| Evaluation | Macro-F1, sensitivity, AUROC/AUPRC, Brier score, calibration, subgroup and missing-modality analyses, and participant-level bootstrap intervals. |
| Leakage safeguards | Keep each participant within one data partition; fit preprocessing and model choices only on permitted training/validation data; respect prediction-time cut-offs. |
| Trustworthiness | Assess calibration, uncertainty, explainability, robustness, and limitations; model explanations are not causal claims. |

## Research plan

The central hypothesis is that combining modalities and longitudinal trajectories can add predictive value beyond single-modality or baseline-only models. The experiments compare models on the same held-out participants and assess not only discrimination, but also calibration, robustness to missing data, and subgroup performance.

The system is designed as a pipeline: ADNI data governance and harmonization → participant/visit representation → feature and modality modelling → fusion and prediction → evaluation and interpretation. The [overview](docs/overview/README.md) describes the research questions, hypotheses, tasks, and architecture.

## Data and methods

The core dataset is a participant-by-visit table, aligned using both participant and visit/time information. Raw ADNI files remain immutable and are never downloaded by this software or committed to public version control. The initial implemented models use numerical features; direct imaging and broader multimodal modelling are planned extensions.

All five base learners use probability outputs and share leakage-aware preprocessing. The ensemble strategies are equal-weight soft voting, validation-tuned weighted voting, and stacking trained on participant-grouped out-of-fold predictions. Planned extensions include multimodal fusion, missing-modality handling, and temporal encoders. See [data and cohort](docs/data/README.md) and [methods](docs/methods/README.md).

## Evaluation and responsible use

Evaluation is participant-based: repeated visits from one person must not cross train, validation, and test partitions. Confidence intervals resample participants rather than individual rows. Results should include class-level performance and calibration, and report missingness, exclusions, subgroup sizes, and limitations. The platform is for research support only—not diagnosis or treatment. More detail is in [evaluation and trustworthiness](docs/evaluation/README.md).

## Implementation status

The current foundation includes five base learners, the three ensemble approaches, participant-level splitting, metrics, bootstrap confidence intervals, and a synthetic-data demo. ADNI loading and harmonization, cohort construction, multimodal fusion, progression models, calibration, explainability, and the dashboard are planned. The synthetic demo verifies execution only; its scores have no scientific meaning.

The code is organized under `src/models/`, `src/ensembles/`, and `src/experiments/`; runnable experiment scripts live under `experiments/`. See [implementation status, project structure, and roadmap](docs/project/README.md).

## Quick start

Run commands from the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux:       source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.run_task_a
pytest -q
```

## Documentation

- [Documentation home](docs/README.md)
- [Overview, research question, and architecture](docs/overview/README.md)
- [ADNI data and cohort considerations](docs/data/README.md)
- [Models, ensembles, and fusion methods](docs/methods/README.md)
- [Evaluation, leakage prevention, and trustworthiness](docs/evaluation/README.md)
- [Implementation status, structure, and roadmap](docs/project/README.md)
