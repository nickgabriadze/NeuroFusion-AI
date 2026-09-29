# NeuroFusion-AI

Explainable multimodal ML for Alzheimer's classification (CN/MCI/AD) and MCI->AD progression on ADNI.

## Run
```bash
pip install -r requirements.txt
python -m experiments.run_task_a     # demo on synthetic data
pytest -q                            # leakage + ensemble tests
```

## Layout
| Folder | Contents |
|---|---|
| `data/raw/...` | Manually downloaded ADNI files (immutable, git-ignored) |
| `data/dictionaries, interim, processed` | Data dictionaries and derived datasets |
| `configs/` | YAML experiment configuration |
| `src/data_import/` | Loaders / harmonisation (synthetic loader for now) |
| `src/models/` | One file per base learner (M1-M5) + registry |
| `src/ensembles/` | Soft voting, weighted voting, stacking |
| `src/experiments/` | Splits, metrics, bootstrap CIs |
| `experiments/` | Runnable experiment scripts |
| `models/` | Saved model artefacts |
| `tests/` | Leakage and ensemble tests |
| `reports/`, `notebooks/` | Outputs and exploration |
