# Unified Credit Card Fraud Research Project

This project contains TWO ways of running the same fixed experimental workflow:

1. `notebooks/COMPLETE_fraud_research_all_experiments.ipynb` is **fully standalone** (no imports from local `fraud_project`) and contains the complete code for EXP_001–EXP_017 plus EXP_018 validation-only paired error analysis, EDA, plots, exports and a disabled final-test gate.
2. `fraud_project/` contains **modularized versions of the exact same function definitions**, accessed via `run_experiments.py` and `run_analysis.py`.

The project packages your four original **historical validation-prediction files** in `historical_predictions/`. These are **not new results**. They allow EXP_018 to run without retraining when you do not have a GPU/CPU budget for re-running 17 models.

## Quick start on Windows

Unzip and open Command Prompt in the top-level project folder.

```bat
python -m pip install -r requirements.txt
jupyter notebook notebooks/COMPLETE_fraud_research_all_experiments.ipynb
```

Run the notebook cells from top to bottom. On first run, leave `RUN_MODEL_TRAINING=True` to re-run 17 experiments. The notebook downloads `data/creditcard.csv` or uses your copy and checks the original SHA-256. For a fast analysis-only session, change `RUN_MODEL_TRAINING=False`; the bundled historical predictions are loaded for EXP_018. If you copy your earlier result files into `results/`, setting `REUSE_EXISTING_RUNS=True` avoids retraining provided their shapes match the original protocol. Do **not** combine outputs from mismatched protocols.

Or, use the modular command-line implementation:

```bat
python run_experiments.py EXP_008 EXP_010 EXP_012
python run_experiments.py EXP_001 EXP_002 EXP_003 EXP_004 EXP_005 EXP_006 EXP_007 EXP_008 EXP_009 EXP_010 EXP_011 EXP_012 EXP_013 EXP_014 EXP_015 EXP_016 EXP_017
python run_analysis.py --bootstrap 500
python run_analysis.py --dataset data\creditcard.csv --bootstrap 500
```

`run_experiments.py` intentionally uses different filename from your old `run_experiment.py` to avoid overwriting the previous project. The command names above should be run **from the top-level project directory**.

## Modular structure

- `fraud_project/config.py`: dataset info, seed and frozen 17-experiment registry
- `fraud_project/data.py`: download, checksum, audit, chronology and features
- `fraud_project/models.py`: exact 8 model-family constructors and scoring
- `fraud_project/metrics.py`: AP, ROC-AUC, top-k and threshold metrics
- `fraud_project/engine.py`: train/validate/compare and one-time final test gate
- `fraud_project/analysis.py`: EXP_018 paired cases, temporal segments and exploratory bootstrap
- `run_experiments.py`: training CLI
- `run_analysis.py`: paired analysis CLI
- `tests/test_consistency.py`: smoke checks and exact comparison against historical prediction files
- `RESEARCH_ROADMAP.md`: meaning of editorial checks, limitations and future tasks

## Scientific integrity

- The original ULB dataset contains 284,807 transactions including 492 frauds before cleaning; after 1,081 exact duplicate removals, 283,726 transactions and 473 frauds remain.
- Time-respecting split: train 170,236 / validation 56,746 / sealed test 56,744. Only 57 frauds occur in validation.
- Original input feature semantics V1–V28 are anonymous PCA components, not customer attributes.
- Model/metric/SMOTE settings are **unchanged** from the user's prior run.
- Sealed test is deliberately not run. Independent IEEE-CIS replication is **pending**, not fabricated.
- Avoid publishing any claims based solely on repeated validation model comparisons as if they were independent tests.

## Replication status

`EXP_001–017`: 17 original completed model configurations, reproducible in both formats. `EXP_018`: paired analysis is reproducible from historical predictions. `EXP_020` and final held-out evaluation: future work requiring explicit completion and verification. **The presence of a runnable notebook does not mean the full independent research study has been validated.**
