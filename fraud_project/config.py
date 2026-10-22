"""Frozen experiment IDs and hyperparameters from original validated study."""
SEED = 42

V_COLS = [f"V{i}" for i in range(1, 29)]

REQUIRED_COLS = ["Time", *V_COLS, "Amount", "Class"]

FEATURE_NAMES = V_COLS + ["log_amount", "time_sin", "time_cos"]

DATASET_URL = "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv"

ORIGINAL_DATASET_PAGE = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

EXPERIMENTS = {
    "EXP_001": {"name": "Unweighted logistic regression", "family": "lr", "balanced": False, "smote": False,
                "hypothesis": "A simple linear baseline captures meaningful signal under native class prevalence."},
    "EXP_002": {"name": "Class-weighted logistic regression", "family": "lr", "balanced": True, "smote": False,
                "hypothesis": "Changing only class weights improves fraud recall without unacceptable alert burden."},
    "EXP_003": {"name": "Unweighted histogram gradient boosting", "family": "hgb", "balanced": False, "smote": False,
                "hypothesis": "Nonlinear tree boosting outperforms the simple linear baseline on future records."},
    "EXP_004": {"name": "Class-weighted histogram gradient boosting", "family": "hgb", "balanced": True, "smote": False,
                "hypothesis": "Class weighting improves boosted fraud detection at fixed alert budgets."},
    "EXP_005": {"name": "Extra Trees (2021 AutoML comparator)", "family": "extra", "balanced": False, "smote": False,
                "hypothesis": "Extra Trees is competitive with boosted trees under a leakage-safe split."},
    "EXP_006": {"name": "Train-only SMOTE + unweighted boosting", "family": "hgb", "balanced": False, "smote": True,
                "hypothesis": "Training-only minority interpolation improves boosted ranking relative to EXP_003."},
    "EXP_007": {"name": "Benign-only isolation forest", "family": "iforest", "balanced": False, "smote": False,
                "hypothesis": "An anomaly model trained only on normal training events can identify a subset of future fraud."},

    # Stage 2: new model-family comparisons with a single fixed configuration per
    # family, followed by within-family class weighting / train-only SMOTE.
    # No final test access, hyperparameter search, or validation-based early stopping.
    "EXP_008": {"name": "XGBoost unweighted", "family": "xgb", "balanced": False, "smote": False,
                "hypothesis": "Regularized histogram XGBoost is competitive under the original class prevalence."},
    "EXP_009": {"name": "XGBoost class-weighted", "family": "xgb", "balanced": True, "smote": False,
                "hypothesis": "Changing only train-fold class weighting improves XGBoost ranking vs EXP_008."},
    "EXP_010": {"name": "LightGBM unweighted", "family": "lgbm", "balanced": False, "smote": False,
                "hypothesis": "Restricted-leaf LightGBM is competitive under the original class prevalence."},
    "EXP_011": {"name": "LightGBM class-weighted", "family": "lgbm", "balanced": True, "smote": False,
                "hypothesis": "Changing only train-fold class weighting improves LightGBM ranking vs EXP_010."},
    "EXP_012": {"name": "CatBoost unweighted", "family": "cat", "balanced": False, "smote": False,
                "hypothesis": "CatBoost with fixed depth/iterations performs competitively on PCA-based numerical features."},
    "EXP_013": {"name": "CatBoost class-weighted", "family": "cat", "balanced": True, "smote": False,
                "hypothesis": "Changing only train-fold class weighting improves CatBoost ranking vs EXP_012."},
    "EXP_014": {"name": "Train-only SMOTE + XGBoost", "family": "xgb", "balanced": False, "smote": True,
                "hypothesis": "Training-only 1:10 SMOTE changes XGBoost ranking vs EXP_008."},
    "EXP_015": {"name": "Train-only SMOTE + LightGBM", "family": "lgbm", "balanced": False, "smote": True,
                "hypothesis": "Training-only 1:10 SMOTE changes LightGBM ranking vs EXP_010."},
    "EXP_016": {"name": "Train-only SMOTE + CatBoost", "family": "cat", "balanced": False, "smote": True,
                "hypothesis": "Training-only 1:10 SMOTE changes CatBoost ranking vs EXP_012."},
    "EXP_017": {"name": "AdaBoost decision stumps", "family": "ada", "balanced": False, "smote": False,
                "hypothesis": "A simple AdaBoost baseline establishes whether costly boosters are necessary."},
}

EXPECTED_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"

