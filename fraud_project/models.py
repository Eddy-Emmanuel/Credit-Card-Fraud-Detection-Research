"""Model constructors: hyperparameters are unchanged from the prior experiments."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import SEED

def build_model(spec: dict):
    family = spec['family']
    if family == 'lr':
        return make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=1200, class_weight='balanced' if spec['balanced'] else None,
            solver='lbfgs', random_state=SEED))
    if family == 'hgb':
        return HistGradientBoostingClassifier(
            max_iter=160, max_leaf_nodes=15, min_samples_leaf=30,
            learning_rate=0.07, l2_regularization=1.0, random_state=SEED)
    if family == 'extra':
        return ExtraTreesClassifier(n_estimators=160, min_samples_leaf=2,
                                    max_features=0.8, n_jobs=-1, random_state=SEED)
    if family == 'iforest':
        return IsolationForest(n_estimators=160, max_samples=min(10000, int(1e7)),
                               contamination='auto', n_jobs=-1, random_state=SEED)
    if family == 'xgb':
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError('XGBoost is required for EXP_008/009/014. Run: pip install -r requirements.txt') from exc
        return XGBClassifier(
            objective='binary:logistic', n_estimators=350, learning_rate=0.045,
            max_depth=4, min_child_weight=8, subsample=0.85,
            colsample_bytree=0.90, reg_lambda=4.0, tree_method='hist',
            eval_metric='logloss', n_jobs=4, random_state=SEED)
    if family == 'lgbm':
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError('LightGBM is required for EXP_010/011/015. Run: pip install -r requirements.txt') from exc
        return LGBMClassifier(
            objective='binary', n_estimators=350, learning_rate=0.045,
            num_leaves=15, min_child_samples=50, subsample=0.85,
            subsample_freq=1, colsample_bytree=0.90, reg_lambda=4.0,
            verbosity=-1, n_jobs=4, random_state=SEED)
    if family == 'cat':
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError('CatBoost is required for EXP_012/013/016. Run: pip install -r requirements.txt') from exc
        return CatBoostClassifier(
            loss_function='Logloss', iterations=350, learning_rate=0.045,
            depth=5, l2_leaf_reg=4.0, random_seed=SEED,
            thread_count=4, verbose=False, allow_writing_files=False)
    if family == 'ada':
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
            n_estimators=200, learning_rate=0.07, random_state=SEED)
    raise KeyError(family)

def scores_for_model(model, X: pd.DataFrame, spec: dict) -> np.ndarray:
    if spec['family'] == 'iforest':
        return -model.score_samples(X)
    return model.predict_proba(X)[:, 1]
