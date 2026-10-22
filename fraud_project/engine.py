"""Run a registered experiment, compare metrics, optionally evaluate sealed test once."""
import json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from .config import SEED, FEATURE_NAMES, EXPERIMENTS
from .data import make_features
from .models import build_model, scores_for_model
from .metrics import choose_threshold, ranking_metrics, threshold_metrics

def fit_experiment(exp_id, train, valid, results_dir):
    if exp_id not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {exp_id}. Options: {list(EXPERIMENTS)}')
    results_dir=Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    spec=EXPERIMENTS[exp_id]
    X_train=make_features(train); y_train=train.Class.to_numpy(dtype=int)
    X_valid=make_features(valid); y_valid=valid.Class.to_numpy(dtype=int)
    model=build_model(spec)
    t0=time.perf_counter()
    train_count=len(y_train)
    if spec['family']=='iforest':
        model.fit(X_train.loc[y_train==0])
    else:
        if spec['smote']:
            from imblearn.over_sampling import SMOTE
            if y_train.sum() < 6:
                raise ValueError('At least 6 fraud examples required for default SMOTE.')
            # Important: only training data are augmented, never validation/test.
            X_train, y_train = SMOTE(sampling_strategy=0.10, random_state=SEED).fit_resample(X_train, y_train)
        fit_kwargs={}
        if spec['family'] in ('hgb','xgb','lgbm','cat','ada') and spec['balanced']:
            fit_kwargs['sample_weight']=compute_sample_weight(class_weight='balanced',y=y_train)
        model.fit(X_train,y_train,**fit_kwargs)
    fit_sec=time.perf_counter()-t0
    scores=scores_for_model(model,X_valid,spec)
    threshold, val_best_f2=choose_threshold(y_valid,scores)
    metrics={'experiment_id':exp_id,'name':spec['name'],'hypothesis':spec['hypothesis'],
             'seed':SEED,'n_training_original':train_count,'n_training_fitted':len(y_train),
             'n_validation':len(y_valid),'fit_seconds':fit_sec,
             'threshold_selected_on':'validation F2 only',
             **ranking_metrics(y_valid,scores),**threshold_metrics(y_valid,scores,threshold)}
    (results_dir/f'{exp_id}_valid_metrics.json').write_text(json.dumps(metrics,indent=2))
    # Fixed parameters are logged independently. This log contains no held-out
    # observations or hidden test labels.
    (results_dir/f'{exp_id}_model_config.json').write_text(json.dumps({
        'experiment_id': exp_id, 'spec': spec, 'model_params': model.get_params(deep=True),
        'feature_names': FEATURE_NAMES, 'seed': SEED,
    }, indent=2, default=str))
    pd.DataFrame({'time':valid.Time.to_numpy(),'y_true':y_valid,'score':scores}).to_csv(
        results_dir/f'{exp_id}_validation_predictions.csv',index=False)
    return model,metrics,scores

def compare_results(results_dir, experiment_ids):
    files=[Path(results_dir)/f'{i}_valid_metrics.json' for i in experiment_ids]
    records=[json.loads(f.read_text()) for f in files if f.exists()]
    if not records:
        raise FileNotFoundError('No completed experiment results found.')
    df=pd.DataFrame(records).sort_values('average_precision',ascending=False)
    df.to_csv(Path(results_dir)/'validation_leaderboard.csv',index=False)
    return df

def final_test_once(exp_id, train, valid, test, results_dir, *, confirm_unseal=False):
    """Evaluate sealed TEST exactly once with preselected model and VAL threshold.

    No model selection on TEST, and no refit on TRAIN+VALID to avoid changing
    a configuration whose decision threshold was tuned on VAL.
    """
    if not confirm_unseal:
        raise PermissionError('Set confirm_unseal=True ONLY after final model and threshold are locked.')
    r=Path(results_dir)
    final_path=r/'FINAL_TEST_LOCKED.json'
    if final_path.exists():
        raise FileExistsError(f'Final test was already used: {final_path}. Do not run repeated test-driven selection.')
    mfile=r/f'{exp_id}_valid_metrics.json'
    pfile=r/f'{exp_id}_validation_predictions.csv'
    if not mfile.exists() or not pfile.exists():
        raise FileNotFoundError('Run/lock the selected validation experiment first.')
    locked=json.loads(mfile.read_text())
    cutoff=float(locked['threshold'])
    if len(pd.read_csv(pfile)) != len(valid):
        raise ValueError('Validation prediction length mismatch: do not unseal test.')
    spec=EXPERIMENTS[exp_id]
    model=build_model(spec)
    X=make_features(train); y=train.Class.to_numpy(dtype=int)
    if spec['family']=='iforest':
        model.fit(X.loc[y==0])
    else:
        if spec['smote']:
            from imblearn.over_sampling import SMOTE
            X,y=SMOTE(sampling_strategy=0.10,random_state=SEED).fit_resample(X,y)
        kwargs={}
        if spec['family'] in ('hgb','xgb','lgbm','cat','ada') and spec['balanced']:
            kwargs['sample_weight']=compute_sample_weight('balanced',y=y)
        model.fit(X,y,**kwargs)
    scores=scores_for_model(model,make_features(test),spec)
    res={'locked_experiment':exp_id,'selection_rule':'highest VAL average precision; threshold locked on VAL F2',
         'original_validation_ap':locked['average_precision'],'test_n':len(test),
         'test_fraud_n':int(test.Class.sum()),'prevalence':float(test.Class.mean()),
         **ranking_metrics(test.Class,scores),**threshold_metrics(test.Class,scores,cutoff)}
    # Atomic output marking test consumed, no selection allowed afterward.
    final_path.write_text(json.dumps(res,indent=2))
    pd.DataFrame({'time':test.Time.to_numpy(),'y_true':test.Class.to_numpy(),
                  'score':scores}).to_csv(r/'FINAL_TEST_predictions.csv',index=False)
    return res

def source_versions():
    import sklearn, scipy, imblearn
    from importlib.metadata import version, PackageNotFoundError
    optional = {}
    for pkg in ('xgboost', 'lightgbm', 'catboost'):
        try:
            optional[pkg] = version(pkg)
        except PackageNotFoundError:
            optional[pkg] = 'not installed'
    return {'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
            'sklearn':sklearn.__version__,'scipy':scipy.__version__,'imbalanced_learn':imblearn.__version__,
            **optional, 'seed':SEED}
