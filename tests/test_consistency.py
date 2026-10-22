from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from fraud_project.config import EXPERIMENTS,FEATURE_NAMES
from fraud_project.data import temporal_split,make_features
from fraud_project.models import build_model
from fraud_project.metrics import ranking_metrics,top_budget_metrics
from fraud_project.analysis import load_predictions,add_ranks,metrics,pairwise,MODELS

ROOT=Path(__file__).resolve().parents[1]

def test_experiment_registry():
    assert set(EXPERIMENTS)=={f'EXP_{i:03d}' for i in range(1,18)}
    assert EXPERIMENTS['EXP_012']['family']=='cat'

def test_features_and_split():
    rng=np.random.default_rng(12)
    n=250
    df=pd.DataFrame({f'V{i}':rng.normal(size=n) for i in range(1,29)})
    df['Amount']=rng.uniform(0,100,size=n)
    df['Time']=np.arange(n).astype(float)
    df['Class']=np.arange(n)%7==0
    train,valid,test,_=temporal_split(df)
    assert len(train)+len(valid)+len(test)==n
    assert train.Time.max()<valid.Time.min()<test.Time.min()
    f=make_features(train)
    assert f.columns.tolist()==FEATURE_NAMES and f.shape[1]==31

def test_original_predictions_reproduced():
    df=add_ranks(load_predictions(ROOT/'historical_predictions'))
    table=metrics(df).set_index('experiment')
    expected={'EXP_008':0.773451,'EXP_011':0.781462,'EXP_012':0.786158,'EXP_016':0.778371}
    for name,ap in expected.items():
        assert abs(table.loc[name,'average_precision']-ap)<1e-5
    y=df.y_true.to_numpy()
    all_missed=(y==1)
    for name in MODELS: all_missed &= ~df[f'{name}_alert_0.005'].to_numpy()
    assert int(all_missed.sum())==8
    assert int(table.loc['EXP_011','fraud_caught_0.005'])==49
