"""EXP_018: reproduces historical paired error analysis on 4 validation prediction CSVs."""
import hashlib, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from .data import load_audit_data,temporal_split
MODELS = ("EXP_008", "EXP_011", "EXP_012", "EXP_016")
N_EXPECTED = 56746
POS_EXPECTED = 57
DATA_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
BUDGETS = (0.001, 0.005, 0.01)
ANALYSIS_SEED = 20261022

def load_predictions(folder: Path) -> pd.DataFrame:
    loaded = {}
    for model in MODELS:
        p = folder / f'{model}_validation_predictions.csv'
        if not p.is_file():
            raise FileNotFoundError(f'{p} not found. Copy the CSV into {folder}.')
        frame = pd.read_csv(p)
        assert list(frame.columns) == ['time', 'y_true', 'score'], (model, frame.columns.tolist())
        if len(frame) != N_EXPECTED:
            raise ValueError(f'{model}: expected {N_EXPECTED} rows but got {len(frame)}')
        if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f'{model}: missing or non-finite values')
        if not set(frame['y_true'].unique()).issubset({0,1}):
            raise ValueError(f'{model}: labels are not 0/1')
        loaded[model] = frame
    base = loaded[MODELS[0]].copy()
    if int(base['y_true'].sum()) != POS_EXPECTED:
        raise ValueError(f'Expected {POS_EXPECTED} positives, got {int(base.y_true.sum())}')
    if not base.time.is_monotonic_increasing:
        raise ValueError('Validation must be time sorted')
    for name in MODELS[1:]:
        f = loaded[name]
        if not np.array_equal(base.time.to_numpy(),f.time.to_numpy()) or not np.array_equal(base.y_true.to_numpy(), f.y_true.to_numpy()):
            raise ValueError(f'{name}: row alignment failed on ordered Time and labels')
    # A stable source transaction ID was not originally exported. This is
    # inferred row alignment, not cryptographic row-ID verification.
    return pd.DataFrame({'validation_row':np.arange(len(base)), 'time':base.time, 'y_true':base.y_true,
                         **{f'{name}_score':loaded[name]['score'] for name in MODELS}})

def add_ranks(df: pd.DataFrame):
    n=len(df)
    for name in MODELS:
        scores=df[f'{name}_score'].to_numpy()
        order=np.argsort(-scores,kind='mergesort')
        ranks=np.empty(n,dtype=int); ranks[order]=np.arange(1,n+1)
        df[f'{name}_rank']=ranks
        for fraction in BUDGETS:
            k=math.ceil(n*fraction)
            df[f'{name}_alert_{fraction:g}']=ranks<=k
    return df

def metrics(df: pd.DataFrame) -> pd.DataFrame:
    y=df.y_true.to_numpy()
    rows=[]
    for name in MODELS:
        score=df[f'{name}_score'].to_numpy()
        record={'experiment':name, 'n':len(df), 'fraud':int(y.sum()),
                'average_precision':average_precision_score(y,score),
                'roc_auc':roc_auc_score(y,score)}
        for frac in BUDGETS:
            selected=df[f'{name}_alert_{frac:g}'].to_numpy()
            hits=int(y[selected].sum());k=int(selected.sum())
            record[f'alerts_{frac:g}']=k
            record[f'fraud_caught_{frac:g}']=hits
            record[f'fraud_recall_{frac:g}']=hits/int(y.sum())
            record[f'false_alarms_{frac:g}']=k-hits
        rows.append(record)
    return pd.DataFrame(rows).sort_values('average_precision',ascending=False)

def pairwise(df):
    rows=[];y=df.y_true.to_numpy().astype(bool);f=0.005
    for i,a in enumerate(MODELS):
        for b in MODELS[i+1:]:
            aa=df[f'{a}_alert_{f:g}'].to_numpy();bb=df[f'{b}_alert_{f:g}'].to_numpy()
            rows.append({'model_a':a,'model_b':b,'common_alerts':int((aa&bb).sum()),
                         'a_only_alerts':int((aa&~bb).sum()),'b_only_alerts':int((bb&~aa).sum()),
                         'frauds_both':int((aa&bb&y).sum()),
                         'frauds_a_only':int((aa&~bb&y).sum()),
                         'frauds_b_only':int((bb&~aa&y).sum()),
                         'frauds_neither':int((~aa&~bb&y).sum())})
    return pd.DataFrame(rows)

def temporal_segments(df):
    rows=[]
    for label,segment in [('first_half',df.iloc[:len(df)//2]),('second_half',df.iloc[len(df)//2:])]:
        y=segment.y_true.to_numpy()
        for name in MODELS:
            rows.append({'period':label,'experiment':name,'n':len(segment),'fraud':int(y.sum()),
                         'fraud_prevalence':y.mean(),
                         'average_precision':average_precision_score(y,segment[f'{name}_score'])})
    return pd.DataFrame(rows)

def bootstrap_diffs(df, n_boot:int):
    """Stratified paired bootstrap preserving 57/56689 class counts.

    *Exploratory intervals*: ignores timestamp dependence, model-selection
    bias, and represents the empirical validation distribution only.
    """
    y=df.y_true.to_numpy()
    s_cat=df['EXP_012_score'].to_numpy()
    s_lgb=df['EXP_011_score'].to_numpy()
    rng=np.random.default_rng(ANALYSIS_SEED)
    positives=np.flatnonzero(y==1)
    negatives=np.flatnonzero(y==0)
    results=np.empty((n_boot,2))
    for b in range(n_boot):
        ix=np.concatenate([rng.choice(positives,len(positives),replace=True),
                           rng.choice(negatives,len(negatives),replace=True)])
        yy=y[ix];aa=s_cat[ix];bb=s_lgb[ix]
        results[b,0]=average_precision_score(yy,aa)-average_precision_score(yy,bb)
        k=math.ceil(len(yy)*0.005)
        h_cat=int(yy[np.argsort(-aa,kind='mergesort')[:k]].sum())
        h_lgb=int(yy[np.argsort(-bb,kind='mergesort')[:k]].sum())
        results[b,1]=(h_cat-h_lgb)/len(positives)
    return pd.DataFrame([
        {'comparison':'CatBoost minus weighted LightGBM',
         'metric':metric,'observed_difference':float(observed),
         'bootstrap_median':float(np.median(results[:,j])),
         'percentile_2_5':float(np.quantile(results[:,j],0.025)),
         'percentile_97_5':float(np.quantile(results[:,j],0.975)),
         'bootstrap_repetitions':n_boot,'seed':ANALYSIS_SEED,
         'caution':'Exploratory only: iid-stratified transaction assumption; ignores time dependence and repeated validation selection'}
        for j,(metric,observed) in enumerate([
            ('average_precision',average_precision_score(y,s_cat)-average_precision_score(y,s_lgb)),
            ('fraud_recall_at_0.5pct',(int(y[df['EXP_012_alert_0.005']].sum())-int(y[df['EXP_011_alert_0.005']].sum()))/len(positives))])])

def enrich_from_original(df: pd.DataFrame, dataset:Path):
    """Attach legitimate original features strictly to the reconstructed validation rows.

    The old pipeline drops EXACT duplicate rows, sorts by Time with stable mergesort,
    and takes strict time-quantile splits. Do not edit those choices here.
    """
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    h=hashlib.sha256()
    with dataset.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    if h.hexdigest()!=DATA_SHA256:
        raise ValueError('Dataset SHA256 differs from the exact file used in reported experiments; refusing row-level enrichment')
    raw=load_audit_data(dataset,output_dir=None)
    _,valid,_test,_=temporal_split(raw)
    if len(valid)!=len(df) or not np.array_equal(valid.Time.to_numpy(),df.time.to_numpy()) or not np.array_equal(valid.Class.to_numpy(),df.y_true.to_numpy()):
        raise ValueError('Cannot prove validation matches current source-split ordering; no Amount merge was made.')
    for c in ('Amount','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28'):
        df[c]=valid[c].to_numpy()
    return df

def write_figure(df,out:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.6))
    for name in MODELS:
        xs=[100*x for x in BUDGETS]
        ys=[df.loc[df[f'{name}_alert_{x:g}'],'y_true'].sum()/df.y_true.sum() for x in BUDGETS]
        ax.plot(xs,ys,marker='o',label=name)
    ax.set_xlabel('Fraction of transactions reviewed (%)')
    ax.set_ylabel('Fraction of fraudulent transactions detected')
    ax.set_ylim(0,1)
    ax.legend(frameon=False)
    ax.set_title('Validation recall at fixed transaction-review budgets')
    fig.tight_layout(); fig.savefig(out/'budget_vs_recall.png',dpi=170);plt.close(fig)

def run(inputs:Path,out:Path,dataset:Path|None=None,n_boot:int=250):
    out.mkdir(parents=True,exist_ok=True)
    df=add_ranks(load_predictions(inputs))
    if dataset is not None:
        df=enrich_from_original(df,dataset)
    met=metrics(df); pairs=pairwise(df); periods=temporal_segments(df)
    positives=df[df.y_true.eq(1)].copy()
    alert_cols=[f'{m}_alert_0.005' for m in MODELS]
    positives['models_alerting']=positives[alert_cols].sum(axis=1)
    positives['missed_by_all']=positives['models_alerting'].eq(0)
    missed=positives[positives.missed_by_all].copy()
    met.to_csv(out/'validation_metrics_verified.csv',index=False)
    pairs.to_csv(out/'pairwise_alert_overlap.csv',index=False)
    periods.to_csv(out/'temporal_segments.csv',index=False)
    positives.to_csv(out/'all_57_fraud_cases.csv',index=False)
    missed.to_csv(out/'eight_missed_fraud_cases.csv',index=False)
    bootstrap=bootstrap_diffs(df,n_boot)
    bootstrap.to_csv(out/'paired_bootstrap_exploratory.csv',index=False)
    write_figure(df,out)
    report=["# EXP_018: Paired failure analysis (VALIDATION ONLY)","",
       "Do not interpret these results as independent confirmation: all models were selected/compared using the same validation period.","",
       "## Verified model performance", "", met.to_markdown(index=False),"",
       "## Pairwise top-0.5% alert overlap", "",pairs.to_markdown(index=False),"",
       f"At 0.5% capacity (284 alerts), **{len(missed)} frauds** were missed by all four models.","",
       "## Eight missed frauds", "",
       missed[['validation_row','time',*[f'{m}_rank' for m in MODELS]] + (['Amount'] if 'Amount' in df.columns else [])].to_markdown(index=False),"",
       "## Two within-validation temporal segments","",periods.to_markdown(index=False),"",
       "## Exploratory stratified paired bootstrap","",bootstrap.drop(columns=['caution']).to_markdown(index=False),"",
       "These percentile intervals are descriptive only and do not account for within-time dependence or the repeated model selection on validation.","",
       "## Interpretation limits", "",
       "- Predictions do not contain a stable originating row ID; consistent ordered time and class sequences support alignment but do not cryptographically prove it.",
       "- The original dataset's PCA-transformed V1–V28 lack feature semantics, so do not invent behavioural explanations for individual missed frauds.",
       "- This work reads no final TEST predictions or TEST evaluation labels. Original CSV enrichment reconstructs only the validation partition.",
       "- The same review capacity forces false alarms = capacity - true alerts, so it is not independent evidence of financial savings.",
       "- For real temporal replication and delay-aware monitoring, a longer dataset with appropriate time features is required.",""]
    (out/'EXP_018_FINDINGS.md').write_text('\n'.join(report),encoding='utf-8')
    print(f"Paired analysis completed: {len(df):,} validation rows, {len(positives)} frauds, {len(missed)} all-model misses")
    print(met[['experiment','average_precision','fraud_caught_0.005']].to_string(index=False))
    print(f"Files written to {out.resolve()}")
    return met,pairs,missed
