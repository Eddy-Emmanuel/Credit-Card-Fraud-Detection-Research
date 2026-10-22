"""Ranking, fixed-alert-budget, and threshold evaluation utilities."""
import math
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
 recall_score, f1_score, fbeta_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score)

def top_budget_metrics(y, scores, fractions=(0.001, 0.005, 0.01)):
    y=np.asarray(y, dtype=np.int64); scores=np.asarray(scores)
    out={}
    order=np.argsort(-scores, kind='mergesort')
    n_fraud=int(y.sum())
    for frac in fractions:
        k=max(1, int(math.ceil(len(y)*frac)))
        hits=int(y[order[:k]].sum())
        label=f'{100*frac:g}pct'
        out[f'precision_at_{label}']=float(hits/k)
        out[f'recall_at_{label}']=float(hits/n_fraud) if n_fraud else float('nan')
        out[f'alerts_at_{label}']=k
    return out

def ranking_metrics(y, scores):
    y=np.asarray(y,dtype=int); scores=np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite predicted scores.')
    if len(np.unique(y)) != 2:
        raise ValueError('Cannot evaluate ranking with only one class.')
    return {'average_precision':float(average_precision_score(y, scores)),
            'roc_auc':float(roc_auc_score(y, scores)),
            **top_budget_metrics(y,scores)}

def choose_threshold(y, scores, beta=2.0):
    """Select F2 maximizing threshold on VALIDATION only (no test labels)."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds=precision_recall_curve(y, scores)
    if thresholds.size == 0:
        raise ValueError('No thresholds available.')
    numerator=(1+beta**2)*precision[:-1]*recall[:-1]
    denominator=(beta**2)*precision[:-1]+recall[:-1]
    f=np.divide(numerator,denominator,out=np.zeros_like(numerator), where=denominator>0)
    ix=int(np.argmax(f))
    return float(thresholds[ix]),float(f[ix])

def threshold_metrics(y, scores, threshold):
    y=np.asarray(y,dtype=int); pred=(np.asarray(scores)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()
    return {'threshold':float(threshold),'alerts':int(pred.sum()),
            'alert_rate':float(pred.mean()),'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'f2':float(fbeta_score(y,pred,beta=2,zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'mcc':float(matthews_corrcoef(y,pred)),
            'tp':int(tp),'tn':int(tn),'fp':int(fp),'fn':int(fn),
            'false_alarms_per_1000_legit':float(fp/max(1,fp+tn)*1000)}
