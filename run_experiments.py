"""Run original EXP_001..EXP_017 using the modular research package."""
import argparse, json
from pathlib import Path
from fraud_project.config import EXPERIMENTS, EXPECTED_SHA256
from fraud_project.data import acquire_data, load_audit_data, temporal_split, sha256
from fraud_project.engine import fit_experiment, compare_results, source_versions

def main():
    p=argparse.ArgumentParser()
    p.add_argument('experiments', nargs='*', choices=list(EXPERIMENTS))
    p.add_argument('--dataset', default='data/creditcard.csv')
    p.add_argument('--results', default='results')
    p.add_argument('--list', action='store_true')
    p.add_argument('--no-download', action='store_true')
    a=p.parse_args()
    if a.list:
        for e,s in EXPERIMENTS.items(): print(e, s['name'])
        return
    ids=a.experiments or list(EXPERIMENTS)
    out=Path(a.results); out.mkdir(parents=True,exist_ok=True)
    f=acquire_data(a.dataset,try_download=not a.no_download)
    if sha256(f)!=EXPECTED_SHA256: raise RuntimeError('Dataset checksum mismatch. Original ULB CSV required.')
    df=load_audit_data(f,out)
    train,valid,test,info=temporal_split(df)
    (out/'split_protocol.json').write_text(json.dumps(info,indent=2))
    (out/'software_versions.json').write_text(json.dumps(source_versions(),indent=2))
    for exp in ids:
        _,m,_=fit_experiment(exp,train,valid,out)
        print(exp,'VAL AP:',round(m['average_precision'],6),'recall@0.5%:',round(m['recall_at_0.5pct'],6))
    board=compare_results(out,ids)
    print(board[['experiment_id','average_precision','recall_at_0.5pct']].to_string(index=False))
    print('Sealed original TEST NOT evaluated.')
if __name__=='__main__': main()
