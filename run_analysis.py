"""Reproduce EXP_018 on validation-only predictions. Historical inputs can be used."""
import argparse
from pathlib import Path
from fraud_project.analysis import run, MODELS

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pred-dir', type=Path, default=None)
    p.add_argument('--dataset', type=Path, default=None)
    p.add_argument('--out',type=Path,default=Path('analysis_output'))
    p.add_argument('--bootstrap',type=int,default=500)
    args=p.parse_args()
    if args.bootstrap<20: p.error('At least 20 bootstrap iterations required')
    default=Path('results')
    if not all((default/f'{i}_validation_predictions.csv').exists() for i in MODELS):
        default=Path('historical_predictions')
    run(args.pred_dir or default,args.out,args.dataset,args.bootstrap)
if __name__=='__main__':main()
