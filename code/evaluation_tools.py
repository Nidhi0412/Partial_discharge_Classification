"""Evaluation, traced prediction summaries and ROC plots for the shared pipeline."""
import argparse
import csv
import json
import re
from pathlib import Path

from dataset_manifest import CLASS_NAMES, digest, write_csv
from training_pipeline import evaluate_run, prediction_summary, save_json


def read_predictions(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream: rows=list(csv.DictReader(stream))
    if not rows: raise ValueError('No predictions')
    ids=set();splits=set();models=set()
    for r in rows:
        for key in ['image_id','recording_id','fold','checkpoint_sha256','split_sha256','model_type']:
            if not r.get(key): raise ValueError(f'Missing prediction identity {key}')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',r['image_id']):raise ValueError('Invalid image identifier')
        if r['image_id'] in ids: raise ValueError('Repeated image predictions cannot be pooled as independent observations')
        ids.add(r['image_id']);splits.add(r['split_sha256']);models.add(r['model_type'])
        if int(r['y_true'])!=CLASS_NAMES.index(r['material']): raise ValueError('True label order is inconsistent')
    if len(splits)!=1 or len(models)!=1: raise ValueError('Predictions mix experiments/models')
    return rows


def roc_points(rows):
    scores=sorted([(float(r['p_teflon']),int(r['y_true'])) for r in rows],reverse=True)
    positives=sum(y for _,y in scores);negatives=len(scores)-positives
    if not positives or not negatives: raise ValueError('ROC requires both material classes')
    points=[(0.,0.)];tp=fp=0;i=0
    while i<len(scores):
        j=i
        while j<len(scores) and scores[j][0]==scores[i][0]:
            tp+=scores[j][1];fp+=1-scores[j][1];j+=1
        points.append((fp/negatives,tp/positives));i=j
    return points


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('evaluate');p.add_argument('--run-dir',required=True);p.add_argument('--splits',required=True)
    p.add_argument('--bootstrap',type=int,default=1000)
    p=sub.add_parser('summarize');p.add_argument('--predictions',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--bootstrap',type=int,default=0);p.add_argument('--group-by',choices=['recording_id','session_id','group_id'],default='group_id')
    p=sub.add_parser('roc');p.add_argument('--predictions',required=True);p.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    if getattr(args,'bootstrap',0)<0:parser.error('bootstrap must be nonnegative')
    if args.command=='evaluate':return evaluate_run(args)
    rows=read_predictions(args.predictions)
    metrics,records=prediction_summary(rows,bootstrap=getattr(args,'bootstrap',0),group_by=getattr(args,'group_by','recording_id'))
    if len({r['fold'] for r in rows})>1 and any(r['partition']!='validation' for r in rows):
        raise ValueError('Only disjoint development validation folds may be summarized together')
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    save_json(out/'metrics.json',metrics)
    save_json(out/'source.json',dict(predictions=str(Path(args.predictions).resolve()),sha256=digest(args.predictions),
                                    class_names=list(CLASS_NAMES),note='Development validation is not unbiased nested-CV test performance'))
    write_csv(out/'recording_predictions.csv',records,list(records[0]))
    if args.command=='roc':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        points=roc_points(rows)
        fig,axis=plt.subplots(figsize=(6,5))
        axis.plot([x for x,y in points],[y for x,y in points],label=f'Image AUC = {metrics["image"]["auc"]:.4f}')
        axis.plot([0,1],[0,1],'--',color='gray')
        endpoint='Development validation' if all(r['partition']=='validation' for r in rows) else 'Locked holdout'
        axis.set(title=f'{endpoint} ROC: {rows[0]["model_type"]}',xlabel='False positive rate',ylabel='True positive rate')
        axis.legend();fig.tight_layout();fig.savefig(out/'roc.png',dpi=150);plt.close(fig)
        write_csv(out/'roc_points.csv',[dict(false_positive_rate=x,true_positive_rate=y) for x,y in points],['false_positive_rate','true_positive_rate'])
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
