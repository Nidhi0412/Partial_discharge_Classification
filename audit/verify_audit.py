"""Read-only research checks; writes derived evidence only into this audit directory."""
import ast
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev, pstdev

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

def write_csv(name, rows, fields):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

inventory = []
syntax = []
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '.git' in p.parts or 'audit' in p.relative_to(ROOT).parts:
        continue
    inventory.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                          sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    if p.suffix == '.py':
        try:
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
            syntax.append(dict(file=str(p.relative_to(ROOT)), status='PASS', line='', message=''))
        except SyntaxError as e:
            syntax.append(dict(file=str(p.relative_to(ROOT)), status='FAIL', line=e.lineno, message=e.msg))
write_csv('SOURCE_INVENTORY.csv', inventory, ['path', 'bytes', 'sha256'])
write_csv('STATIC_SYNTAX_CHECKS.csv', syntax, ['file', 'status', 'line', 'message'])

rows = []
def record(model, metric, claim, saved, calc, status, reason):
    rows.append(dict(model=model, metric=metric, manuscript_or_repository_value=claim,
                     saved_value=saved, independently_recalculated_value=calc,
                     verification_status=status, reason=reason))

# Reconstruct counts from rounded VGG16 report recalls and explicitly stated supports.
# This is a consistency calculation, not an independent prediction-level verification.
matrices = []
series = {}
for model in ['VGG16', 'ResNet50']:
    series[model] = {k: [] for k in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'macro_precision', 'macro_recall', 'macro_f1']}
    for fold in range(2, 6):
        if model == 'VGG16':
            tn, fp, fn, tp = (160, 0, 5 if fold == 2 else 6, 155 if fold == 2 else 154)
            reason = 'Conditional reconstruction from support=160 per class and rounded recalls; original predictions absent'
        else:
            saved = list(csv.DictReader((ROOT / 'results/ResNet50/crossval_metrics.csv').open()))[fold-2]
            # Conditional on the same balanced 320-image test set; not independently observed counts.
            tp = round(float(saved['recall']) * 160)
            fn = 160 - tp
            correct = round(float(saved['accuracy']) * 320)
            tn, fp = correct-tp, 160-(correct-tp)
            reason = 'Conditional reconstruction from saved precision/recall/accuracy and assumed balanced 320-image test set'
            assert abs(tp/(tp+fp)-float(saved['precision'])) < 1e-12
        acc = (tn+tp)/320
        prec, rec, spec = tp/(tp+fp), tp/(tp+fn), tn/(tn+fp)
        f1 = 2*tp/(2*tp+fp+fn)
        negp, negr = tn/(tn+fn), spec
        negf = 2*tn/(2*tn+fp+fn)
        vals = dict(accuracy=acc, precision=prec, recall=rec, specificity=spec, f1=f1,
                    macro_precision=(prec+negp)/2, macro_recall=(rec+negr)/2, macro_f1=(f1+negf)/2)
        matrices.append(dict(model=model, fold=fold, TN=tn, FP=fp, FN=fn, TP=tp, basis=reason))
        for metric, value in vals.items():
            series[model][metric].append(value)
            record(model, f'fold_{fold}_{metric}', '', '', value, 'CONDITIONAL_REPORT_CONSISTENCY_ONLY', reason)
    for metric, values in series[model].items():
        record(model, f'mean_{metric}', {'accuracy': .982 if model=='VGG16' else .9703}.get(metric,''), '', mean(values),
               'SUMMARY_ARITHMETIC_ONLY', 'Equal-weight four-run mean; positive class=Teflon; not grouped CV validation')
        record(model, f'sample_sd_{metric}', '', '', stdev(values), 'SUMMARY_ARITHMETIC_ONLY', 'ddof=1, four dependent runs')
write_csv('CONDITIONAL_CONFUSION_MATRICES.csv', matrices, ['model','fold','TN','FP','FN','TP','basis'])

aucs = list(csv.DictReader((ROOT/'results/auc_foldwise_summary.csv').open()))
stats = json.loads((ROOT/'results/statistical_analysis_results.json').read_text())
for model in ['VGG16','ResNet50']:
    values = [float(r['AUC']) for r in aucs if r['Model']==model]
    for r in aucs:
        if r['Model']==model:
            record(model, 'fold_'+r['Fold']+'_auc', '', r['AUC'], '', 'BLOCKED_PREDICTIONS_MISSING', 'Cannot reconstruct ROC from AUC summary or raster curve')
    for metric, arr in [('auc',values),('accuracy',series[model]['accuracy'])]:
        entry = stats[model.lower()+'_'+metric]
        for label, calc in [('mean',mean(arr)),('std',stdev(arr))]:
            record(model, metric+'_'+label+'_vs_statistics_json', '', entry[label], calc, 'SUMMARY_COMPARISON', 'Exact CSV AUCs / conditional report counts; sample SD, ddof=1')
        margin=entry['t_critical']*entry['std']/(entry['n_folds']**.5)
        record(model, metric+'_ci_margin_formula', '', entry['margin_error'], margin,
               'FORMULA_REPRODUCED_INFERENCE_UNJUSTIFIED', 't(3)*saved SD/sqrt(4); repeated fixed-test results are not independent test samples')
    record(model, 'mean_auc', .9981 if model=='VGG16' else .9966, mean(values), mean(values),
           'SUMMARY_ARITHMETIC_ONLY', 'Mean of four saved AUCs; original scores absent')
for name in ['accuracy_t_test','auc_t_test','accuracy_wilcoxon','auc_wilcoxon']:
    record('VGG16_vs_ResNet50',name+'_p', '',stats[name]['p_value'],'','NOT_INDEPENDENTLY_VERIFIED', 'Statistics-generating script and exact input arrays absent; shared test observations; Wilcoxon JSON p=0.125')
for name, value in [('original_recordings',400),('images',1600),('development',1280),('training',1024),('validation',256),('test',320)]:
    record('dataset',name,value,'dataset_size_report.txt' if name in ['training','validation','test'] else '', '',
           'UNVERIFIED_FILE_COUNTS', 'Raw recordings and classified image partitions not present in audited repository')
write_csv('RESULT_VERIFICATION.csv', rows, list(rows[0]))

fields=['original_recording_id','material','electrode_configuration','acquisition_conditions','wavelet','true_dwt_level','output_filename','partition','provenance_status','evidence']
write_csv('DATASET_PROVENANCE.csv', [dict.fromkeys(fields,'') | dict(provenance_status='UNAVAILABLE_NO_SAMPLE_RECORDS', evidence='No raw recordings, classified scalograms, or recording-to-image mapping in supplied minimal repository; this is a missing-evidence marker, not a sample')], fields)
write_csv('SPLIT_MANIFEST_TEMPLATE.csv', [], ['original_recording_id','material','electrode_configuration','acquisition_session','image_path','image_sha256','outer_fold','partition'])
print(json.dumps({'inventory_files':len(inventory),'syntax':syntax,
                  'vgg_accuracy_mean':mean(series['VGG16']['accuracy']),
                  'resnet_accuracy_mean':mean(series['ResNet50']['accuracy']),
                  'auc_means':{m:mean([float(r['AUC']) for r in aucs if r['Model']==m]) for m in series}},indent=2))
