"""Plan or launch the corrected two-stage ResNet50 experiment.

The fixed sequence is frozen-head CV, late-layer fine-tuning CV, frozen-head
final fit, late-layer fine-tuning final fit, and one locked-holdout evaluation.
No holdout result is used for model or hyperparameter selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def command(stage, output_root, splits):
    pipeline = str(ROOT / 'code' / 'training_pipeline.py')
    common = [sys.executable, '-u', pipeline]
    head_cv = output_root / 'resnet50_head_cv'
    fine_cv = output_root / 'resnet50_finetune_cv'
    head_final = output_root / 'resnet50_head_final'
    fine_final = output_root / 'resnet50_finetune_final'
    settings = [
        '--model', 'resnet50', '--splits', str(splits),
        '--image-size', '224', '--batch-size', '2', '--weights', 'imagenet',
        '--training-profile', 'reviewed', '--preprocessing-profile', 'resnet50_imagenet',
        '--augmentation-profile', 'scalogram-image', '--augment',
        '--dropout', '0.3', '--l2', '0.0001', '--patience', '8',
        '--resize-interpolation', 'nearest', '--cpu-threads', '4',
        '--data-threads', '1', '--prefetch-batches', '1',
        '--min-free-ram-gb', '1.5', '--min-free-disk-gb', '5',
        '--lr-reduce-factor', '0.2', '--lr-min', '1e-7',
        '--no-legacy-h5-export',
    ]
    if stage == 'head_cv':
        return common + ['cv', '--output-dir', str(head_cv), '--epochs', '25',
                         '--learning-rate', '0.001', '--freeze-backbone'] + settings
    if stage == 'fine_cv':
        return common + ['cv', '--output-dir', str(fine_cv), '--epochs', '35',
                         '--learning-rate', '1e-5', '--backbone-trainable-layers', '33',
                         '--initial-cv-run', str(head_cv)] + settings
    if stage == 'head_final':
        return common + ['final-fit', '--output-dir', str(head_final), '--epochs', '25',
                         '--learning-rate', '0.001', '--freeze-backbone'] + settings
    if stage == 'fine_final':
        return common + ['final-fit', '--output-dir', str(fine_final), '--epochs', '35',
                         '--learning-rate', '1e-5', '--backbone-trainable-layers', '33',
                         '--initial-final-run', str(head_final)] + settings
    if stage == 'evaluate':
        return common + ['evaluate', '--run-dir', str(fine_final),
                         '--splits', str(splits), '--bootstrap', '2000']
    raise ValueError(stage)


def complete(stage, output_root):
    directory = output_root / ('resnet50_finetune_final/holdout_evaluation' if stage == 'evaluate'
                               else {'head_cv':'resnet50_head_cv',
                                     'fine_cv':'resnet50_finetune_cv',
                                     'head_final':'resnet50_head_final',
                                     'fine_final':'resnet50_finetune_final'}[stage])
    status = directory / 'status.json'
    return status.exists() and json.loads(status.read_text()).get('status') == 'COMPLETE'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--splits', default=str(ROOT / 'splits' / 'run01' / 'splits.csv'))
    parser.add_argument('--launch', action='store_true',
                        help='Run all stages sequentially; otherwise only print the commands.')
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    splits = Path(args.splits).resolve()
    stages = ['head_cv', 'fine_cv', 'head_final', 'fine_final', 'evaluate']
    plan = {stage: command(stage, output_root, splits) for stage in stages}
    output_root.mkdir(parents=True, exist_ok=True)
    plan_file = output_root / 'corrected_resnet50_plan.json'
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError('Existing corrected ResNet50 plan differs; use a new output directory')
    if not plan_file.exists():
        plan_file.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for stage in stages:
        print(stage + ':', subprocess.list2cmdline(plan[stage]), flush=True)
        if args.launch and not complete(stage, output_root):
            subprocess.run(plan[stage], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
