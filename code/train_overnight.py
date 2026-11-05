"""Run both backbones sequentially with fixed settings, logs and safe recovery.

Queue: both development CV runs, both final fits, then one locked-holdout
evaluation per model. No model/settings selection uses the holdout.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dataset_manifest import digest, load_splits
from prepare_imagenet_weights import prepare
from training_safety import atomic_json, check_resources, memory_available, training_lock
from legacy_augmentation import HISTORICAL_PARAMETERS
from scalogram_augmentation import PRIMARY_DEFAULTS,operation_order,parameters

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(epochs=50, batch_size=2, image_size=224, learning_rate=1e-5,
                dropout=.5, l2=.01, patience=10, seed=42, weights='imagenet',
                cpu_threads=4, data_threads=1, prefetch_batches=1,
                min_free_ram_gb=1.5, min_free_disk_gb=5., augment=True,
                fine_tuning=True,training_profile='historical',augmentation_profile='scalogram-image',
                resize_interpolation='nearest',augmentation_samples_per_class=5,
                legacy_h5_export=True,lr_reduce_factor=.1,lr_min=1e-7,**PRIMARY_DEFAULTS)


def now(): return datetime.now(timezone.utc).isoformat()


def validate_primary_plan(plan):
    settings=plan['settings']
    if (settings.get('augment') is not True or settings.get('fine_tuning') is not True or
        settings.get('augmentation_profile')!='scalogram-image' or settings.get('image_stretch_fraction')!=0.):
        raise ValueError('Unattended primary training requires fine-tuning, augmentation ON, scalogram-image and no vertical stretching. Use separate explicit runs for historical ablations.')


def make_plan(batch):
    jobs = []
    for command in ['cv', 'final-fit', 'evaluate']:
        for model in ['vgg16', 'resnet50']:
            run = batch/f'{model}_{"cv" if command=="cv" else "final"}'
            jobs.append(dict(name=f'{model}_{command}', command=command,
                             model=model, run_dir=str(run)))
    plan=dict(created_utc=now(), settings=dict(SETTINGS),
                augmentation_parameters=dict(historical=None,specialized=parameters(SETTINGS),
                    operation_order=operation_order(SETTINGS),
                    interpretation='Composed PNG perturbations. No physical MHz, phase or waveform-noise claims.'),
                splits=str(ROOT/'splits/run01/splits.csv'), jobs=jobs,
                policy='Fixed settings; all development CV precedes final fits; holdout evaluated once per model.',
                source_hashes={name:digest(ROOT/'code'/name) for name in
                               ['training_pipeline.py','training_safety.py','dataset_manifest.py',
                                'train_overnight.py','prepare_imagenet_weights.py',
                                'legacy_augmentation.py','scalogram_augmentation.py','training_artifacts.py']})
    validate_primary_plan(plan)
    return plan


def validate_saved_run(job,plan):
    path=Path(job['run_dir'])/'run_config.json'
    if not path.exists():return
    saved=json.loads(path.read_text())
    expected_purpose='development_cv' if job['command']=='cv' else 'final_fit'
    if saved.get('model_type')!=job['model'] or saved.get('purpose')!=expected_purpose:
        raise ValueError('Existing run model/purpose does not match this primary queue; choose a fresh output directory.')
    for key,value in plan['settings'].items():
        expected=None if key=='weights' and value=='none' else value
        if saved.get(key)!=expected:
            raise ValueError(f'Existing run {key} differs from the primary plan; do not mix experiments.')


def command_for(job, plan):
    run = Path(job['run_dir'])
    base = [sys.executable, '-u', str(ROOT/'code/training_pipeline.py')]
    if job['command']=='evaluate':
        return base+['evaluate','--run-dir',str(run),'--splits',plan['splits'],'--bootstrap','1000']
    if (run/'run_config.json').exists():
        validate_saved_run(job,plan)
        return base+['resume','--run-dir',str(run)]
    command = base+[job['command'],'--model',job['model'],'--splits',plan['splits'],
                    '--output-dir',str(run)]
    for key, value in plan['settings'].items():
        if key in {'augment','legacy_h5_export'}:
            command.append('--'+('' if value else 'no-')+key.replace('_','-'))
        elif key=='fine_tuning':
            if not value: command.append('--freeze-backbone')
        else:
            command.extend(['--'+key.replace('_','-'),str(value)])
    return command


def completed(job):
    run = Path(job['run_dir'])
    if job['command']=='evaluate': run = run/'holdout_evaluation'
    path = run/'status.json'
    return path.exists() and json.loads(path.read_text())['status']=='COMPLETE'


def supervise(batch):
    batch.mkdir(parents=True,exist_ok=True)
    if (batch/'stopped_by_user.json').exists():
        raise RuntimeError('This queue was stopped by the user. Preserve it; choose a NEW output directory for the corrected experiment.')
    plan_file=batch/'plan.json'
    if plan_file.exists():
        plan=json.loads(plan_file.read_text())
    else:
        plan=make_plan(batch)
        with plan_file.open('x',encoding='utf-8') as stream: json.dump(plan,stream,indent=2)
    validate_primary_plan(plan)
    for name,expected in plan['source_hashes'].items():
        if digest(ROOT/'code'/name)!=expected: raise RuntimeError(f'Source changed since queue creation: {name}')
    def status(state,**extra):
        atomic_json(batch/'queue_status.json',dict(status=state,updated_utc=now(),
                    supervisor_pid=os.getpid(),**extra))
    if os.name=='nt':
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError('Could not request that Windows stay awake during training')
    try:
        status('PREFLIGHT')
        weights=prepare()
        rows,split=load_splits(plan['splits'])
        atomic_json(batch/'preflight.json',dict(weights=weights,split_sha256=split['split_sha256'],
                    manifest_rows=len(rows),settings=plan['settings'],resources=check_resources(SETTINGS,batch)))
        for job in plan['jobs']:
            validate_saved_run(job,plan)
            if completed(job): continue
            for attempt in range(1,7):
                # Wait for useful headroom before importing TensorFlow in the child.
                waiting_since=time.monotonic()
                while memory_available()<3.0:
                    status('WAITING_FOR_RAM',job=job['name'],attempt=attempt,
                           available_ram_gib=memory_available(),required_to_launch_gib=3.0)
                    if time.monotonic()-waiting_since>12*3600:
                        raise RuntimeError('Insufficient startup RAM for 12 hours; saved work retained')
                    time.sleep(30)
                command=command_for(job,plan)
                logfile=batch/f"{job['name']}_attempt_{attempt}_{int(time.time())}.log"
                print(f"{now()} Starting {job['name']}; log {logfile}",flush=True)
                with logfile.open('x',encoding='utf-8') as log:
                    process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL,env=os.environ.copy())
                    while process.poll() is None:
                        status('RUNNING',job=job['name'],attempt=attempt,child_pid=process.pid,
                               log=str(logfile),command=command,available_ram_gib=memory_available())
                        time.sleep(5)
                if process.returncode==0 and completed(job):
                    print(f"{now()} Completed {job['name']}",flush=True)
                    break
                failure_file=Path(job['run_dir'])/'failure.json'
                failure=json.loads(failure_file.read_text()) if failure_file.exists() else {}
                # Never retry/re-evaluate a failed holdout automatically, or hide code errors.
                retryable=(job['command']!='evaluate' and
                           failure.get('error_type') in {'ResourceLimitError','ResourceExhaustedError','MemoryError'})
                if not retryable or attempt==6:
                    raise RuntimeError(f"{job['name']} failed (exit {process.returncode}); inspect {logfile}. {failure}")
                status('RESOURCE_RETRY',job=job['name'],attempt=attempt,log=str(logfile),failure=failure)
                time.sleep(30)
        summary={model:json.loads((batch/f'{model}_final/holdout_evaluation/metrics.json').read_text())
                 for model in ['vgg16','resnet50']}
        atomic_json(batch/'both_models_holdout_summary.json',summary)
        status('COMPLETE',summary=str(batch/'both_models_holdout_summary.json'))
        print(f'{now()} BOTH MODELS COMPLETE',flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        status('INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
               error_type=type(exc).__name__,message=str(exc))
        raise
    finally:
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',required=True)
    args=parser.parse_args()
    os.environ['KERAS_HOME']=str(ROOT/'runtime_cache/keras')
    os.environ['MPLCONFIGDIR']=str(ROOT/'runtime_cache/matplotlib')
    os.environ['PYTHONUNBUFFERED']='1'
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
    with training_lock(ROOT/'.overnight.lock'):
        supervise(Path(args.batch_dir).resolve())


if __name__=='__main__': main()
