"""One explicit training/evaluation route for binary insulation classification.

Heavy dependencies are loaded only when training/inference is explicitly requested.
The locked holdout is never evaluated during development CV.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import random
import shutil
import sys

from dataset_manifest import (CLASS_NAMES, FIELDS, assign_groups, digest, load_splits,
                              write_csv)
from training_safety import (atomic_json, check_resources, configure_runtime,
                             export_history, recovery_callback, snapshot_path, training_lock)

PRED_FIELDS = ['image_id','image_path','recording_id','session_id','material','electrode',
               'image_sha256','group_id','fold','partition','y_true','p_teflon','y_pred',
               'model_type','checkpoint_path','checkpoint_sha256','split_sha256']


def canonical_model(name):
    name = name.lower()
    if name not in {'vgg16', 'resnet50'}:
        raise ValueError(f'Unknown architecture {name}')
    return name


def tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError('TensorFlow is missing in this interpreter. Use the documented Python 3.11 environment.') from exc
    return tf


def build_model(name, config):
    tf = tensorflow(); name = canonical_model(name)
    constructor = {'vgg16': tf.keras.applications.VGG16,
                   'resnet50': tf.keras.applications.ResNet50}[name]
    base = constructor(weights=config.get('weights', 'imagenet'), include_top=False,
                       input_shape=(config['image_size'], config['image_size'], 3))
    for layer in base.layers:
        layer.trainable = False
    if config.get('fine_tuning', True):
        if name == 'vgg16':
            enabled = False
            for layer in base.layers:
                enabled = enabled or layer.name == 'block5_conv1'
                layer.trainable = enabled
        else:
            for layer in base.layers[-config.get('resnet_trainable_layers', 15):]:
                # Avoid updating pretrained batch-normalization statistics on tiny batches.
                layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(config['dropout'])(x)
    regularizer = tf.keras.regularizers.l2(config['l2']) if config['l2'] else None
    y = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizer,
                              name='p_teflon')(x)
    model = tf.keras.Model(base.input, y, name=name+'_insulation')
    model.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate']),
                  loss='binary_crossentropy', metrics=['accuracy'])
    names = {layer.name for layer in model.layers}
    expected = 'block5_conv3' if name == 'vgg16' else 'conv5_block3_out'
    if expected not in names or model.output_shape[-1] != 1:
        raise RuntimeError('Built architecture does not match requested binary model')
    return model


def make_dataset(rows, config, training=False):
    tf = tensorflow()
    if not rows:
        raise ValueError('Dataset membership is empty')
    # This explicit ordered membership is also the order used by prediction export.
    rows = sorted(rows, key=lambda r: r['image_id'])
    paths = [r['image_path'] for r in rows]
    labels = [CLASS_NAMES.index(r['material']) for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def decode(path, label):
        image = tf.image.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [config['image_size'], config['image_size']])
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.ensure_shape(image, [config['image_size'], config['image_size'], 3])
        tf.debugging.assert_all_finite(image, 'Nonfinite image tensor')
        return image, tf.cast(label, tf.float32)[None]
    options = tf.data.Options(); options.experimental_deterministic = True
    options.threading.private_threadpool_size = config.get('data_threads', 1)
    options.threading.max_intra_op_parallelism = 1
    ds = ds.with_options(options)
    if training:
        # Shuffle paths, not decoded float32 images (which can consume nearly 1 GiB).
        ds = ds.shuffle(len(rows), seed=config['seed'], reshuffle_each_iteration=True)
    ds = ds.map(decode, num_parallel_calls=config.get('data_threads', 1))
    ds = ds.batch(config['batch_size'])
    if training and config.get('augment', False):
        # Optional bounded time-axis translation of rendered images; no frequency mixing.
        # This is not waveform phase shifting or proof of physical invariance.
        augmentation = tf.keras.layers.RandomTranslation(
            height_factor=0.0, width_factor=config.get('time_shift', .03),
            fill_mode='constant', fill_value=0.0, seed=config['seed'])
        def augment(images, labels):
            transformed = tf.clip_by_value(augmentation(images, training=True), 0., 1.)
            tf.debugging.assert_all_finite(transformed, 'Nonfinite augmentation')
            tf.debugging.assert_equal(tf.shape(images), tf.shape(transformed))
            return transformed, labels
        ds = ds.map(augment, num_parallel_calls=1)
    return ds.prefetch(config.get('prefetch_batches', 1)), rows


def binary_metrics(labels, probabilities):
    if not labels or len(labels) != len(probabilities):
        raise ValueError('Labels and probabilities must be nonempty and aligned')
    if any(y not in [0, 1] for y in labels) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid binary labels/probabilities')
    preds = [int(p >= .5) for p in probabilities]
    tn = sum(y == 0 and p == 0 for y,p in zip(labels,preds))
    fp = sum(y == 0 and p == 1 for y,p in zip(labels,preds))
    fn = sum(y == 1 and p == 0 for y,p in zip(labels,preds))
    tp = sum(y == 1 and p == 1 for y,p in zip(labels,preds))
    def div(a,b): return a/b if b else 0.0
    def per_class(correct, incorrect_positive, incorrect_negative):
        return dict(precision=div(correct, correct+incorrect_positive),
                    recall=div(correct, correct+incorrect_negative),
                    f1=div(2*correct,2*correct+incorrect_positive+incorrect_negative))
    negative, positive = per_class(tn,fn,fp), per_class(tp,fp,fn)
    n0, n1 = labels.count(0), labels.count(1)
    auc = None
    if n0 and n1:
        ordered = sorted(zip(probabilities,labels)); rank_sum = 0.; i = 0
        while i < len(ordered):
            j = i+1
            while j < len(ordered) and ordered[j][0] == ordered[i][0]: j += 1
            rank_sum += ((i+1+j)/2) * sum(label for _,label in ordered[i:j])
            i = j
        auc = (rank_sum - n1*(n1+1)/2) / (n0*n1)
    return dict(samples=len(labels), confusion_matrix=[[tn,fp],[fn,tp]],
                accuracy=(tn+tp)/len(labels), auc=auc, specificity=div(tn,tn+fp),
                positive_class='Teflon', per_class=dict(Nomex=negative,Teflon=positive),
                macro={k:(negative[k]+positive[k])/2 for k in positive})


def recording_predictions(rows):
    grouped = defaultdict(list)
    for row in rows: grouped[row['recording_id']].append(row)
    result = []
    for recording_id, values in sorted(grouped.items()):
        if len({int(r['y_true']) for r in values}) != 1:
            raise ValueError('A recording has conflicting true labels')
        if len({r['fold'] for r in values}) != 1:
            raise ValueError('OOF predictions contain repeated recording assignments')
        result.append(dict(recording_id=recording_id, session_id=values[0].get('session_id',''),
                           group_id=values[0].get('group_id',recording_id),
                           y_true=int(values[0]['y_true']),
                           p_teflon=sum(float(v['p_teflon']) for v in values)/len(values),
                           image_count=len(values)))
    return result


def prediction_summary(rows, bootstrap=0, seed=42, group_by='recording_id'):
    records = recording_predictions(rows)
    result = dict(image=binary_metrics([int(r['y_true']) for r in rows], [float(r['p_teflon']) for r in rows]),
                  recording=binary_metrics([r['y_true'] for r in records], [r['p_teflon'] for r in records]),
                  aggregation='Mean p_teflon across all images of each original recording; threshold >= 0.5')
    if bootstrap:
        if any(not r.get(group_by) for r in records): raise ValueError('Missing uncertainty grouping ID')
        clusters = defaultdict(list)
        for row in records: clusters[row[group_by]].append(row)
        rng = random.Random(seed); keys = sorted(clusters)
        scores = []
        for _ in range(bootstrap):
            sample = [r for key in rng.choices(keys,k=len(keys)) for r in clusters[key]]
            scores.append(binary_metrics([r['y_true'] for r in sample], [r['p_teflon'] for r in sample])['accuracy'])
        scores.sort()
        result['recording_accuracy_cluster_bootstrap_95_percentile'] = dict(
            lower=scores[int(.025*(len(scores)-1))], upper=scores[int(.975*(len(scores)-1))],
            draws=bootstrap, grouping=group_by, independent_clusters=len(keys),
            limitation='Conditional on this fitted model; does not include model selection or training variability')
    return result, records


def save_json(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def package_versions():
    result = {}
    for name in ['tensorflow','tensorflow-cpu','tensorflow-intel','numpy','keras','Pillow','PyWavelets']:
        try: result[name] = metadata.version(name)
        except metadata.PackageNotFoundError: pass
    return result


def predict_rows(model, rows, config, checkpoint, model_type, split_sha):
    ds, ordered = make_dataset(rows, config, training=False)
    scores = model.predict(ds, verbose=0).reshape(-1).tolist()
    if len(scores) != len(ordered): raise RuntimeError('Prediction membership mismatch')
    cp_hash = digest(checkpoint)
    output = []
    for row, score in zip(ordered,scores):
        value = dict(row)
        value.update(y_true=CLASS_NAMES.index(row['material']), p_teflon=float(score),
                     y_pred=int(score >= .5), model_type=model_type,
                     checkpoint_path=str(Path(checkpoint).resolve()), checkpoint_sha256=cp_hash,
                     split_sha256=split_sha)
        output.append(value)
    binary_metrics([r['y_true'] for r in output], [r['p_teflon'] for r in output])
    return output


def train_one(model_type, train_rows, validation_rows, config, out):
    tf = tensorflow(); tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config['seed'])
    check_resources(config, out)
    state_path = out/'recovery_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    model = (tf.keras.models.load_model(snapshot_path(out, state['latest_file']))
             if state else build_model(model_type, config))
    train, effective_train = make_dataset(train_rows, config, training=True)
    validation, effective_val = make_dataset(validation_rows, config, training=False)
    if {r['group_id'] for r in effective_train} & {r['group_id'] for r in effective_val}:
        raise ValueError('Training/validation group overlap')
    for name, members in [('effective_train.csv', effective_train), ('effective_validation.csv', effective_val)]:
        path = out/name
        if path.exists():
            with path.open(newline='', encoding='utf-8') as stream:
                previous = list(csv.DictReader(stream))
            if previous != [{k: str(v) for k,v in row.items()} for row in members]:
                raise ValueError('Recovery membership differs from original training membership')
        else:
            write_csv(path,members,list(members[0]))
    if not (out/'architecture.json').exists():
        save_json(out/'architecture.json', json.loads(model.to_json()))
        save_json(out/'trainable_layers.json', [dict(name=l.name,trainable=l.trainable,weights=len(l.weights)) for l in model.layers])
    checkpoint = out/'best.keras'
    callback = recovery_callback(tf, out, config, state)
    try:
        if not state or not state['finished']:
            model.fit(train, validation_data=validation, epochs=config['epochs'],
                      initial_epoch=state['completed_epochs'] if state else 0,
                      callbacks=[callback], verbose=2, shuffle=False)
    except (Exception, KeyboardInterrupt) as exc:
        atomic_json(out/'interruption.json', dict(error_type=type(exc).__name__, message=str(exc),
                    recovery='Resume from last committed epoch; unfinished epoch is repeated.'))
        raise
    state = callback.state
    export_history(out, state)
    temporary = out/'best.pending.keras'
    shutil.copyfile(snapshot_path(out, state['best_file']), temporary)
    temporary.replace(checkpoint)
    # Evaluate exactly the saved best-val-loss checkpoint, not a different final epoch.
    del model
    tf.keras.backend.clear_session()
    best = tf.keras.models.load_model(str(checkpoint), compile=False)
    return best, checkpoint


def run_training(args):
    rows, split_config = load_splits(args.splits)
    tf = tensorflow()  # Fail before creating run outputs when the environment is unavailable.
    model_type = canonical_model(args.model)
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32 or args.patience < 1:
        raise ValueError('epochs/batch_size/patience must be positive and image_size >=32')
    if not 0 <= args.dropout < 1 or args.l2 < 0 or args.learning_rate <= 0 or not 0 <= args.time_shift < .5:
        raise ValueError('Invalid dropout/L2/learning-rate/time-shift configuration')
    config = dict(model_type=model_type,image_size=args.image_size,batch_size=args.batch_size,
                  epochs=args.epochs,learning_rate=args.learning_rate,dropout=args.dropout,l2=args.l2,
                  patience=args.patience,seed=args.seed,augment=args.augment,time_shift=args.time_shift,
                  fine_tuning=not args.freeze_backbone,weights=None if args.weights=='none' else 'imagenet',
                  preprocessing='RGB decode, resize bilinear, float32 / 255',class_names=list(CLASS_NAMES),
                  purpose='development_cv' if args.command=='cv' else 'final_fit',
                  split_manifest=str(Path(args.splits).resolve()), split_sha256=split_config['split_sha256'],
                  group_by=split_config['group_by'],folds=split_config['folds'],python=sys.version,tensorflow=tf.__version__,platform=platform.platform(),
                  invocation=sys.argv,created_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=digest(__file__),
                  packages=package_versions(),
                  selection_metric='val_loss',
                  cpu_threads=args.cpu_threads,data_threads=args.data_threads,prefetch_batches=args.prefetch_batches,
                  min_free_ram_gb=args.min_free_ram_gb,min_free_disk_gb=args.min_free_disk_gb,
                  safety_source_sha256=digest(Path(__file__).with_name('training_safety.py')),
                  evaluation_note='Development CV tunes/selects settings; it is not untouched nested outer-CV performance.')
    out = Path(args.output_dir).resolve()
    if getattr(args, 'resume', False):
        original = json.loads((out/'run_config.json').read_text())
        for key in ['source_sha256','safety_source_sha256','split_sha256']:
            if config[key] != original[key]:
                raise ValueError(f'Cannot resume: {key} changed')
        config = original
        if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['status']=='COMPLETE':
            raise ValueError('This run already completed')
    else:
        out.mkdir(parents=True,exist_ok=False)
        save_json(out/'run_config.json',config)
    configure_runtime(tf, config)
    try:
        atomic_json(out/'resource_check.json', check_resources(config, out))
        if args.command == 'cv':
            predictions = []
            folds = sorted({r['fold'] for r in rows},key=int)
            for fold in folds:
                dest = out/f'fold_{fold}'; dest.mkdir(exist_ok=True)
                train = [r for r in rows if r['fold']==fold and r['partition']=='train']
                val = [r for r in rows if r['fold']==fold and r['partition']=='validation']
                per_config = dict(config,seed=args.seed+int(fold))
                if not (dest/'fold_config.json').exists(): save_json(dest/'fold_config.json',per_config)
                if (dest/'completed.json').exists():
                    completed = json.loads((dest/'completed.json').read_text())
                    for filename, expected in completed.items():
                        if digest(dest/filename) != expected: raise ValueError('Completed fold artifact changed')
                    with (dest/'predictions.csv').open(newline='',encoding='utf-8') as stream:
                        pred = list(csv.DictReader(stream))
                    for row in pred:
                        row.update(y_true=int(row['y_true']),p_teflon=float(row['p_teflon']),y_pred=int(row['y_pred']))
                    predictions.extend(pred)
                    continue
                best, checkpoint = train_one(model_type,train,val,per_config,dest)
                pred = predict_rows(best,val,per_config,checkpoint,model_type,config['split_sha256'])
                if (dest/'predictions.csv').exists():
                    (dest/'predictions.csv').unlink()  # Incomplete fold export, regenerated from committed model.
                write_csv(dest/'predictions.csv',pred,PRED_FIELDS)
                metrics, records = prediction_summary(pred)
                atomic_json(dest/'metrics.json',metrics)
                atomic_json(dest/'completed.json',{name:digest(dest/name) for name in ['best.keras','predictions.csv','metrics.json']})
                predictions.extend(pred)
                del best
                tf.keras.backend.clear_session()
            if len({r['image_id'] for r in predictions}) != len(predictions):
                raise ValueError('Repeated OOF image predictions')
            if (out/'oof_predictions.csv').exists(): (out/'oof_predictions.csv').unlink()
            write_csv(out/'oof_predictions.csv',predictions,PRED_FIELDS)
            metrics, records = prediction_summary(predictions)
            metrics['purpose'] = 'development validation; settings/early stopping used these folds'
            atomic_json(out/'metrics.json',metrics)
        else:
            dev = [r for r in rows if r['fold']=='1' and r['partition']!='test']
            assign = assign_groups(dev,[.8,.2],args.seed+100,'group_id')
            train = [dict(r,partition='train',fold='final') for r in dev if assign[r['group_id']]==0]
            val = [dict(r,partition='validation',fold='final') for r in dev if assign[r['group_id']]==1]
            if any({r['material'] for r in subset}!=set(CLASS_NAMES) for subset in [train,val]):
                raise ValueError('Final grouped validation lacks a class')
            if {r['image_sha256'] for r in train} & {r['image_sha256'] for r in val}:
                raise ValueError('Final training/validation contains identical cross-partition content; inspect mapping')
            best, checkpoint = train_one(model_type,train,val,config,out)
            atomic_json(out/'final_checkpoint.json',dict(path=str(checkpoint),sha256=digest(checkpoint),
                                                       note='Not yet evaluated on locked holdout'))
        atomic_json(out/'status.json',dict(status='COMPLETE',purpose=config['purpose']))
    except (Exception, KeyboardInterrupt) as exc:
        atomic_json(out/'status.json',dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',purpose=config['purpose']))
        atomic_json(out/'failure.json',dict(status='FAILED',error_type=type(exc).__name__,message=str(exc)))
        raise


def evaluate_run(args):
    run = Path(args.run_dir).resolve()
    config = json.loads((run/'run_config.json').read_text(encoding='utf-8'))
    if config['purpose'] != 'final_fit':
        raise ValueError('Locked holdout requires a final_fit run selected using development information')
    status = json.loads((run/'status.json').read_text(encoding='utf-8'))
    if status['status'] != 'COMPLETE': raise ValueError('Final fit did not complete')
    rows, split = load_splits(args.splits)
    if split['split_sha256'] != config['split_sha256']:
        raise ValueError('Evaluation split differs from trained run')
    checkpoint = json.loads((run/'final_checkpoint.json').read_text(encoding='utf-8'))
    if digest(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('Selected checkpoint changed')
    # Enforce one primary holdout evaluation per final fit; no test-based checkpoint selection.
    tf = tensorflow()
    dest = run/'holdout_evaluation'; dest.mkdir(exist_ok=False)
    try:
        model = tf.keras.models.load_model(checkpoint['path'],compile=False)
        test = [r for r in rows if r['fold']=='1' and r['partition']=='test']
        predictions = predict_rows(model,test,config,checkpoint['path'],config['model_type'],config['split_sha256'])
        write_csv(dest/'predictions.csv',predictions,PRED_FIELDS)
        metrics, records = prediction_summary(predictions,bootstrap=args.bootstrap,seed=config['seed'],group_by=config['group_by'])
        write_csv(dest/'recording_predictions.csv',records,list(records[0]))
        save_json(dest/'metrics.json',metrics)
        save_json(dest/'status.json',dict(status='COMPLETE',endpoint='locked_holdout'))
    except Exception as exc:
        save_json(dest/'failure.json',dict(status='FAILED',message=str(exc)))
        raise


def smoke_test():
    """Explicit opt-in tiny CPU test: synthetic images, both real backbones, no download."""
    import tempfile
    import numpy as np
    tf = tensorflow()
    with tempfile.TemporaryDirectory(prefix='pd_smoke_') as directory:
        out = Path(directory)
        rows = []
        for i in range(8):
            p = out/f'{i}.png'
            pixels = np.random.default_rng(i).integers(0,256,(32,32,3),dtype=np.uint8)
            tf.io.write_file(str(p),tf.io.encode_png(pixels))
            rows.append(dict(image_id=str(i),image_path=str(p),recording_id=str(i),group_id=str(i),
                             material=CLASS_NAMES[i%2],electrode='',session_id='',image_sha256=digest(p),fold='1',partition='validation'))
        for name in ['vgg16','resnet50']:
            config = dict(image_size=32,batch_size=2,epochs=1,learning_rate=.001,dropout=.1,l2=0.,
                          patience=1,seed=42,augment=True,time_shift=.03,fine_tuning=False,weights=None)
            folder = out/name;folder.mkdir()
            # Exercise augmentation through fit, callbacks, persistence and prediction export.
            model, checkpoint = train_one(name,rows[:4],rows[4:],config,folder)
            pred = predict_rows(model,rows[4:],config,checkpoint,name,'synthetic-smoke-only')
            assert len(pred)==4 and all(0 <= r['p_teflon'] <= 1 for r in pred)
            ds,_ = make_dataset(rows[4:],config,False)
            first = model.predict(ds,verbose=0)
            reloaded = tf.keras.models.load_model(str(checkpoint),compile=False)
            np.testing.assert_allclose(first,reloaded.predict(ds,verbose=0),rtol=1e-5,atol=1e-6)
            print(f'PASS: {name} synthetic fit, training augmentation, checkpoint reload and prediction alignment')


def main(argv=None, default_model='vgg16'):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for command in ['cv','final-fit']:
        p = sub.add_parser(command)
        p.add_argument('--splits',required=True);p.add_argument('--output-dir',required=True)
        p.add_argument('--model',choices=['vgg16','resnet50'],default=default_model)
        p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=2)
        p.add_argument('--image-size',type=int,default=224);p.add_argument('--learning-rate',type=float,default=1e-5)
        p.add_argument('--dropout',type=float,default=.5);p.add_argument('--l2',type=float,default=.01)
        p.add_argument('--patience',type=int,default=10);p.add_argument('--seed',type=int,default=42)
        p.add_argument('--augment',action='store_true');p.add_argument('--time-shift',type=float,default=.03)
        p.add_argument('--freeze-backbone',action='store_true')
        p.add_argument('--weights',choices=['imagenet','none'],default='imagenet')
        p.add_argument('--cpu-threads',type=int,default=4)
        p.add_argument('--data-threads',type=int,default=1)
        p.add_argument('--prefetch-batches',type=int,default=1)
        p.add_argument('--min-free-ram-gb',type=float,default=1.5)
        p.add_argument('--min-free-disk-gb',type=float,default=5.0)
    p = sub.add_parser('resume',help='Continue saved settings and last completed epoch of an interrupted run')
    p.add_argument('--run-dir',required=True)
    p = sub.add_parser('evaluate')
    p.add_argument('--run-dir',required=True);p.add_argument('--splits',required=True)
    p.add_argument('--bootstrap',type=int,default=1000)
    sub.add_parser('smoke-test')
    args = parser.parse_args(argv)
    if args.command == 'smoke-test': smoke_test()
    elif args.command == 'evaluate':
        if args.bootstrap < 0: parser.error('bootstrap must be nonnegative')
        evaluate_run(args)
    else:
        if args.command == 'resume':
            config = json.loads((Path(args.run_dir)/'run_config.json').read_text())
            args = argparse.Namespace(**dict(config,
                command='cv' if config['purpose']=='development_cv' else 'final-fit',
                model=config['model_type'],splits=config['split_manifest'],output_dir=args.run_dir,
                freeze_backbone=not config['fine_tuning'],weights='none' if config['weights'] is None else 'imagenet',
                resume=True))
        if min(args.cpu_threads,args.data_threads,args.prefetch_batches) < 1:
            parser.error('Thread and prefetch limits must be positive')
        if not math.isfinite(args.min_free_ram_gb) or not math.isfinite(args.min_free_disk_gb) or min(args.min_free_ram_gb,args.min_free_disk_gb) <= 0:
            parser.error('RAM/disk reserves must be finite and positive')
        with training_lock(Path(__file__).resolve().parents[1]/'.training.lock'):
            run_training(args)


if __name__ == '__main__':
    main()
