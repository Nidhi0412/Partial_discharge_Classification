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
MODEL_TYPES = ('vgg16','resnet50','small_cnn','mobilenet_v3_small')


def canonical_model(name):
    name = name.lower()
    if name not in MODEL_TYPES:
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
    if name == 'small_cnn':
        regularizer = tf.keras.regularizers.l2(config['l2']) if config['l2'] else None
        inputs = tf.keras.Input((config['image_size'],config['image_size'],3),name='scalogram')
        x = inputs
        for filters in (16,32,64):
            x = tf.keras.layers.Conv2D(filters,3,padding='same',activation='relu',
                                       kernel_regularizer=regularizer)(x)
            x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(config['dropout'])(x)
        y = tf.keras.layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizer,
                                  name='p_teflon')(x)
        model = tf.keras.Model(inputs,y,name='small_cnn_insulation')
        model.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate']),
                      loss='binary_crossentropy',metrics=['accuracy'])
        return model

    constructor = {'vgg16': tf.keras.applications.VGG16,
                   'resnet50': tf.keras.applications.ResNet50,
                   'mobilenet_v3_small': tf.keras.applications.MobileNetV3Small}[name]
    constructor_args = dict(weights=config.get('weights','imagenet'),include_top=False,
                            input_shape=(config['image_size'],config['image_size'],3))
    if name == 'mobilenet_v3_small': constructor_args['include_preprocessing'] = False
    base = constructor(**constructor_args)
    for layer in base.layers:
        layer.trainable = False
    if config.get('fine_tuning', True):
        if name == 'vgg16':
            enabled = False
            for layer in base.layers:
                enabled = enabled or layer.name == 'block5_conv1'
                layer.trainable = enabled
        else:
            default_layers = 20 if name == 'mobilenet_v3_small' else 15
            for layer in base.layers[-config.get('backbone_trainable_layers',default_layers):]:
                # Avoid updating pretrained batch-normalization statistics on tiny batches.
                layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    if name == 'mobilenet_v3_small':
        inputs = tf.keras.Input((config['image_size'],config['image_size'],3),name='scalogram')
        features = base(tf.keras.layers.Rescaling(2.0,offset=-1.0,name='mobilenet_preprocess')(inputs))
        model_input = inputs
    else:
        features = base.output
        model_input = base.input
    x = tf.keras.layers.GlobalAveragePooling2D()(features)
    x = tf.keras.layers.Dropout(config['dropout'])(x)
    historical = config.get('training_profile') == 'historical'
    regularizer = tf.keras.regularizers.l2(config['l2']) if config['l2'] and not (historical and name=='vgg16') else None
    y = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizer,
                              bias_regularizer=regularizer if historical and name=='resnet50' else None,
                              name='p_teflon')(x)
    model = tf.keras.Model(model_input, y, name=name+'_insulation')
    optimizer_args = dict(clipnorm=1.0) if historical and name=='resnet50' else {}
    model.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate'],**optimizer_args),
                  loss='binary_crossentropy', metrics=['accuracy'])
    names = {layer.name for layer in model.layers}
    expected = {'vgg16':'block5_conv3','resnet50':'conv5_block3_out'}.get(name)
    if (expected and expected not in names) or model.output_shape[-1] != 1:
        raise RuntimeError('Built architecture does not match requested binary model')
    if config.get('fine_tuning',True) and not any(l.trainable and l.weights for l in base.layers):
        raise RuntimeError('Fine-tuning requested but no backbone weights are trainable')
    return model


def configure_loaded_backbone(tf, model, model_type, trainable_layers, learning_rate):
    """Enable only late weighted feature layers in a warm-started application model."""
    if model_type == 'mobilenet_v3_small':
        base = model.get_layer('MobileNetV3Small')
        backbone_layers = base.layers
        base.trainable = True
    elif model_type == 'resnet50':
        # ResNet50 is represented as flattened application layers in this model.
        # Exclude the task head and pooling/dropout layers from backbone selection.
        head = model.get_layer('p_teflon')
        head_index = model.layers.index(head)
        backbone_layers = model.layers[:head_index]
        base = model
    else:
        raise ValueError('Warm-start fine-tuning is supported only for MobileNetV3-Small and ResNet50')
    for layer in backbone_layers:
        layer.trainable = False
    for layer in backbone_layers[-trainable_layers:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    if not any(layer.trainable and layer.weights for layer in backbone_layers):
        raise RuntimeError('Warm-start fine-tuning enabled no weighted backbone layer')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',metrics=['accuracy'])
    return base


def make_dataset(rows, config, training=False):
    tf = tensorflow()
    if not rows:
        raise ValueError('Dataset membership is empty')
    if training and any(r.get('partition')!='train' for r in rows):
        raise ValueError('Training/augmentation accepts training partition members only')
    # This explicit ordered membership is also the order used by prediction export.
    rows = sorted(rows, key=lambda r: r['image_id'])
    paths = [r['image_path'] for r in rows]
    labels = [CLASS_NAMES.index(r['material']) for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    profile=config.get('augmentation_profile','time-translation')
    if profile not in {'historical-image','time-translation','scalogram-image','combined-image'}:
        raise ValueError('Unknown augmentation profile')
    numpy_aug=training and config.get('augment',False) and profile!='time-translation'
    if numpy_aug:
        if profile=='historical-image':
            from legacy_augmentation import make_transform
        else:
            from scalogram_augmentation import make_transform
        transform=make_transform(tf, rows, config)
    def decode(path, label):
        image = tf.image.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [config['image_size'], config['image_size']],
                                method=config.get('resize_interpolation','bilinear'))
        image = tf.cast(image, tf.float32) / 255.0
        if numpy_aug:
            seed=tf.random.uniform([],maxval=2**31-1,dtype=tf.int32,seed=config['seed'])
            image=tf.numpy_function(transform,[image,path,seed],tf.float32)
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
    if training and config.get('augment', False) and not numpy_aug:
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
    preprocessing_profile = config.get('preprocessing_profile','unit_scale')
    if preprocessing_profile == 'resnet50_imagenet':
        if config.get('model_type') != 'resnet50':
            raise ValueError('Native ResNet50 preprocessing may only be used with ResNet50')
        def preprocess_resnet(images, labels):
            # Decode/augmentation operates in [0,1]; the official application
            # transform expects RGB values in [0,255] and produces BGR mean-centred input.
            images = tf.keras.applications.resnet50.preprocess_input(images * 255.0)
            tf.debugging.assert_all_finite(images, 'Nonfinite ResNet50 input')
            return images, labels
        ds = ds.map(preprocess_resnet, num_parallel_calls=1)
    elif preprocessing_profile != 'unit_scale':
        raise ValueError(f'Unknown preprocessing profile {preprocessing_profile}')
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
    for name in ['tensorflow','tensorflow-cpu','tensorflow-intel','numpy','keras','Pillow','PyWavelets','scipy','tensorboard','h5py']:
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


def validate_fit_membership(train_rows, validation_rows):
    if not train_rows or not validation_rows:
        raise ValueError('Training and validation must be nonempty')
    for members,partition in [(train_rows,'train'),(validation_rows,'validation')]:
        if any(r.get('partition')!=partition for r in members):
            raise ValueError(f'Unexpected member in {partition} partition')
    for key in ['image_id','image_path','recording_id','group_id','image_sha256']:
        if any(not row.get(key) for row in train_rows+validation_rows):
            raise ValueError(f'Missing {key} before training')
        if {r[key] for r in train_rows} & {r[key] for r in validation_rows}:
            raise ValueError(f'Training/validation {key} overlap')


def train_one(model_type, train_rows, validation_rows, config, out):
    validate_fit_membership(train_rows, validation_rows)
    tf = tensorflow(); tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config['seed'])
    check_resources(config, out)
    state_path = out/'recovery_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state:
        model = tf.keras.models.load_model(snapshot_path(out,state['latest_file']))
    elif config.get('initial_checkpoint'):
        initial = Path(config['initial_checkpoint'])
        if digest(initial) != config['initial_checkpoint_sha256']:
            raise ValueError('Warm-start checkpoint changed')
        model = tf.keras.models.load_model(str(initial),compile=False)
        configure_loaded_backbone(tf,model,model_type,config['backbone_trainable_layers'],config['learning_rate'])
    else:
        model = build_model(model_type, config)
    input_config=dict(config,augmentation_export_dir=str(out/'augmented_samples'),
                      augmentation_ledger_dir=str(out/'augmentation_ledger'))
    train, effective_train = make_dataset(train_rows, input_config, training=True)
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
        if model_type in {'mobilenet_v3_small','resnet50'}:
            base_layers=(model.get_layer('MobileNetV3Small').layers if model_type=='mobilenet_v3_small'
                         else model.layers[:model.layers.index(model.get_layer('p_teflon'))])
            save_json(out/'backbone_trainable_layers.json',
                      [dict(name=l.name,trainable=l.trainable,weights=len(l.weights),
                            layer_type=type(l).__name__) for l in base_layers])
    checkpoint = out/'best.keras'
    callback = recovery_callback(tf, out, config, state)
    callbacks=[callback]
    if config.get('training_profile')=='historical':
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(out/'logs'),
                         histogram_freq=1,write_graph=True,write_images=True,update_freq='epoch'))
    try:
        if not state or not state['finished']:
            model.fit(train, validation_data=validation, epochs=config['epochs'],
                      initial_epoch=state['completed_epochs'] if state else 0,
                      callbacks=callbacks, verbose=2, shuffle=False)
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
    from training_artifacts import export_figures, export_model_files
    export_figures(out,history=state['history'])
    export_model_files(tf,best,out,state,enabled=config.get('legacy_h5_export',False))
    return best, checkpoint


def run_training(args):
    rows, split_config = load_splits(args.splits)
    tf = tensorflow()  # Fail before creating run outputs when the environment is unavailable.
    model_type = canonical_model(args.model)
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32 or args.patience < 1:
        raise ValueError('epochs/batch_size/patience must be positive and image_size >=32')
    if not 0 <= args.dropout < 1 or args.l2 < 0 or args.learning_rate <= 0 or not 0 <= args.time_shift < .5:
        raise ValueError('Invalid dropout/L2/learning-rate/time-shift configuration')
    from legacy_augmentation import HISTORICAL_PARAMETERS
    from scalogram_augmentation import operation_order,parameters
    specialized=parameters(vars(args))
    initial_cv_run = str(Path(args.initial_cv_run).resolve()) if args.initial_cv_run else None
    initial_final_run = str(Path(args.initial_final_run).resolve()) if args.initial_final_run else None
    if initial_cv_run and initial_final_run:
        raise ValueError('Choose only one warm-start source')
    initial_run = initial_cv_run or initial_final_run
    if initial_run:
        source_config=json.loads((Path(initial_run)/'run_config.json').read_text())
        source_status=json.loads((Path(initial_run)/'status.json').read_text())
        expected_source_purpose='development_cv' if initial_cv_run else 'final_fit'
        expected_command='cv' if initial_cv_run else 'final-fit'
        if (model_type not in {'mobilenet_v3_small','resnet50'} or source_config['model_type']!=model_type or
                source_config['split_sha256']!=split_config['split_sha256'] or
                source_config['purpose']!=expected_source_purpose or source_status.get('status')!='COMPLETE' or
                args.command!=expected_command):
            raise ValueError('Warm-start CV run has incompatible model, split, purpose or completion status')
    config = dict(model_type=model_type,image_size=args.image_size,batch_size=args.batch_size,
                  epochs=args.epochs,learning_rate=args.learning_rate,dropout=args.dropout,l2=args.l2,
                  patience=args.patience,seed=args.seed,augment=args.augment,time_shift=args.time_shift,
                  fine_tuning=not args.freeze_backbone,weights=None if args.weights=='none' else 'imagenet',
                  preprocessing_profile=args.preprocessing_profile,
                   preprocessing=('RGB decode, resize and training-only augmentation in [0,1], then official ResNet50 preprocess_input'
                                  if args.preprocessing_profile=='resnet50_imagenet' else
                                  'RGB decode, resize, float32 / 255, then MobileNetV3 [-1,1] scaling'
                                  if model_type=='mobilenet_v3_small' else
                                  f'RGB decode, resize {args.resize_interpolation}, float32 / 255'),class_names=list(CLASS_NAMES),
                  resize_interpolation=args.resize_interpolation,training_profile=args.training_profile,
                  augmentation_profile=args.augmentation_profile,
                  augmentation_order=operation_order(vars(args)),
                  augmentation_parameters=dict(historical=HISTORICAL_PARAMETERS if args.augmentation_profile in {'historical-image','combined-image'} else None,
                      specialized=specialized if args.augmentation_profile in {'scalogram-image','combined-image'} else None,
                      time_shift=args.time_shift if args.augmentation_profile=='time-translation' else None),
                  **specialized,
                  augmentation_samples_per_class=args.augmentation_samples_per_class,
                  legacy_h5_export=args.legacy_h5_export,
                  lr_reduce_factor=args.lr_reduce_factor,lr_min=args.lr_min,
                  initial_cv_run=initial_cv_run,initial_final_run=initial_final_run,
                  backbone_trainable_layers=args.backbone_trainable_layers,
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
                  additional_sources={name:digest(Path(__file__).with_name(name)) for name in ['legacy_augmentation.py','scalogram_augmentation.py','training_artifacts.py']},
                  evaluation_note='Development CV tunes/selects settings; it is not untouched nested outer-CV performance.')
    out = Path(args.output_dir).resolve()
    if getattr(args, 'resume', False):
        original = json.loads((out/'run_config.json').read_text())
        for key in ['source_sha256','safety_source_sha256','split_sha256','additional_sources']:
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
                if initial_cv_run:
                    initial=Path(initial_cv_run)/f'fold_{fold}'/'best.keras'
                    completed_file=Path(initial_cv_run)/f'fold_{fold}'/'completed.json'
                    completed=json.loads(completed_file.read_text())
                    if completed.get('best.keras')!=digest(initial):
                        raise ValueError(f'Warm-start fold {fold} checkpoint is not the committed source model')
                    per_config.update(initial_checkpoint=str(initial),initial_checkpoint_sha256=digest(initial))
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
                from training_artifacts import export_figures
                export_figures(dest,predictions=pred,scope=f'Development validation fold {fold}')
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
            from training_artifacts import export_figures
            export_figures(out,predictions=predictions,scope='Development out-of-fold validation')
        else:
            dev = [r for r in rows if r['fold']=='1' and r['partition']!='test']
            assign = assign_groups(dev,[.8,.2],args.seed+100,'group_id')
            train = [dict(r,partition='train',fold='final') for r in dev if assign[r['group_id']]==0]
            val = [dict(r,partition='validation',fold='final') for r in dev if assign[r['group_id']]==1]
            if any({r['material'] for r in subset}!=set(CLASS_NAMES) for subset in [train,val]):
                raise ValueError('Final grouped validation lacks a class')
            if {r['image_sha256'] for r in train} & {r['image_sha256'] for r in val}:
                raise ValueError('Final training/validation contains identical cross-partition content; inspect mapping')
            final_config=dict(config)
            if initial_final_run:
                checkpoint_info=json.loads((Path(initial_final_run)/'final_checkpoint.json').read_text())
                initial=Path(checkpoint_info['path'])
                if checkpoint_info['sha256']!=digest(initial):
                    raise ValueError('Warm-start final checkpoint is not the committed source model')
                final_config.update(initial_checkpoint=str(initial),initial_checkpoint_sha256=digest(initial))
            best, checkpoint = train_one(model_type,train,val,final_config,out)
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
        from training_artifacts import export_figures
        export_figures(dest,predictions=predictions,scope='Locked holdout')
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
        for name in MODEL_TYPES:
            config = dict(image_size=32,batch_size=2,epochs=1,learning_rate=.001,dropout=.1,l2=0.,
                          patience=1,seed=42,augment=True,time_shift=.03,fine_tuning=False,weights=None)
            folder = out/name;folder.mkdir()
            # Exercise augmentation through fit, callbacks, persistence and prediction export.
            model, checkpoint = train_one(name,[dict(r,partition='train') for r in rows[:4]],rows[4:],config,folder)
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
        p.add_argument('--model',choices=MODEL_TYPES,default=default_model)
        p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=2)
        p.add_argument('--image-size',type=int,default=224);p.add_argument('--learning-rate',type=float,default=1e-5)
        p.add_argument('--dropout',type=float,default=.5);p.add_argument('--l2',type=float,default=.01)
        p.add_argument('--patience',type=int,default=10);p.add_argument('--seed',type=int,default=42)
        p.add_argument('--augment',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--time-shift',type=float,default=.03)
        p.add_argument('--training-profile',choices=['historical','reviewed'],default='historical')
        p.add_argument('--augmentation-profile',choices=['historical-image','time-translation','scalogram-image','combined-image'],default='scalogram-image')
        from scalogram_augmentation import PRIMARY_DEFAULTS
        for key,value in PRIMARY_DEFAULTS.items():p.add_argument('--'+key.replace('_','-'),type=float,default=value)
        p.add_argument('--resize-interpolation',choices=['nearest','bilinear'],default='nearest')
        p.add_argument('--preprocessing-profile',choices=['unit_scale','resnet50_imagenet'],default='unit_scale')
        p.add_argument('--augmentation-samples-per-class',type=int,default=5)
        p.add_argument('--legacy-h5-export',action=argparse.BooleanOptionalAction,default=True)
        p.add_argument('--lr-reduce-factor',type=float,default=.1)
        p.add_argument('--lr-min',type=float,default=1e-7)
        p.add_argument('--freeze-backbone',action='store_true')
        p.add_argument('--initial-cv-run')
        p.add_argument('--initial-final-run')
        p.add_argument('--backbone-trainable-layers',type=int,default=20)
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
        if args.backbone_trainable_layers < 1:
            parser.error('Backbone trainable layer count must be positive')
        if args.preprocessing_profile=='resnet50_imagenet' and args.model!='resnet50':
            parser.error('resnet50_imagenet preprocessing requires --model resnet50')
        if not math.isfinite(args.min_free_ram_gb) or not math.isfinite(args.min_free_disk_gb) or min(args.min_free_ram_gb,args.min_free_disk_gb) <= 0:
            parser.error('RAM/disk reserves must be finite and positive')
        if args.augmentation_samples_per_class<1 or not 0<args.lr_reduce_factor<1 or not 0<=args.lr_min<=args.learning_rate:
            parser.error('Invalid augmentation sample count or learning-rate plateau settings')
        with training_lock(Path(__file__).resolve().parents[1]/'.training.lock'):
            run_training(args)


if __name__ == '__main__':
    main()
