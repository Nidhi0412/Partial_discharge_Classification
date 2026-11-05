"""Explicit 224px fine-tuning/recovery check; never reads the locked holdout.

Run separately for each architecture so process memory is released between models.
These two-epoch, four-image checks are engineering checks, not research results.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import training_pipeline as pipeline
from training_safety import (atomic_json, check_resources, configure_runtime,
                             training_lock, snapshot_path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['vgg16','resnet50'])
    p.add_argument('--output-dir',required=True)
    p.add_argument('--historical',action='store_true',help='Verify restored historical augmentation/model/output profile')
    p.add_argument('--png-specialized',action='store_true',help='Verify combined historical + specialized PNG chain')
    p.add_argument('--primary-png',action='store_true',help='Verify final primary PNG profile without rotation/flip/stretch')
    p.add_argument('--weights',choices=['none','imagenet'],default='none')
    args=p.parse_args()
    if sum([args.historical,args.png_specialized,args.primary_png])>1:p.error('Choose one augmentation profile')
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=False)
    config=dict(image_size=224,batch_size=2,epochs=2,learning_rate=1e-5,dropout=.5,l2=.01,
                patience=10,seed=42,augment=True,time_shift=.03,fine_tuning=True,weights=None if args.weights=='none' else 'imagenet',
                cpu_threads=4,data_threads=1,prefetch_batches=1,min_free_ram_gb=1.5,min_free_disk_gb=5.)
    if args.historical or args.png_specialized or args.primary_png:
        config.update(training_profile='historical',augmentation_profile='historical-image',
                      resize_interpolation='nearest',augmentation_samples_per_class=5,
                      legacy_h5_export=True,lr_reduce_factor=.1,lr_min=1e-7)
    if args.png_specialized:
        from scalogram_augmentation import DEFAULTS
        config.update(augmentation_profile='combined-image',**DEFAULTS)
    if args.primary_png:
        from scalogram_augmentation import PRIMARY_DEFAULTS
        config.update(augmentation_profile='scalogram-image',**PRIMARY_DEFAULTS)
    with training_lock(ROOT/'.training.lock'):
        tf=pipeline.tensorflow(); configure_runtime(tf,config)
        resources=check_resources(config,out)
        rows,split=pipeline.load_splits(ROOT/'splits/run01/splits.csv')
        def subset(partition):
            return [next(r for r in rows if r['fold']=='1' and r['partition']==partition and r['material']==material)
                    for material in ['Nomex','Teflon']]
        train,val=subset('train'),subset('validation')
        original=pipeline.recovery_callback
        original_builder=pipeline.build_model
        weight_capture={}
        def capture_builder(*args,**kwargs):
            model=original_builder(*args,**kwargs)
            active=[l for l in model.layers if isinstance(l,tf.keras.layers.Conv2D) and l.trainable][-1]
            frozen=[l for l in model.layers if isinstance(l,tf.keras.layers.Conv2D) and not l.trainable][0]
            weight_capture.update(active_name=active.name,active_before=active.kernel.numpy().copy(),
                frozen_name=frozen.name,frozen_before=frozen.kernel.numpy().copy())
            return model
        def interrupting_callback(*args,**kwargs):
            callback=original(*args,**kwargs)
            def on_epoch_begin(epoch,logs=None):
                if epoch==1: raise KeyboardInterrupt('Deliberate recovery check after completed epoch 1')
            callback.on_epoch_begin=on_epoch_begin
            return callback
        started=time.perf_counter()
        with patch.object(pipeline,'recovery_callback',interrupting_callback),patch.object(pipeline,'build_model',capture_builder):
            try: pipeline.train_one(args.model,train,val,config,out)
            except KeyboardInterrupt: pass
            else: raise AssertionError('Expected deliberate interruption')
        state=json.loads((out/'recovery_state.json').read_text())
        assert state['completed_epochs']==1
        restored=tf.keras.models.load_model(snapshot_path(out,state['latest_file']))
        import numpy as np
        assert not np.array_equal(weight_capture['active_before'],restored.get_layer(weight_capture['active_name']).kernel.numpy()),'Backbone kernel did not update'
        np.testing.assert_array_equal(weight_capture['frozen_before'],restored.get_layer(weight_capture['frozen_name']).kernel.numpy())
        before=int(restored.optimizer.iterations.numpy())
        assert before==1, before
        del restored
        model,checkpoint=pipeline.train_one(args.model,train,val,config,out)
        state=json.loads((out/'recovery_state.json').read_text())
        assert state['completed_epochs']==2 and len(state['history']['loss'])==2
        latest=tf.keras.models.load_model(snapshot_path(out,state['latest_file']))
        after=int(latest.optimizer.iterations.numpy())
        assert after==2, after
        del latest
        gc.collect()
        pred=pipeline.predict_rows(model,val,config,checkpoint,args.model,split['split_sha256'])
        assert len(pred)==2
        if args.historical or args.png_specialized or args.primary_png:
            assert len(list((out/'augmented_samples').glob('*/*_augmented.npy')))==2
            assert (out/'training_history.png').is_file() and (out/'model.h5').is_file()
            assert (out/'best_weights.h5').is_file()
            legacy=tf.keras.models.load_model(out/'model.h5',compile=False)
            comparison=pipeline.predict_rows(legacy,val,config,checkpoint,args.model,split['split_sha256'])
            import numpy as np
            np.testing.assert_allclose([r['p_teflon'] for r in pred],[r['p_teflon'] for r in comparison],rtol=1e-5,atol=1e-6)
            del legacy
            from training_artifacts import export_figures
            export_figures(out,predictions=pred,scope='Engineering validation only')
            assert (out/'confusion_matrix.png').is_file() and (out/'roc.png').is_file()
            if args.png_specialized or args.primary_png:
                examples=json.loads((out/'augmentation_samples_manifest.json').read_text())
                expected_stages=4 if args.primary_png else 6
                assert len(examples)==2 and all(len(e['stages'])==expected_stages for e in examples)
                if args.primary_png:
                    expected=['horizontal_image_shift','row_band_mask','column_band_mask','scaled_pixel_noise']
                    assert all([s['operation'] for s in e['stages']]==expected for e in examples)
                ledgers=list((out/'augmentation_ledger').glob('*.jsonl'))
                generated=[json.loads(line) for path in ledgers for line in path.read_text().splitlines()]
                assert generated and all(r['partition']=='train' for r in generated)
                assert {r['recording_id'] for r in generated}<={r['recording_id'] for r in train}
        result=dict(status='PASS',model=args.model,image_size=224,batch_size=2,fine_tuning=True,
                    weights='ImageNet: cached official weights' if args.weights=='imagenet' else 'random: no download; not research accuracy',train_images=2,validation_images=2,
                    actual_backbone_update=True,frozen_backbone_unchanged=True,
                    checked_trainable_convolution=weight_capture['active_name'],checked_frozen_convolution=weight_capture['frozen_name'],
                    holdout_images_trained_or_evaluated=0,interrupted_after_epoch=1,resumed_to_epoch=2,
                    note='Split loading verifies hashes of all files; holdout tensors/predictions are never used.',
                    optimizer_steps_before=before,optimizer_steps_after=after,
                    seconds=round(time.perf_counter()-started,2),resources_before=resources,
                    resources_after=check_resources(config,out),config=config,
                    limitation='Tiny development subset; no guarantee of full-run completion or bitwise random-state replay.')
        atomic_json(out/'check_result.json',result)
        print(json.dumps(result,indent=2))


if __name__=='__main__': main()
