"""Prove a real backbone weight changes while frozen weights stay fixed.

Tiny synthetic 32x32 one-step engineering check; not research training.
Run each architecture in its own process.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from training_pipeline import tensorflow,build_model,load_splits
from training_safety import configure_runtime,check_resources,atomic_json,training_lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['vgg16','resnet50'],required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    config=dict(image_size=32,batch_size=2,epochs=1,learning_rate=1e-5,dropout=.5,l2=.01,
                training_profile='historical',fine_tuning=True,weights=None,cpu_threads=4)
    with training_lock(ROOT/'.training.lock'):
        tf=tensorflow();configure_runtime(tf,config);check_resources(config,ROOT)
        tf.keras.utils.set_random_seed(42)
        model=build_model(args.model,config)
        import numpy as np
        trainable=[l for l in model.layers if isinstance(l,tf.keras.layers.Conv2D) and l.trainable]
        frozen=[l for l in model.layers if isinstance(l,tf.keras.layers.Conv2D) and not l.trainable]
        assert trainable and frozen
        active=trainable[-1];locked=frozen[0]
        before=active.kernel.numpy().copy();frozen_before=locked.kernel.numpy().copy()
        head_before=model.get_layer('p_teflon').kernel.numpy().copy()
        model.train_on_batch(tf.random.uniform((2,32,32,3)),tf.constant([[0.],[1.]]))
        assert not np.array_equal(before,active.kernel.numpy()), 'Backbone kernel did not change'
        np.testing.assert_array_equal(frozen_before,locked.kernel.numpy())
        assert not np.array_equal(head_before,model.get_layer('p_teflon').kernel.numpy())
        bn=[l for l in model.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
        assert all(not l.trainable for l in bn)
        rows,split=load_splits(ROOT/'splits/run01/splits.csv')
        images={r['image_id']:r for r in rows}
        source_csv_links=sum(bool(r.get('source_csv_path')) for r in images.values())
        result=dict(status='PASS',model=args.model,actual_backbone_weight_changed=True,
                    changed_layer=active.name,frozen_layer_unchanged=locked.name,
                    head_weight_changed=True,batch_normalization_frozen=all(not l.trainable for l in bn),
                    backbone_max_abs_update=float(np.max(np.abs(before-active.kernel.numpy()))),
                    trainable_backbone_convolutions=[l.name for l in trainable],
                    split_sha256=split['split_sha256'],split_and_file_hash_checks='PASS',
                    mapped_images=len(images),images_with_source_csv_link=source_csv_links,
                    limitation='Checks code-level mapping and actual weight updates; acquisition/session independence requires source evidence.')
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
