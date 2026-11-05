"""Restore the effective historical ImageDataGenerator path, on training only.

Parameters are evidenced in both archived model_training files' get_data_generator.
Custom frequency/phase layers declared elsewhere are not part of that path.
"""
import json
from pathlib import Path
import threading

from dataset_manifest import CLASS_NAMES, digest

HISTORICAL_PARAMETERS = dict(rotation_range=10, width_shift_range=.1,
    height_shift_range=.1, zoom_range=.1, horizontal_flip=True,
    brightness_range=[.9,1.1], fill_mode='nearest', rescale=1./255)
_RNG_LOCK = threading.Lock()


def apply_historical(generator, image, seed):
    """Shared exact historical operators, returning actual sampled parameters."""
    import numpy as np
    raw=np.asarray(image,dtype=np.float32)*255.
    with _RNG_LOCK:
        previous=np.random.get_state()
        try: params=generator.get_random_transform(raw.shape,seed=int(seed))
        finally: np.random.set_state(previous)
    result=generator.standardize(generator.apply_transform(raw.copy(),params)).astype(np.float32)
    return result,{k:(v.item() if hasattr(v,'item') else v) for k,v in params.items()}


def make_transform(tf, rows, config):
    import numpy as np
    from PIL import Image
    generator = tf.keras.preprocessing.image.ImageDataGenerator(**HISTORICAL_PARAMETERS)
    by_path = {r['image_path']:r for r in rows}
    export = config.get('augmentation_export_dir')
    # Bound saved examples, not the number of augmented training inputs.
    selected = set(r['image_id'] for material in CLASS_NAMES
                   for r in sorted([r for r in rows if r['material']==material],key=lambda r:r['image_id'])
                   [:config.get('augmentation_samples_per_class',5)])

    def transform(image, path, seed):
        raw=np.asarray(image,dtype=np.float32)*255.
        seed=int(seed)
        augmented,params=apply_historical(generator,image,seed)
        if augmented.shape!=image.shape or not np.isfinite(augmented).all():
            raise ValueError('Historical augmentation produced invalid pixels/shape')
        if export:
            path=path.item() if hasattr(path,'item') else path
            source=by_path[path.decode('utf-8')]
            if source['image_id'] in selected:
                folder=Path(export)/source['material'];folder.mkdir(parents=True,exist_ok=True)
                metadata=folder/(source['image_id']+'.json')
                if not metadata.exists():
                    original=folder/(source['image_id']+'_original.png')
                    output=folder/(source['image_id']+'_augmented.png')
                    tensor=folder/(source['image_id']+'_augmented.npy')
                    Image.fromarray(np.uint8(np.clip(np.rint(raw),0,255))).save(original)
                    Image.fromarray(np.uint8(np.clip(np.rint(augmented*255),0,255))).save(output)
                    np.save(tensor,augmented,allow_pickle=False)
                    info=dict(source,augmentation_seed=seed,profile='historical-image',
                              transform_parameters={k:(v.item() if hasattr(v,'item') else v) for k,v in params.items()},
                              original_preview=str(original),augmented_preview=str(output),
                              exact_generated_tensor=str(tensor),tensor_sha256=digest(tensor),
                              note='Generated training-pipeline example; prefetch may run before batch consumption. '
                                   'PNG is quantized; NPY is the exact float32 output. Not a new independent recording.')
                    with metadata.open('x',encoding='utf-8') as stream:json.dump(info,stream,indent=2)
        return augmented
    return transform
