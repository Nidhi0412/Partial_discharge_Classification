"""Composed PNG augmentation, training only, with replayable source evidence.

Row/column masking, shifting, vertical deformation and noise are image-space
operations. No MHz calibration, waveform phase shift or acquisition is invented.
"""
import hashlib
import json
import math
from pathlib import Path
import threading
import uuid
from datetime import datetime, timezone

from dataset_manifest import CLASS_NAMES, digest
from legacy_augmentation import HISTORICAL_PARAMETERS, apply_historical

DEFAULTS=dict(row_mask_fraction=.15,column_mask_fraction=.15,
              image_shift_fraction=.15,image_stretch_fraction=.15,pixel_noise_std=.05)
PRIMARY_DEFAULTS=dict(row_mask_fraction=.15,column_mask_fraction=.15,
                     image_shift_fraction=.03,image_stretch_fraction=0.,pixel_noise_std=.05)
OPERATION_ORDER=['historical_image','vertical_image_stretch','horizontal_image_shift',
                 'row_band_mask','column_band_mask','scaled_pixel_noise']
_EXPORT_LOCK=threading.Lock()


def parameters(config):
    base=PRIMARY_DEFAULTS if config.get('augmentation_profile')=='scalogram-image' else DEFAULTS
    result={key:float(config.get(key,value)) for key,value in base.items()}
    for key,value in result.items():
        limit=.5 if key!='pixel_noise_std' else 1.
        if not math.isfinite(value) or not 0<=value<limit:
            raise ValueError(f'{key} must be finite and in [0,{limit})')
    return result


def operation_order(config):
    if not config.get('augment',False):return []
    profile=config.get('augmentation_profile','scalogram-image')
    if profile=='historical-image':return ['historical_image']
    if profile=='time-translation':return ['horizontal_translation']
    if profile not in {'scalogram-image','combined-image'}:raise ValueError('Unknown augmentation profile')
    p=parameters(config)
    result=['historical_image'] if profile=='combined-image' else []
    for key,name in [('image_stretch_fraction','vertical_image_stretch'),('image_shift_fraction','horizontal_image_shift'),
                     ('row_mask_fraction','row_band_mask'),('column_mask_fraction','column_band_mask'),
                     ('pixel_noise_std','scaled_pixel_noise')]:
        if p[key]:result.append(name)
    return result


def tensor_hash(image):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(image,dtype=np.float32).tobytes()).hexdigest()


def compose(image,seed,config,initial_stages=None):
    """Each stage receives the previous stage's tensor, never the original again."""
    import numpy as np
    from scipy import ndimage
    original=np.asarray(image,dtype=np.float32)
    if original.ndim!=3 or original.shape[-1]!=3 or not np.isfinite(original).all():
        raise ValueError('Expected finite HxWx3 RGB image')
    if original.min()<0 or original.max()>1:
        raise ValueError('PNG augmentation expects normalized pixels in [0,1]')
    p=parameters(config);rng=np.random.default_rng(int(seed))
    current=original.copy();stages=list(initial_stages or [])
    h,w,_=current.shape
    def add(name,output,params):
        nonlocal current
        output=np.clip(output,0.,1.).astype(np.float32)
        if output.shape!=original.shape or not np.isfinite(output).all():
            raise ValueError(f'{name} produced invalid pixels/shape')
        stages.append(dict(operation=name,parameters=params,input_sha256=tensor_hash(current),
                           output_sha256=tensor_hash(output),tensor=output.copy()))
        current=output
    if p['image_stretch_fraction']:
        factor=float(rng.uniform(1-p['image_stretch_fraction'],1+p['image_stretch_fraction']))
        matrix=np.diag([1./factor,1.,1.])
        offset=[(h-1)/2*(1-1./factor),0.,0.]
        stretched=ndimage.affine_transform(current,matrix,offset=offset,
                     output_shape=current.shape,order=1,mode='nearest',prefilter=False)
        add('vertical_image_stretch',stretched,dict(factor=factor,interpolation='linear',border='nearest',
             interpretation='Image-row deformation, not calibrated frequency scaling'))
    maximum=int(w*p['image_shift_fraction'])
    if maximum:
        shift=int(rng.integers(1,maximum+1))*(1 if rng.integers(0,2) else -1)
        shifted=ndimage.shift(current,(0,shift,0),order=0,mode='nearest',prefilter=False)
        add('horizontal_image_shift',shifted,dict(pixels=shift,border='nearest',wrap=False,
             interpretation='Image-column shift, not waveform phase manipulation'))
    for axis,key,name in [(0,'row_mask_fraction','row_band_mask'),(1,'column_mask_fraction','column_band_mask')]:
        size=current.shape[axis];maximum=int(size*p[key])
        if maximum:
            width=int(rng.integers(1,maximum+1));start=int(rng.integers(0,size-width+1))
            masked=current.copy();slices=[slice(None)]*3;slices[axis]=slice(start,start+width)
            masked[tuple(slices)]=0.
            add(name,masked,dict(axis=axis,start_pixel=start,width_pixels=width,fill_rgb=[0.,0.,0.],
                 interpretation='Image occlusion; rows/columns do not supply MHz/seconds calibration'))
    if p['pixel_noise_std']:
        noisy=current+rng.normal(0.,p['pixel_noise_std'],current.shape).astype(np.float32)*np.abs(current)
        add('scaled_pixel_noise',noisy,dict(stddev=p['pixel_noise_std'],scaled_by='absolute current RGB intensity',
             interpretation='Pixel perturbation, not laboratory waveform AWGN'))
    return current,stages


def save_example(source,original,augmented,stages,seed,config):
    import numpy as np
    from PIL import Image
    folder=Path(config['augmentation_export_dir'])/source['material']
    with _EXPORT_LOCK:
        folder.mkdir(parents=True,exist_ok=True)
        metadata=folder/(source['image_id']+'.json')
        if metadata.exists(): return
        original_png=folder/(source['image_id']+'_original.png')
        output_png=folder/(source['image_id']+'_augmented.png')
        exact=folder/(source['image_id']+'_augmented.npy')
        def png(path,value):Image.fromarray(np.uint8(np.rint(np.clip(value,0,1)*255))).save(path)
        png(original_png,original);png(output_png,augmented);np.save(exact,augmented,allow_pickle=False)
        exported=[]
        for index,stage in enumerate(stages):
            # Operation names stay in JSON; short filenames avoid Windows MAX_PATH.
            stem=folder/f"{source['image_id']}_s{index:02d}"
            png(stem.with_suffix('.png'),stage['tensor'])
            np.save(stem.with_suffix('.npy'),stage['tensor'],allow_pickle=False)
            exported.append(dict((k,v) for k,v in stage.items() if k!='tensor'))
            exported[-1].update(tensor_path=str(stem.with_suffix('.npy')),preview_path=str(stem.with_suffix('.png')))
        info=dict(source,augmentation_id=hashlib.sha256(f"{source['image_id']}|{int(seed)}|{config['augmentation_profile']}".encode()).hexdigest(),
                  augmentation_seed=int(seed),profile=config['augmentation_profile'],parameters=parameters(config),
                  original_preview=str(original_png),augmented_preview=str(output_png),
                  exact_generated_tensor=str(exact),tensor_sha256=digest(exact),stages=exported,
                  interpretation='PNG image-space augmentation, not additional independent recordings.',
                  note='Exact float32 examples from the training input pipeline; prefetch does not prove optimizer consumption. PNG previews are quantized.')
        temporary=metadata.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(info,indent=2),encoding='utf-8');temporary.replace(metadata)


def make_transform(tf,rows,config):
    import numpy as np
    if any(r.get('partition')!='train' for r in rows):
        raise ValueError('Only training parents may enter specialized augmentation')
    if config.get('augmentation_profile') not in {'scalogram-image','combined-image'}:
        raise ValueError('Unknown specialized image augmentation profile')
    settings=parameters(config)
    if config['augmentation_profile']=='scalogram-image' and not any(settings.values()):
        raise ValueError('Specialized augmentation has no enabled operation; use an explicit no-augmentation baseline')
    by_path={r['image_path']:r for r in rows}
    selected={r['image_id'] for material in CLASS_NAMES
              for r in sorted([r for r in rows if r['material']==material],key=lambda r:r['image_id'])
              [:config.get('augmentation_samples_per_class',5)]}
    generator=(tf.keras.preprocessing.image.ImageDataGenerator(**HISTORICAL_PARAMETERS)
               if config['augmentation_profile']=='combined-image' else None)
    ledger=None
    if config.get('augmentation_ledger_dir'):
        folder=Path(config['augmentation_ledger_dir']);folder.mkdir(parents=True,exist_ok=True)
        ledger=folder/f'generated_inputs_{uuid.uuid4().hex}.jsonl'
    def transform(image,path,seed):
        path=path.item() if hasattr(path,'item') else path
        source=by_path[path.decode('utf-8')]
        original=np.asarray(image,dtype=np.float32);current=original.copy();initial=[]
        if generator is not None:
            current,params=apply_historical(generator,current,int(seed))
            initial=[dict(operation='historical_image',parameters=params,input_sha256=tensor_hash(original),
                          output_sha256=tensor_hash(current),tensor=current.copy())]
        result,stages=compose(current,int(seed),config,initial)
        # All intermediate outputs must link to the next stage's input.
        if any(a['output_sha256']!=b['input_sha256'] for a,b in zip(stages,stages[1:])):
            raise RuntimeError('Augmentation composition chain is broken')
        if config.get('augmentation_export_dir') and source['image_id'] in selected:
            save_example(source,original,result,stages,seed,config)
        if ledger:
            record=dict(source,profile=config['augmentation_profile'],augmentation_seed=int(seed),
                        augmentation_id=hashlib.sha256(f"{source['image_id']}|{int(seed)}|{config['augmentation_profile']}".encode()).hexdigest(),
                        generation_occurrence_id=uuid.uuid4().hex,generated_utc=datetime.now(timezone.utc).isoformat(),
                        parameters=settings,scope='generated training inputs, including possible prefetch',
                        output_tensor_sha256=tensor_hash(result),
                        stages=[{k:v for k,v in stage.items() if k!='tensor'} for stage in stages])
            with _EXPORT_LOCK,ledger.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
        return result
    return transform
