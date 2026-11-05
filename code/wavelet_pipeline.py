"""Explicit, versioned scalograms and recording-to-image provenance.

Only run for newly approved raw-data processing. This implementation does not
reproduce historical positional dc1 labels or silently regenerate old images.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

from dataset_manifest import CLASS_NAMES, FIELDS, digest, write_csv

DEFAULT_CONFIG = dict(schema_version=2, wavelets=['db3'], levels=[1], decomposition_level=5,
                      scales=list(range(2,128)), cwt_wavelet='morl', representation='magnitude',
                      normalization='per_image', vmin=None, vmax=None, colormap='jet',
                      image_size=224, skiprows=21, image_row_order='high_frequency_top',
                      coordinate_note='Rows follow CWT scale index, not uniformly spaced physical frequency')


def read_signal(path, skiprows=21):
    import numpy as np
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        for _ in range(skiprows): next(reader,None)
        values = []
        for index,row in enumerate(reader,skiprows+1):
            if not row: continue
            try: values.append((float(row[0]),float(row[1])))
            except (ValueError,IndexError) as exc:
                raise ValueError(f'Invalid numeric TIME/CH1 row {index} in {path}') from exc
    array = np.asarray(values,dtype=float)
    if len(values)<2 or not np.isfinite(array).all():
        raise ValueError('At least two finite TIME/CH1 samples required')
    dt = np.diff(array[:,0])
    if np.any(dt<=0) or not np.allclose(dt,dt.mean(),rtol=1e-4,atol=abs(dt.mean())*1e-6):
        raise ValueError('Time samples must be strictly increasing and uniformly spaced')
    return array[:,0],array[:,1],float(dt.mean())


def transform(signal, original_dt, wavelet, true_level, config):
    import numpy as np
    import pywt
    if original_dt<=0 or not np.isfinite(original_dt): raise ValueError('Invalid sampling interval')
    signal = np.asarray(signal,dtype=float)
    if signal.ndim!=1 or not np.isfinite(signal).all(): raise ValueError('Expected finite 1D signal')
    maximum=pywt.dwt_max_level(len(signal),pywt.Wavelet(wavelet).dec_len)
    depth=config['decomposition_level']
    if not 1<=true_level<=depth<=maximum:
        raise ValueError(f'Invalid detail/decomposition level; maximum useful depth {maximum}')
    coeffs=pywt.wavedec(signal,wavelet,level=depth)
    detail=coeffs[-true_level]  # [cA_depth,cD_depth,...,cD1]
    effective_dt=original_dt*(2**true_level)
    scales=np.asarray(config['scales'],dtype=float)
    if scales.ndim!=1 or len(scales)<2 or np.any(scales<=0) or np.any(np.diff(scales)<=0):
        raise ValueError('CWT scales must be positive and strictly increasing')
    cwt,freq=pywt.cwt(detail,scales,config['cwt_wavelet'],sampling_period=effective_dt)
    image_values=np.abs(cwt)
    if config['representation']=='energy': image_values=image_values**2
    elif config['representation']!='magnitude': raise ValueError('Unknown scalogram representation')
    if not np.isfinite(image_values).all():raise ValueError('Nonfinite scalogram values; inspect signal magnitude')
    if config['normalization']=='per_image': low,high=float(image_values.min()),float(image_values.max())
    elif config['normalization']=='fixed':
        low,high=config['vmin'],config['vmax']
        if low is None or high is None or not high>low: raise ValueError('Fixed normalization requires vmin<vmax')
    else: raise ValueError('Unknown normalization')
    normalized=np.zeros_like(image_values) if high==low else np.clip((image_values-low)/(high-low),0,1)
    metadata=dict(wavelet=wavelet,true_dwt_level=true_level,decomposition_level=depth,
                  original_dt_seconds=original_dt,detail_dt_seconds=effective_dt,
                  detail_samples=len(detail),cwt_row_frequency_hz=freq.tolist(),scales=scales.tolist(),
                  representation=config['representation'],normalization_min=low,normalization_max=high,
                  interpretation='Frequency is for the decimated coefficient sequence; not an RF band-center mapping')
    return normalized,detail,metadata


def image_from_values(values, config):
    import numpy as np
    import matplotlib
    from PIL import Image
    rgb=matplotlib.colormaps[config['colormap']](values,bytes=True)[...,:3]
    return Image.fromarray(np.asarray(rgb,dtype='uint8')).resize(
        (config['image_size'],config['image_size']),Image.Resampling.BILINEAR)


def generate(raw_manifest,output_dir,config):
    import numpy as np
    if config.get('image_row_order')!='high_frequency_top':raise ValueError('Unsupported image row ordering')
    if not config.get('wavelets') or not config.get('levels'):
        raise ValueError('Specify at least one wavelet and true detail level')
    if len(set(config['wavelets']))!=len(config['wavelets']) or len(set(config['levels']))!=len(config['levels']):
        raise ValueError('Repeated wavelet/level variants would overwrite generated outputs')
    raw_manifest=Path(raw_manifest).resolve()
    with raw_manifest.open(newline='',encoding='utf-8-sig') as stream: rows=list(csv.DictReader(stream))
    if not rows: raise ValueError('Raw manifest empty')
    recordings=set()
    for r in rows:
        if not r.get('recording_id') or r.get('material') not in CLASS_NAMES or not r.get('csv_path'):
            raise ValueError('Raw manifest requires recording_id, material, csv_path; electrode/session_id optional')
        if r['recording_id'] in recordings: raise ValueError('Raw manifest has repeated recording_id')
        recordings.add(r['recording_id'])
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'generation_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    config_sha=digest(out/'generation_config.json');images=[]
    try:
        for r in rows:
            p=Path(r['csv_path']);p=p.resolve() if p.is_absolute() else (raw_manifest.parent/p).resolve()
            time,signal,dt=read_signal(p,config['skiprows'])
            source_sha=digest(p)
            token=hashlib.sha256(r['recording_id'].encode()).hexdigest()[:24]
            for wavelet in config['wavelets']:
                for level in config['levels']:
                    values,detail,metadata=transform(signal,dt,wavelet,level,config)
                    image_id=f'{token}_{wavelet}_D{level}'
                    image=out/(image_id+'.png');image_from_values(values,config).save(image)
                    np.save(out/(image_id+'_coefficients.npy'),detail,allow_pickle=False)
                    (out/(image_id+'_coordinates.json')).write_text(json.dumps(metadata,indent=2),encoding='utf-8')
                    images.append(dict(image_id=image_id,image_path=str(image),recording_id=r['recording_id'],
                        material=r['material'],electrode=r.get('electrode',''),session_id=r.get('session_id',''),
                        image_sha256=digest(image),source_csv_path=str(p),source_csv_sha256=source_sha,
                        wavelet=wavelet,true_dwt_level=level,generation_config_sha256=config_sha))
        write_csv(out/'images.csv',images,FIELDS)
        (out/'status.json').write_text(json.dumps(dict(status='COMPLETE',recordings=len(rows),images=len(images))),encoding='utf-8')
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(status='FAILED',error=str(exc))),encoding='utf-8')
        raise
    return out/'images.csv'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-manifest',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--wavelets',nargs='+',default=['db3'])
    parser.add_argument('--levels',nargs='+',type=int,default=[1])
    parser.add_argument('--representation',choices=['magnitude','energy'],default='magnitude')
    parser.add_argument('--config',help='Optional explicit generation JSON; no class-dependent band selection')
    args=parser.parse_args(argv)
    config=dict(DEFAULT_CONFIG)
    if args.config: config.update(json.loads(Path(args.config).read_text(encoding='utf-8')))
    else: config.update(wavelets=args.wavelets,levels=args.levels,representation=args.representation)
    print(generate(args.raw_manifest,args.output_dir,config))


if __name__=='__main__':main()
