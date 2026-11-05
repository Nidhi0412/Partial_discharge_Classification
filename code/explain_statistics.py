"""Recording-aware descriptive Grad-CAM diagnostics; no physical-energy claims."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from dataset_manifest import digest


def validate_plan(config):
    required={'seed','bootstrap_draws','mass_fraction','mask_fraction','random_masks',
              'shift_pixels','regions','primary_metric','mask_fill','scope','target','min_clusters'}
    if required-set(config):raise ValueError(f'Missing plan fields: {required-set(config)}')
    if any(type(config[key]) is not int for key in ['seed','bootstrap_draws','random_masks','min_clusters']):
        raise ValueError('Seed/count configuration must use integers')
    if not 0<config['mass_fraction']<=1 or not 0<config['mask_fraction']<1:
        raise ValueError('Mass/mask fractions out of range')
    if config['bootstrap_draws']<100 or config['random_masks']<2:
        raise ValueError('Use >=100 bootstrap draws and >=2 random mask controls')
    if config['min_clusters']<2:raise ValueError('At least two independent clusters required for any interval')
    if not config['shift_pixels'] or any(type(v) is not int or v==0 for v in config['shift_pixels']):
        raise ValueError('Specify nonzero integer image-axis translations')
    if config['mask_fill']!='per_image_channel_mean':raise ValueError('Unsupported masking baseline')
    if config['primary_metric']!='faithfulness_margin':raise ValueError('Unsupported primary metric')
    if config['target'] not in ['true','Nomex','Teflon']:raise ValueError('Invalid frozen explanation target')
    names=set()
    for r in config['regions']:
        if r['name'] in names:raise ValueError('Duplicate region label')
        names.add(r['name'])
        if not 0<=r['x0']<r['x1']<=1 or not 0<=r['y0']<r['y1']<=1:
            raise ValueError('Region bounds must be fractional image coordinates')
    return config


def freeze_plan(config_path,output):
    config=validate_plan(json.loads(Path(config_path).read_text(encoding='utf-8')))
    canonical=json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False)
    plan=dict(config=config,config_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
              frozen_utc=datetime.now(timezone.utc).isoformat(),source_config=str(Path(config_path).resolve()),
              code_sha256=digest(__file__),
              pipeline_sha256=digest(Path(__file__).with_name('explain_pipeline.py')),
              note='A local analysis specification, not external preregistration or proof of prospective intent')
    with Path(output).open('x',encoding='utf-8') as stream:json.dump(plan,stream,indent=2,allow_nan=False)
    return plan


def load_plan(path):
    plan=json.loads(Path(path).read_text(encoding='utf-8'))
    config=validate_plan(plan['config'])
    canonical=json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest()!=plan['config_sha256']:
        raise ValueError('Frozen analysis specification was changed')
    if digest(__file__)!=plan['code_sha256']:
        raise ValueError('Statistics implementation changed after plan freeze; version/freeze a new plan before map inspection')
    if digest(Path(__file__).with_name('explain_pipeline.py'))!=plan['pipeline_sha256']:
        raise ValueError('Explanation pipeline changed after plan freeze')
    return plan


def heatmap_metrics(heat,plan):
    import numpy as np
    h=np.asarray(heat,dtype=float)
    if h.ndim!=2 or not np.isfinite(h).all() or (h<0).any():
        raise ValueError('Expected finite nonnegative 2D activation map')
    result=dict(map_is_zero=int(h.sum()==0),normalized_entropy=None,mass_area_fraction=None,
                centroid_x=None,centroid_y=None)
    result.update({'region_'+r['name']:None for r in plan['regions']})
    if not h.size or h.sum()==0:return result
    mass=h/h.sum();nonzero=mass[mass>0]
    entropy=float(-(nonzero*np.log(nonzero)).sum())
    result['normalized_entropy']=entropy/math.log(h.size) if h.size>1 else 0.
    cumulative=np.cumsum(np.sort(mass.ravel())[::-1])
    cells=min(h.size,int(np.searchsorted(cumulative,plan['mass_fraction'],side='left'))+1)
    result['mass_area_fraction']=cells/h.size
    ys=(np.arange(h.shape[0])+.5)/h.shape[0];xs=(np.arange(h.shape[1])+.5)/h.shape[1]
    result['centroid_x']=float((mass*xs[None,:]).sum());result['centroid_y']=float((mass*ys[:,None]).sum())
    for r in plan['regions']:
        mask=(ys[:,None]>=r['y0'])&(ys[:,None]<r['y1'])&(xs[None,:]>=r['x0'])&(xs[None,:]<r['x1'])
        result['region_'+r['name']]=float(mass[mask].sum())
    return result


def cosine_similarity(a,b):
    import numpy as np
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    if a.size!=b.size or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid aligned maps')
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.clip(np.dot(a,b)/denominator,-1.,1.)) if denominator else None


def shift_image(image,shift):
    """Bounded horizontal translation with per-image channel-mean edge fill."""
    import numpy as np
    image=np.asarray(image)
    if image.ndim!=3 or abs(shift)>=image.shape[1] or not shift:raise ValueError('Invalid image/shift')
    result=np.broadcast_to(image.mean(axis=(0,1)),image.shape).copy()
    if shift>0:result[:,shift:,:]=image[:,:-shift,:]
    else:result[:,:shift,:]=image[:,-shift:,:]
    return result


def aligned_maps(original,shifted,shift):
    if not shift or abs(shift)>=original.shape[1]:raise ValueError('Invalid alignment shift')
    return (original[:,:-shift],shifted[:,shift:]) if shift>0 else (original[:,-shift:],shifted[:,:shift])


def masking_inputs(image,heat,plan,seed):
    import numpy as np
    image=np.asarray(image,dtype=float);heat=np.asarray(heat,dtype=float)
    if image.ndim!=3 or heat.shape!=image.shape[:2]:raise ValueError('Mask/map shape mismatch')
    n=heat.size;k=max(1,min(n-1,int(math.ceil(n*plan['mask_fraction']))))
    # Stable tie-breaking. Flat maps are excluded by the caller, not interpreted.
    top=np.argsort(-heat.ravel(),kind='stable')[:k]
    rng=np.random.default_rng(seed)
    indices=[top]+[rng.choice(n,k,replace=False) for _ in range(plan['random_masks'])]
    fill=image.mean(axis=(0,1));masked=[]
    for selected in indices:
        copy=image.copy().reshape(n,image.shape[-1]);copy[selected]=fill
        masked.append(copy.reshape(image.shape))
    return np.stack(masked),k


def faithfulness_metrics(baseline,scores):
    import numpy as np
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or len(values)<3 or not np.isfinite(values).all():raise ValueError('Need attended mask plus random controls')
    attended=float(baseline-values[0]);random_drops=baseline-values[1:]
    return dict(attended_score_drop=attended,random_score_drop=float(random_drops.mean()),
                faithfulness_margin=attended-float(random_drops.mean()),
                random_control_sd=float(random_drops.std(ddof=1)))


def summarize(rows,plan,group_by):
    """Average variants within recording; resample independent recording/session clusters.

    Separate summaries for each model/material/target. No class-difference p-values.
    """
    import numpy as np
    if not rows:raise ValueError('No diagnostic observations')
    image_ids=[r['image_id'] for r in rows]
    if len(image_ids)!=len(set(image_ids)):raise ValueError('Repeated image observations cannot be treated as independent')
    if group_by not in {'recording_id','session_id','group_id'}:raise ValueError('Invalid uncertainty grouping')
    metadata={'image_id','recording_id','session_id','material','target','model_type','checkpoint_sha256',
              'image_sha256','map_path','plan_sha256','partition','fold','group_id'}
    metrics=sorted(set().union(*(set(r) for r in rows))-metadata)
    recording_groups=defaultdict(list)
    for r in rows:
        if not r.get('recording_id') or not r.get(group_by):raise ValueError('Missing independent grouping metadata')
        key=(r['model_type'],r['material'],r['target'],r['recording_id'])
        recording_groups[key].append(r)
    recordings=[]
    for key,values in sorted(recording_groups.items()):
        if len({r.get('session_id','') for r in values})!=1:raise ValueError('Recording session metadata conflicts')
        if len({(r['fold'],r['checkpoint_sha256']) for r in values})!=1:
            raise ValueError('A recording is repeated across fold/checkpoint estimates')
        row=dict(model_type=key[0],material=key[1],target=key[2],recording_id=key[3],
                 session_id=values[0].get('session_id',''),group_id=values[0].get('group_id',key[3]),images=len(values))
        for metric in metrics:
            valid=[float(v[metric]) for v in values if v.get(metric) is not None]
            if any(not math.isfinite(v) for v in valid):raise ValueError('Nonfinite diagnostic')
            row[metric]=sum(valid)/len(valid) if valid else None
            row[metric+'_valid_images']=len(valid)
        recordings.append(row)
    strata=defaultdict(list)
    for r in recordings:strata[(r['model_type'],r['material'],r['target'])].append(r)
    output=[]
    for key,values in sorted(strata.items()):
        for metric in metrics:
            valid=[r for r in values if r[metric] is not None]
            clusters=defaultdict(list)
            for r in valid:clusters[r[group_by]].append(r[metric])
            mean=sum(r[metric] for r in valid)/len(valid) if valid else None
            low=high=None
            if len(clusters)>=plan['min_clusters']:
                rng=np.random.default_rng(plan['seed']);keys=sorted(clusters);draws=[]
                for _ in range(plan['bootstrap_draws']):
                    sample=[v for idx in rng.integers(0,len(keys),len(keys)) for v in clusters[keys[idx]]]
                    draws.append(sum(sample)/len(sample))
                low,high=[float(x) for x in np.quantile(draws,[.025,.975])]
            output.append(dict(model_type=key[0],material=key[1],target=key[2],metric=metric,
                mean=mean,ci95_lower=low,ci95_upper=high,recordings_total=len(values),
                recordings_valid=len(valid),recordings_excluded=len(values)-len(valid),
                images_total=sum(r['images'] for r in values),
                images_valid=sum(r[metric+'_valid_images'] for r in values),independent_clusters=len(clusters),
                cluster_unit=group_by,interval_status='DESCRIPTIVE_PERCENTILE' if low is not None else 'INSUFFICIENT_VALID_CLUSTERS'))
    return recordings,output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('freeze');p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    freeze_plan(args.config,args.output)
    print('Frozen local specification. Freeze before inspecting final maps; do not tune against holdout explanations.')


if __name__=='__main__':main()
