"""Class-specific binary Grad-CAM using the saved model and trained preprocessing."""
import argparse
import json
from pathlib import Path

from dataset_manifest import CLASS_NAMES, digest, write_csv
from training_pipeline import make_dataset, save_json, tensorflow


def binary_target(probability, class_index):
    if class_index not in [0,1]:raise ValueError('Binary class index must be 0 or 1')
    return probability if class_index==1 else 1-probability


def compute_gradcam(grad_model,images,target):
    tf=tensorflow()
    with tf.GradientTape() as tape:
        activation,probabilities=grad_model(images,training=False)
        score=binary_target(probabilities[:,0],target)
    gradients=tape.gradient(score,activation)
    if gradients is None:raise RuntimeError('No gradient for the selected convolution layer')
    weights=tf.reduce_mean(gradients,axis=(0,1,2))
    heat=tf.maximum(tf.reduce_sum(activation[0]*weights,axis=-1),0)
    return tf.math.divide_no_nan(heat,tf.reduce_max(heat)).numpy(),float(probabilities[0,0])


def image_diagnostics(model,grad_model,image,heat,target,plan,seed):
    import numpy as np
    from explain_statistics import (aligned_maps,cosine_similarity,faithfulness_metrics,
                                    heatmap_metrics,masking_inputs,shift_image)
    tf=tensorflow()
    result=heatmap_metrics(heat,plan)
    result.update(stability_cosine=None,stability_prediction_change_rate=None,
                  attended_score_drop=None,random_score_drop=None,faithfulness_margin=None,
                  random_control_sd=None)
    upsample=lambda h:tf.image.resize(h[None,...,None],image.shape[:2],method='bilinear')[0,:,:,0].numpy()
    full=upsample(heat);base_probability=float(model(image[None,...],training=False)[0,0])
    baseline=float(binary_target(base_probability,target))
    stability=[];changes=[]
    for shift in plan['shift_pixels']:
        perturbed=shift_image(image,shift)
        shifted_heat,probability=compute_gradcam(grad_model,tf.convert_to_tensor(perturbed[None,...],dtype=tf.float32),target)
        a,b=aligned_maps(full,upsample(shifted_heat),shift)
        cosine=cosine_similarity(a,b)
        if cosine is not None:stability.append(cosine)
        changes.append(int((probability>=.5)!=(base_probability>=.5)))
    if stability:result['stability_cosine']=float(np.mean(stability))
    result['stability_prediction_change_rate']=float(np.mean(changes))
    # A flat map has no distinguished attended region: no arbitrary top-mask claim.
    result['map_is_flat']=int(float(np.ptp(full))<=1e-12)
    if result['map_is_flat']:result['stability_cosine']=None
    if not result['map_is_flat']:
        masked,_=masking_inputs(image,full,plan,seed)
        scores=model(tf.convert_to_tensor(masked,dtype=tf.float32),training=False).numpy()[:,0]
        result.update(faithfulness_metrics(baseline,binary_target(scores,target)))
    return result


def validate_analysis_cohort(run,checkpoint,prediction_path,rows,config):
    """Statistics must use the complete pipeline endpoint, not selected nice maps."""
    from dataset_manifest import load_splits
    run=Path(run).resolve()
    status=json.loads((run/'status.json').read_text(encoding='utf-8'))
    if status['status']!='COMPLETE':raise ValueError('Statistics require a completed run')
    splits,metadata=load_splits(config['split_manifest'])
    if metadata['split_sha256']!=config['split_sha256']:raise ValueError('Run split differs')
    if config['purpose']=='final_fit':
        expected_file=run/'holdout_evaluation/predictions.csv'
        expected=[r for r in splits if r['fold']=='1' and r['partition']=='test']
        evaluation=json.loads((run/'holdout_evaluation/status.json').read_text(encoding='utf-8'))
        if evaluation['status']!='COMPLETE':raise ValueError('Holdout evaluation did not complete')
    elif config['purpose']=='development_cv':
        expected_file=checkpoint.parent/'predictions.csv'
        fold=checkpoint.parent.name.removeprefix('fold_')
        expected=[r for r in splits if r['fold']==fold and r['partition']=='validation']
    else:raise ValueError('Unknown experiment endpoint')
    if Path(prediction_path).resolve()!=expected_file.resolve():
        raise ValueError('Statistics require the pipeline-produced full endpoint prediction file')
    if len(rows)!=len(expected) or {r['image_id'] for r in rows}!={r['image_id'] for r in expected}:
        raise ValueError('Statistics cohort is incomplete or selected')
    indexed={r['image_id']:r for r in expected}
    for r in rows:
        if any(r.get(key,'')!=indexed[r['image_id']].get(key,'') for key in
               ['recording_id','session_id','material','image_sha256','partition','fold']):
            raise ValueError('Prediction metadata differs from the split manifest')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--checkpoint',required=True,help='Exact .keras checkpoint from this run')
    parser.add_argument('--predictions',required=True,help='Explicit selection from traced predictions; contains no guessed label order')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--target',choices=['Nomex','Teflon','true'])
    parser.add_argument('--count',type=int,help='Visual sample size; defaults to 10 without a statistics plan')
    parser.add_argument('--analysis-plan',help='Frozen plan from explain_statistics.py; analyzes the entire supplied prediction cohort')
    args=parser.parse_args(argv)
    if args.count is not None and args.count<1:parser.error('count must be positive')
    if args.analysis_plan and args.count is not None:
        parser.error('Statistics use the full declared prediction cohort; do not truncate with --count')
    from evaluation_tools import read_predictions
    import numpy as np
    config=json.loads((Path(args.run_dir)/'run_config.json').read_text(encoding='utf-8'))
    predictions=read_predictions(args.predictions)
    checkpoint=Path(args.checkpoint).resolve()
    checkpoint.relative_to(Path(args.run_dir).resolve())
    cp_hash=digest(checkpoint)
    plan=None
    if args.analysis_plan:
        from explain_statistics import load_plan
        plan=load_plan(args.analysis_plan)
        if any(abs(v)>=config['image_size'] for v in plan['config']['shift_pixels']):
            raise ValueError('Frozen translation exceeds input image width')
        if args.target is not None and args.target!=plan['config']['target']:
            raise ValueError('Explanation target differs from frozen plan')
        args.target=plan['config']['target']
    else:args.target=args.target or 'true'
    selected=sorted(predictions,key=lambda r:r['image_id'])
    if plan is None:selected=selected[:args.count or 10]
    if any(r['checkpoint_sha256']!=cp_hash for r in selected):
        raise ValueError('Selected prediction rows do not belong to this checkpoint; select one fold explicitly')
    if any(r['split_sha256']!=config['split_sha256'] or r['model_type']!=config['model_type'] for r in selected):
        raise ValueError('Prediction rows differ from the selected experiment')
    if any(digest(r['image_path'])!=r['image_sha256'] for r in selected):raise ValueError('Selected input changed')
    if plan and any(not r.get('recording_id') or not r.get(config['group_by']) for r in selected):
        raise ValueError('Statistics require trusted recording/session grouping metadata')
    if plan:validate_analysis_cohort(args.run_dir,checkpoint,args.predictions,selected,config)
    tf=tensorflow();model=tf.keras.models.load_model(str(checkpoint),compile=False)
    layer='block5_conv3' if config['model_type']=='vgg16' else 'conv5_block3_out'
    grad_model=tf.keras.Model(model.inputs,[model.get_layer(layer).output,model.output])
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    save_json(out/'selection.json',dict(target=args.target,checkpoint_sha256=cp_hash,
        selection='Full supplied prediction cohort' if plan else 'First count rows sorted by image_id; descriptive visual sample only',
        predictions_sha256=digest(args.predictions),rows=selected))
    if plan:
        save_json(out/'analysis_plan.json',plan)
        save_json(out/'analysis_context.json',dict(plan_file_sha256=digest(args.analysis_plan),
            target=args.target,model_type=config['model_type'],cluster_unit=config['group_by'],
            selection='All rows in the hashed prediction cohort, not just correct predictions',
            checkpoint_sha256=cp_hash,statistics_scope='Descriptive conditional-model image-space diagnostics',
            limitation='Masking/translation may be out of distribution; no physical energy or RF-band inference. Intervals exclude training/model-selection uncertainty.'))
    dataset,ordered=make_dataset(selected,dict(config,batch_size=1),False)
    summary=[];diagnostics=[]
    for (images,_),row in zip(dataset,ordered):
        target=CLASS_NAMES.index(row['material'] if args.target=='true' else args.target)
        heat,probability=compute_gradcam(grad_model,images,target)
        if plan and abs(probability-float(row['p_teflon']))>1e-5:
            raise ValueError('Recomputed checkpoint probability differs from stored prediction')
        filename=out/(row['image_id']+'.npy');np.save(filename,heat,allow_pickle=False)
        summary.append(dict(image_id=row['image_id'],recording_id=row['recording_id'],
                            target=CLASS_NAMES[target],p_teflon=probability,
                            mean_normalized_activation=float(heat.mean()),heatmap_path=str(filename),
                            checkpoint_sha256=cp_hash))
        if plan:
            import hashlib
            from explain_statistics import summarize
            seed=plan['config']['seed']+int(hashlib.sha256(row['image_id'].encode()).hexdigest()[:8],16)
            values=image_diagnostics(model,grad_model,images.numpy()[0],heat,target,plan['config'],seed)
            values.update(image_id=row['image_id'],recording_id=row['recording_id'],
                session_id=row.get('session_id',''),material=row['material'],target=CLASS_NAMES[target],
                group_id=row['group_id'],
                model_type=config['model_type'],checkpoint_sha256=cp_hash,image_sha256=row['image_sha256'],
                map_path=str(filename),plan_sha256=plan['config_sha256'],partition=row['partition'],fold=row['fold'])
            diagnostics.append(values)
    write_csv(out/'activation_summary.csv',summary,list(summary[0]))
    if plan:
        recordings,statistics=summarize(diagnostics,plan['config'],config['group_by'])
        write_csv(out/'image_diagnostics.csv',diagnostics,list(diagnostics[0]))
        write_csv(out/'recording_diagnostics.csv',recordings,list(recordings[0]))
        write_csv(out/'explainability_statistics.csv',statistics,list(statistics[0]))
    print(f'Saved {len(summary)} numerical class-specific maps. Grad-CAM is descriptive, not proof of a physical mechanism.')


if __name__=='__main__':main()
