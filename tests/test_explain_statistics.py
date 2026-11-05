"""Synthetic checks only; no research maps, checkpoints or model training."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from explain_statistics import (aligned_maps,cosine_similarity,faithfulness_metrics,
                               freeze_plan,heatmap_metrics,load_plan,masking_inputs,
                               shift_image,summarize,validate_plan)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy required for map-statistic tests')
class StatisticsTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np=np
        self.plan=json.loads((ROOT/'config/explainability_plan.json').read_text())
        self.plan.update(bootstrap_draws=100,min_clusters=2)

    def test_spike_and_uniform_concentration(self):
        np=self.np;spike=np.zeros((10,10));spike[1,2]=1
        a=heatmap_metrics(spike,self.plan);b=heatmap_metrics(np.ones((10,10)),self.plan)
        self.assertAlmostEqual(a['normalized_entropy'],0)
        self.assertAlmostEqual(a['mass_area_fraction'],.01)
        self.assertAlmostEqual(b['normalized_entropy'],1)
        self.assertAlmostEqual(b['mass_area_fraction'],.8)

    def test_scale_invariant_not_raw_intensity(self):
        heat=self.np.arange(1,101).reshape(10,10)
        self.assertEqual(heatmap_metrics(heat,self.plan),heatmap_metrics(heat*100,self.plan))

    def test_zero_maps_are_undefined_not_fake_localization(self):
        m=heatmap_metrics(self.np.zeros((4,4)),self.plan)
        self.assertEqual(m['map_is_zero'],1)
        self.assertIsNone(m['normalized_entropy']);self.assertIsNone(m['centroid_x'])

    def test_nonfinite_maps_rejected(self):
        with self.assertRaises(ValueError):heatmap_metrics([[float('nan')]],self.plan)

    def test_regions_and_centroid_use_pixel_centers(self):
        heat=self.np.zeros((10,10));heat[1,2]=1;m=heatmap_metrics(heat,self.plan)
        self.assertAlmostEqual(m['centroid_x'],.25);self.assertAlmostEqual(m['centroid_y'],.15)
        self.assertEqual(m['region_left_time_third'],1)
        self.assertEqual(m['region_right_time_third'],0)

    def test_alignment_of_both_shift_directions(self):
        x=self.np.arange(60).reshape(4,5,3).astype(float)
        for shift in [-1,1]:
            changed=shift_image(x,shift)
            a,b=aligned_maps(x[:,:,0],changed[:,:,0],shift)
            self.np.testing.assert_array_equal(a,b)
            self.assertAlmostEqual(cosine_similarity(a,b),1)

    def test_zero_cosine_is_undefined(self):
        self.assertIsNone(cosine_similarity([0,0],[1,2]))

    def test_attended_and_random_masks_have_equal_budget(self):
        np=self.np;image=np.arange(300,dtype=float).reshape(10,10,3)
        heat=np.arange(100).reshape(10,10)
        masks,k=masking_inputs(image,heat,self.plan,42)
        self.assertEqual(len(masks),self.plan['random_masks']+1)
        for value in masks:self.assertEqual(int(np.any(value!=image,axis=-1).sum()),k)
        self.np.testing.assert_array_equal(masks,masking_inputs(image,heat,self.plan,42)[0])

    def test_faithfulness_is_paired_with_random_baseline(self):
        m=faithfulness_metrics(.9,[.4,.8,.7])
        self.assertAlmostEqual(m['attended_score_drop'],.5)
        self.assertAlmostEqual(m['random_score_drop'],.15)
        self.assertAlmostEqual(m['faithfulness_margin'],.35)

    def make_rows(self):
        rows=[]
        for rec,values in [('a',[0.,1.]),('b',[1.]),('c',[1.])]:
            for index,v in enumerate(values):
                rows.append(dict(image_id=rec+str(index),recording_id=rec,session_id='s'+rec,
                    model_type='vgg16',material='Nomex',target='Nomex',checkpoint_sha256='cp',
                    partition='test',fold='1',concentration=v,undefined=None))
        return rows

    def test_variants_are_averaged_before_bootstrap(self):
        records,stats=summarize(self.make_rows(),self.plan,'recording_id')
        row=next(r for r in stats if r['metric']=='concentration')
        self.assertEqual(row['recordings_valid'],3)
        self.assertAlmostEqual(row['mean'],(0.5+1+1)/3)
        self.assertNotAlmostEqual(row['mean'],.75)
        self.assertIsNotNone(row['ci95_lower'])

    def test_too_few_sessions_get_no_interval(self):
        rows=self.make_rows()
        for r in rows:r['session_id']='one_session'
        _,stats=summarize(rows,self.plan,'session_id')
        self.assertTrue(all(r['ci95_lower'] is None for r in stats))
        undefined=next(r for r in stats if r['metric']=='undefined')
        self.assertEqual(undefined['recordings_excluded'],3)

    def test_duplicate_images_and_repeated_fold_models_rejected(self):
        rows=self.make_rows()
        with self.assertRaises(ValueError):summarize(rows+[rows[0]],self.plan,'recording_id')
        rows[1]['fold']='2'
        with self.assertRaises(ValueError):summarize(rows,self.plan,'recording_id')

    def test_freeze_detects_edits_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            config=Path(d)/'config.json';config.write_text(json.dumps(self.plan))
            frozen=Path(d)/'plan.json';freeze_plan(config,frozen)
            self.assertEqual(load_plan(frozen)['config'],self.plan)
            with self.assertRaises(FileExistsError):freeze_plan(config,frozen)
            data=json.loads(frozen.read_text());data['config']['seed']=99;frozen.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_plan(frozen)

    def test_complete_endpoint_cohort_required(self):
        from explain_pipeline import validate_analysis_cohort
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'status.json').write_text('{"status":"COMPLETE"}')
            endpoint=run/'holdout_evaluation';endpoint.mkdir()
            (endpoint/'status.json').write_text('{"status":"COMPLETE"}')
            rows=[dict(image_id=str(i),recording_id='r'+str(i),session_id='s',material='Nomex',
                       image_sha256='hash'+str(i),partition='test',fold='1') for i in range(2)]
            config=dict(purpose='final_fit',split_manifest='splits.csv',split_sha256='split')
            with patch('dataset_manifest.load_splits',return_value=(rows,{'split_sha256':'split'})):
                validate_analysis_cohort(run,run/'best.keras',endpoint/'predictions.csv',rows,config)
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    validate_analysis_cohort(run,run/'best.keras',endpoint/'predictions.csv',rows[:1],config)
                with self.assertRaisesRegex(ValueError,'full endpoint'):
                    validate_analysis_cohort(run,run/'best.keras',run/'selected.csv',rows,config)


@unittest.skipUnless(importlib.util.find_spec('tensorflow') and importlib.util.find_spec('numpy'),
                     'TensorFlow/NumPy unavailable')
class SyntheticRuntimeTest(unittest.TestCase):
    def test_actual_gradients_stability_masking_and_zero_map(self):
        import numpy as np
        import tensorflow as tf
        from explain_pipeline import compute_gradcam,image_diagnostics
        inputs=tf.keras.Input(shape=(8,8,3))
        layer=tf.keras.layers.Conv2D(2,1,use_bias=False,kernel_initializer='ones',name='synthetic_conv')(inputs)
        score=tf.keras.layers.Dense(1,activation='sigmoid',use_bias=False,kernel_initializer='ones')(
            tf.keras.layers.GlobalAveragePooling2D()(layer))
        model=tf.keras.Model(inputs,score);grad=tf.keras.Model(inputs,[layer,score])
        image=np.linspace(0,.3,8*8*3,dtype='float32').reshape(8,8,3)
        plan=json.loads((ROOT/'config/explainability_plan.json').read_text())
        plan.update(random_masks=2,shift_pixels=[-1,1])
        heat,p=compute_gradcam(grad,tf.constant(image[None,...]),1)
        result=image_diagnostics(model,grad,image,heat,1,plan,42)
        self.assertGreater(p,.5);self.assertIsNotNone(result['faithfulness_margin'])
        self.assertIsNotNone(result['stability_cosine'])
        negative,_=compute_gradcam(grad,tf.constant(image[None,...]),0)
        result=image_diagnostics(model,grad,image,negative,0,plan,42)
        self.assertEqual(result['map_is_zero'],1)
        self.assertIsNone(result['faithfulness_margin'])
        self.assertIsNone(result['stability_cosine'])


if __name__=='__main__':unittest.main()
