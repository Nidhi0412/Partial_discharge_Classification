"""Fast contract checks using temporary synthetic data, never the research dataset."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from dataset_manifest import (CLASS_NAMES,FIELDS,digest,inventory_images,load_splits,
                              make_splits,read_images,validate_split,write_csv)
from training_pipeline import binary_metrics,canonical_model,prediction_summary,recording_predictions
from evaluation_tools import read_predictions,roc_points
from explain_pipeline import binary_target


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.images=[]
        for i in range(40):
            for variant in range(2):
                p=self.base/f'r{i}_v{variant}.png';p.write_bytes(f'synthetic-{i}-{variant}'.encode())
                self.images.append(dict(image_id=f'r{i}_v{variant}',image_path=str(p),recording_id=f'r{i}',
                    material=CLASS_NAMES[i%2],electrode='Flat' if (i//2)%2 else 'Pointed',
                    session_id=f's{i//2}',image_sha256=digest(p)))
        self.manifest=self.base/'images.csv';write_csv(self.manifest,self.images,FIELDS)

    def split(self,name='split',group_by='recording_id'):
        path=make_splits(self.manifest,self.base/name,folds=5,seed=42,group_by=group_by)
        return load_splits(path)

    def test_all_groups_remain_disjoint_and_coverage_complete(self):
        rows,config=self.split()
        self.assertEqual(len(rows),400)
        self.assertEqual(validate_split(rows),['1','2','3','4','5'])
        val=[r['image_id'] for r in rows if r['partition']=='validation']
        self.assertEqual(len(val),len(set(val)))
        self.assertEqual({r['material'] for r in rows if r['partition']=='test'},set(CLASS_NAMES))

    def test_sessions_can_group_multiple_materials(self):
        rows,config=self.split(group_by='session_id')
        self.assertEqual(validate_split(rows,'session_id'),['1','2','3','4','5'])
        for fold in ['1','2','3','4','5']:
            assignments={}
            for r in rows:
                if r['fold']==fold:
                    self.assertEqual(assignments.setdefault(r['session_id'],r['partition']),r['partition'])

    def test_split_is_deterministic(self):
        first,_=self.split('first');second,_=self.split('second')
        self.assertEqual(first,second)

    def test_content_co_location_keeps_distinct_recording_ids(self):
        Path(self.images[2]['image_path']).write_bytes(Path(self.images[0]['image_path']).read_bytes())
        self.images[2]['image_sha256']=digest(self.images[2]['image_path'])
        manifest=self.base/'content.csv';write_csv(manifest,self.images,FIELDS)
        path=make_splits(manifest,self.base/'content_split',folds=5,keep_identical_together=True)
        rows,config=load_splits(path)
        self.assertEqual(config['group_by'],'group_id')
        self.assertEqual(config['acquisition_group_by'],'recording_id')
        self.assertEqual({r['recording_id'] for r in rows},{r['recording_id'] for r in self.images})
        for fold in ['1','2','3','4','5']:
            linked=[r for r in rows if r['fold']==fold and r['recording_id'] in ['r0','r1']]
            self.assertEqual(len({r['partition'] for r in linked}),1)
            self.assertEqual(len({r['group_id'] for r in linked}),1)

    def test_recording_bootstrap_retains_content_cluster(self):
        rows=[dict(recording_id=r,group_id='same_content',fold='1',y_true=0,p_teflon=.1) for r in ['a','b']]
        summary,records=prediction_summary(rows,bootstrap=10,group_by='group_id')
        self.assertEqual(len(records),2)
        self.assertEqual(summary['recording_accuracy_cluster_bootstrap_95_percentile']['independent_clusters'],1)

    def test_original_files_not_moved_or_modified(self):
        self.split()
        self.assertTrue(all(Path(r['image_path']).is_file() and digest(r['image_path'])==r['image_sha256'] for r in self.images))

    def test_missing_recording_id_is_rejected(self):
        rows=[dict(r,recording_id='') for r in self.images]
        p=self.base/'unknown.csv';write_csv(p,rows,FIELDS)
        with self.assertRaisesRegex(ValueError,'recording_id'):read_images(p)

    def test_conflicting_recording_labels_rejected(self):
        rows=[dict(r) for r in self.images];rows[1]['material']='Teflon'
        p=self.base/'conflicting.csv';write_csv(p,rows,FIELDS)
        with self.assertRaisesRegex(ValueError,'Inconsistent'):read_images(p)

    def test_duplicate_image_id_rejected(self):
        rows=[dict(r) for r in self.images];rows[1]['image_id']=rows[0]['image_id']
        p=self.base/'duplicate.csv';write_csv(p,rows,FIELDS)
        with self.assertRaisesRegex(ValueError,'duplicate'):read_images(p)

    def test_image_id_cannot_be_a_path(self):
        rows=[dict(r) for r in self.images];rows[0]['image_id']='../outside'
        p=self.base/'invalid_id.csv';write_csv(p,rows,FIELDS)
        with self.assertRaisesRegex(ValueError,'safe identifier'):read_images(p)

    def test_inventory_does_not_invent_recording_ids(self):
        p=self.base/'template.csv';count=inventory_images(self.base,p)
        with p.open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(count,80);self.assertTrue(all(not r['recording_id'] for r in rows))

    def test_changed_image_rejected_before_training(self):
        make_splits(self.manifest,self.base/'split',folds=5)
        Path(self.images[0]['image_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Image changed'):load_splits(self.base/'split/splits.csv')

    def test_changed_manifest_rejected(self):
        make_splits(self.manifest,self.base/'split',folds=5)
        p=self.base/'split/splits.csv';p.write_bytes(p.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'manifest changed'):load_splits(p)

    def test_output_directory_never_overwritten(self):
        self.split()
        with self.assertRaises(FileExistsError):self.split()

    def test_cross_partition_recording_rejected(self):
        rows,_=self.split()
        first=next(r for r in rows if r['fold']=='1' and r['partition']=='validation')
        train=next(r for r in rows if r['fold']=='1' and r['partition']=='train')
        first['recording_id']=train['recording_id'];first['group_id']=train['group_id']
        with self.assertRaises(ValueError):validate_split(rows)

    def test_cross_partition_identical_content_rejected(self):
        rows,_=self.split()
        train=next(r for r in rows if r['fold']=='1' and r['partition']=='train')
        other=next(r for r in rows if r['fold']=='1' and r['partition']=='validation')
        for r in rows:
            if r['image_id']==other['image_id']:r['image_sha256']=train['image_sha256']
        with self.assertRaisesRegex(ValueError,'image_sha256 crosses'):validate_split(rows)

    def test_missing_fold_assignment_rejected(self):
        rows,_=self.split();rows=[r for r in rows if not(r['fold']=='2' and r['partition']=='validation')]
        with self.assertRaises(ValueError):validate_split(rows)

    def test_model_selection_is_case_normalized(self):
        self.assertEqual(canonical_model('VGG16'),'vgg16')
        self.assertEqual(canonical_model('resnet50'),'resnet50')
        with self.assertRaises(ValueError):canonical_model('unknown')

    def test_factory_calls_requested_backbone(self):
        from types import SimpleNamespace
        from training_pipeline import build_model
        for requested,constructor,layer in [('VGG16','VGG16','block5_conv3'),('resnet50','ResNet50','conv5_block3_out')]:
            tf=MagicMock()
            base=MagicMock();base.layers=[SimpleNamespace(name=layer,trainable=True)]
            getattr(tf.keras.applications,constructor).return_value=base
            tf.keras.Model.return_value.layers=base.layers
            tf.keras.Model.return_value.output_shape=(None,1)
            config=dict(image_size=32,fine_tuning=False,weights=None,dropout=.1,l2=0.,learning_rate=.001)
            with patch('training_pipeline.tensorflow',return_value=tf):build_model(requested,config)
            getattr(tf.keras.applications,constructor).assert_called_once()
            other='ResNet50' if constructor=='VGG16' else 'VGG16'
            getattr(tf.keras.applications,other).assert_not_called()

    def test_sigmoid_threshold_and_metrics(self):
        m=binary_metrics([0,0,1,1],[.1,.8,.9,.5])
        self.assertEqual(m['confusion_matrix'],[[1,1],[0,2]])
        self.assertEqual(m['accuracy'],.75)
        self.assertEqual(m['specificity'],.5)
        self.assertEqual(m['auc'],.75)

    def test_auc_ties_and_perfect_order(self):
        self.assertEqual(binary_metrics([0,1],[.5,.5])['auc'],.5)
        self.assertEqual(binary_metrics([0,1],[.1,.9])['auc'],1.)
        self.assertIsNone(binary_metrics([1],[.7])['auc'])

    def test_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):binary_metrics([0,1],[.1,float('nan')])

    def test_recording_endpoint_and_bootstrap(self):
        rows=[dict(recording_id='a',fold='1',y_true=0,p_teflon=p,session_id='s1') for p in [.1,.3]]
        rows += [dict(recording_id='b',fold='2',y_true=1,p_teflon=p,session_id='s2') for p in [.7,.9]]
        summary,records=prediction_summary(rows,bootstrap=20,group_by='session_id')
        self.assertEqual(summary['recording']['samples'],2)
        self.assertEqual(summary['recording']['accuracy'],1.)
        self.assertAlmostEqual(records[0]['p_teflon'],.2)
        self.assertEqual(summary['recording_accuracy_cluster_bootstrap_95_percentile']['independent_clusters'],2)

    def test_repeated_recording_oof_predictions_rejected(self):
        rows=[dict(recording_id='a',fold=f,y_true=0,p_teflon=.1) for f in ['1','2']]
        with self.assertRaisesRegex(ValueError,'repeated recording'):recording_predictions(rows)

    def test_binary_gradcam_targets(self):
        self.assertAlmostEqual(binary_target(.8,0),.2)
        self.assertEqual(binary_target(.8,1),.8)

    def test_roc_ties_have_one_threshold(self):
        rows=[dict(y_true=0,p_teflon=.5),dict(y_true=1,p_teflon=.5)]
        self.assertEqual(roc_points(rows),[(0.,0.),(1.,1.)])

    def test_cli_help_does_not_import_tensorflow(self):
        import subprocess
        for filename in ['model_training.py','model_training_vgg16.py','split_dataset.py',
                         'evaluation.py','run_gradcam.py','generate_clean_scalograms.py']:
            p=subprocess.run([sys.executable,str(ROOT/'code'/filename),'--help'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)


@unittest.skipUnless(os.environ.get('PD_RUN_TF_SMOKE')=='1' and importlib.util.find_spec('tensorflow'),
                     'Actual TensorFlow training smoke test is opt-in; run the explicit smoke-test command')
class TensorFlowSmokeTest(unittest.TestCase):
    def test_actual_backbones_fit_save_reload(self):
        from training_pipeline import smoke_test
        smoke_test()


if __name__=='__main__':unittest.main()
