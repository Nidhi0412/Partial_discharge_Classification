import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from train_overnight import make_plan,command_for,completed,validate_primary_plan


class QueueTests(unittest.TestCase):
    def test_primary_queue_rejects_disabled_augmentation_feature_only_or_wrong_profile(self):
        with tempfile.TemporaryDirectory() as d:
            for key,value in [('augment',False),('fine_tuning',False),('augmentation_profile','combined-image'),('image_stretch_fraction',.15)]:
                plan=make_plan(Path(d));plan['settings'][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):validate_primary_plan(plan)
            self.assertTrue(make_plan(Path(d))['settings']['augment'])

    def test_fixed_order_and_small_batch_pretrained_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            plan=make_plan(Path(d))
            self.assertEqual([j['name'] for j in plan['jobs']],
                ['vgg16_cv','resnet50_cv','vgg16_final-fit','resnet50_final-fit',
                 'vgg16_evaluate','resnet50_evaluate'])
            cmd=command_for(plan['jobs'][0],plan)
            self.assertEqual(cmd[cmd.index('--batch-size')+1],'2')
            self.assertEqual(cmd[cmd.index('--weights')+1],'imagenet')
            self.assertNotIn('--freeze-backbone',cmd)
            self.assertIn('--augment',cmd)
            self.assertEqual(cmd[cmd.index('--augmentation-profile')+1],'scalogram-image')
            self.assertEqual(cmd[cmd.index('--image-stretch-fraction')+1],'0.0')
            self.assertEqual(cmd[cmd.index('--image-shift-fraction')+1],'0.03')
            self.assertIsNone(plan['augmentation_parameters']['historical'])
            self.assertEqual(plan['augmentation_parameters']['operation_order'],
                ['horizontal_image_shift','row_band_mask','column_band_mask','scaled_pixel_noise'])
            self.assertEqual(cmd[cmd.index('--epochs')+1],'50')
            self.assertIn('--legacy-h5-export',cmd)

    def test_interrupted_training_resumes_and_evaluation_uses_final_run(self):
        with tempfile.TemporaryDirectory() as d:
            plan=make_plan(Path(d)); job=plan['jobs'][0]
            run=Path(job['run_dir']);run.mkdir()
            saved=dict(plan['settings'],model_type=job['model'],purpose='development_cv')
            (run/'run_config.json').write_text(json.dumps(saved))
            self.assertIn('resume',command_for(job,plan))
            (run/'run_config.json').write_text(json.dumps(dict(saved,augment=False)))
            with self.assertRaisesRegex(ValueError,'differs'):command_for(job,plan)
            (run/'run_config.json').write_text(json.dumps(saved))
            evaluation=plan['jobs'][-2]
            command=command_for(evaluation,plan)
            self.assertIn('evaluate',command)
            self.assertTrue(command[command.index('--run-dir')+1].endswith('vgg16_final'))
            self.assertFalse(completed(evaluation))
            status=Path(evaluation['run_dir'])/'holdout_evaluation'
            status.mkdir(parents=True)
            (status/'status.json').write_text(json.dumps({'status':'COMPLETE'}))
            self.assertTrue(completed(evaluation))


if __name__=='__main__': unittest.main()
