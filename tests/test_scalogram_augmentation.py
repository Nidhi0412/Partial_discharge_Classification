import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from scalogram_augmentation import DEFAULTS,PRIMARY_DEFAULTS,operation_order,compose,make_transform,parameters,tensor_hash


class ScalogramAugmentationTests(unittest.TestCase):
    def test_example_exports_keep_short_names_under_deep_windows_paths(self):
        import numpy as np
        from scalogram_augmentation import save_example
        image=np.random.default_rng(2).uniform(.1,.9,(32,32,3)).astype('float32')
        result,stages=compose(image,1,DEFAULTS)
        with tempfile.TemporaryDirectory() as d:
            # Original verbose stage filenames exceed 260 chars here on Windows.
            parent=Path(d)/('d'*max(1,205-len(str(Path(d)))-1))
            source=dict(image_id='a'*24,material='Teflon',recording_id='r1',group_id='g1',partition='train')
            config=dict(augmentation_export_dir=str(parent),augmentation_profile='combined-image',**DEFAULTS)
            save_example(source,image,result,stages,1,config)
            files=[p for p in parent.rglob('*') if p.is_file()]
            self.assertTrue(files)
            self.assertTrue(all(len(str(p))<260 for p in files))
            metadata=json.loads((parent/'Teflon'/('a'*24+'.json')).read_text())
            self.assertTrue(all(Path(s['tensor_path']).is_file() for s in metadata['stages']))

    def test_primary_profile_has_no_geometric_generator_or_stretch(self):
        import numpy as np
        from unittest.mock import patch
        image=np.random.default_rng(9).uniform(.1,.9,(224,224,3)).astype('float32')
        row=dict(image_id='one',image_path='one.png',material='Nomex',partition='train',recording_id='r1',group_id='g1')
        config=dict(augmentation_profile='scalogram-image',augment=True)
        self.assertEqual(parameters(config),PRIMARY_DEFAULTS)
        with patch('scalogram_augmentation.apply_historical',side_effect=AssertionError('Historical geometry must not run')):
            # No TensorFlow preprocessing object is needed for the primary profile.
            result=make_transform(None,[row],config)(image,b'one.png',42)
        expected,stages=compose(image,42,config)
        np.testing.assert_array_equal(result,expected)
        self.assertEqual([s['operation'] for s in stages],operation_order(config))
        self.assertEqual(len(stages),4)
        self.assertLessEqual(abs(stages[0]['parameters']['pixels']),int(224*.03))
        self.assertNotIn('historical_image',operation_order(config))
        self.assertNotIn('vertical_image_stretch',operation_order(config))
        self.assertEqual(operation_order(dict(config,augment=False)),[])

    def test_every_operation_is_composed_and_replayable(self):
        import numpy as np
        image=np.random.default_rng(7).uniform(.1,.9,(32,40,3)).astype('float32')
        first,stages=compose(image,42,DEFAULTS)
        second,replay=compose(image,42,DEFAULTS)
        np.testing.assert_array_equal(first,second)
        self.assertEqual([s['operation'] for s in stages],
            ['vertical_image_stretch','horizontal_image_shift','row_band_mask','column_band_mask','scaled_pixel_noise'])
        self.assertEqual(stages[0]['input_sha256'],tensor_hash(image))
        for previous,following in zip(stages,stages[1:]):
            self.assertEqual(previous['output_sha256'],following['input_sha256'])
        self.assertEqual(stages[-1]['output_sha256'],tensor_hash(first))
        self.assertTrue(all(s['input_sha256']!=s['output_sha256'] for s in stages))
        self.assertEqual(first.shape,image.shape)
        self.assertTrue(np.isfinite(first).all() and first.min()>=0 and first.max()<=1)
        self.assertFalse(np.array_equal(first,compose(image,43,DEFAULTS)[0]))

    def test_row_and_column_masks_preserve_unmasked_pixels(self):
        import numpy as np
        image=np.ones((32,40,3),dtype='float32')*.7
        for axis,key in [(0,'row_mask_fraction'),(1,'column_mask_fraction')]:
            config={k:0 for k in DEFAULTS};config[key]=.15
            result,stages=compose(image,8,config)
            p=stages[0]['parameters'];start=p['start_pixel'];width=p['width_pixels']
            self.assertLessEqual(width,int(image.shape[axis]*.15))
            expected=image.copy();index=[slice(None)]*3;index[axis]=slice(start,start+width);expected[tuple(index)]=0
            np.testing.assert_array_equal(result,expected)

    def test_horizontal_shift_does_not_wrap(self):
        import numpy as np
        image=np.tile(np.linspace(0,1,40,dtype='float32')[None,:,None],(32,1,3))
        config={k:0 for k in DEFAULTS};config['image_shift_fraction']=.15
        result,stages=compose(image,42,config);shift=stages[0]['parameters']['pixels']
        if shift>0:
            np.testing.assert_array_equal(result[:,shift:],image[:,:-shift])
            np.testing.assert_array_equal(result[:,:shift],np.repeat(image[:,:1],shift,axis=1))
        else:
            np.testing.assert_array_equal(result[:,:shift],image[:,-shift:])
            np.testing.assert_array_equal(result[:,shift:],np.repeat(image[:,-1:],-shift,axis=1))

    def test_invalid_inputs_and_nontraining_parents_fail_loudly(self):
        import numpy as np
        for value in [float('nan'),float('inf'),-.1,.5]:
            with self.assertRaises(ValueError):parameters(dict(row_mask_fraction=value))
        with self.assertRaises(ValueError):compose(np.full((32,32,3),float('nan')),1,DEFAULTS)
        for partition in ['validation','test']:
            with self.assertRaisesRegex(ValueError,'training parents'):
                make_transform(None,[dict(partition=partition)],dict(augmentation_profile='scalogram-image'))

    def test_actual_training_tensor_and_all_stage_parent_evidence_saved(self):
        import numpy as np
        import tensorflow as tf
        from training_pipeline import make_dataset
        from dataset_manifest import digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'input.png'
            pixels=np.random.default_rng(4).integers(0,256,(32,32,3),dtype='uint8')
            tf.io.write_file(str(path),tf.io.encode_png(pixels))
            row=dict(image_id='original1',image_path=str(path),recording_id='lab_record1',group_id='parent_group1',
                     image_sha256=digest(path),material='Nomex',partition='train',fold='2')
            config=dict(image_size=32,batch_size=1,seed=42,augment=True,
                        augmentation_profile='combined-image',resize_interpolation='nearest',
                        augmentation_export_dir=str(root/'examples'),augmentation_ledger_dir=str(root/'ledger'),**DEFAULTS)
            ds,_=make_dataset([row],config,True)
            actual,label=next(iter(ds))
            metadata=json.loads((root/'examples/Nomex/original1.json').read_text())
            np.testing.assert_array_equal(actual[0].numpy(),np.load(metadata['exact_generated_tensor']))
            self.assertEqual(metadata['recording_id'],'lab_record1')
            self.assertEqual(metadata['group_id'],'parent_group1')
            self.assertEqual(metadata['partition'],'train')
            self.assertEqual(len(metadata['stages']),6)
            for stage in metadata['stages']:
                self.assertEqual(tensor_hash(np.load(stage['tensor_path'])),stage['output_sha256'])
            records=[json.loads(line) for p in (root/'ledger').glob('*.jsonl') for line in p.read_text().splitlines()]
            self.assertTrue(records)
            self.assertTrue(all(r['image_id']==row['image_id'] and r['recording_id']==row['recording_id'] and
                                r['group_id']==row['group_id'] and r['partition']=='train' for r in records))
            # Identical augment=True configuration must leave inference/validation clean.
            validation_config=dict(config,augmentation_export_dir=str(root/'val_examples'),augmentation_ledger_dir=str(root/'val_ledger'))
            val,_=make_dataset([dict(row,partition='validation')],validation_config,False)
            validation=next(iter(val))[0].numpy()
            np.testing.assert_allclose(validation[0],pixels.astype('float32')/255,rtol=0,atol=1e-7)
            np.testing.assert_array_equal(validation,next(iter(val))[0].numpy())
            self.assertFalse((root/'val_examples').exists())
            self.assertFalse((root/'val_ledger').exists())


if __name__=='__main__':unittest.main()
