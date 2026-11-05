import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import training_pipeline as pipeline
from legacy_augmentation import HISTORICAL_PARAMETERS,make_transform


class HistoricalAugmentationTests(unittest.TestCase):
    def test_fit_rejects_wrong_partitions_and_related_parent_groups(self):
        train=dict(image_id='a',image_path='a.png',recording_id='r1',group_id='g1',image_sha256='hash1',partition='train')
        validation=dict(image_id='b',image_path='b.png',recording_id='r2',group_id='g2',image_sha256='hash2',partition='validation')
        pipeline.validate_fit_membership([train],[validation])
        for key in ['image_id','image_path','recording_id','group_id','image_sha256']:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'overlap'):
                pipeline.validate_fit_membership([train],[dict(validation,**{key:train[key]})])
        with self.assertRaisesRegex(ValueError,'Unexpected member'):
            pipeline.validate_fit_membership([dict(train,partition='test')],[validation])

    def test_launch_defaults_cannot_silently_disable_augmentation(self):
        with patch.object(pipeline,'run_training') as run:
            pipeline.main(['cv','--splits','unused.csv','--output-dir','unused'])
        args=run.call_args.args[0]
        self.assertTrue(args.augment)
        self.assertEqual(args.augmentation_profile,'scalogram-image')
        self.assertEqual(args.image_stretch_fraction,0.)
        self.assertEqual(args.image_shift_fraction,.03)
        self.assertEqual(args.training_profile,'historical')
        self.assertEqual(args.epochs,50)
        self.assertEqual(args.batch_size,2)
        self.assertEqual(args.lr_reduce_factor,.1)

    def test_effective_transform_matches_historical_generator(self):
        import numpy as np
        import tensorflow as tf
        image=np.random.default_rng(8).integers(0,256,(32,32,3),dtype=np.uint8).astype('float32')/255.
        transform=make_transform(tf,[],dict(seed=42))
        np.random.seed(61); previous=np.random.get_state()
        result=transform(image,b'not-exported',np.int32(123))
        subsequent=np.random.random()
        np.random.set_state(previous)
        self.assertEqual(subsequent,np.random.random())
        generator=tf.keras.preprocessing.image.ImageDataGenerator(**HISTORICAL_PARAMETERS)
        params=generator.get_random_transform(image.shape,seed=123)
        expected=generator.standardize(generator.apply_transform(image*255.,params)).astype('float32')
        np.testing.assert_allclose(result,expected,rtol=0,atol=0)
        self.assertGreater(float(np.abs(result-image).mean()),.01)

    def test_actual_training_tensor_saved_and_validation_is_unaugmented(self):
        import numpy as np
        import tensorflow as tf
        from dataset_manifest import digest
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); path=out/'source.png'
            image=np.random.default_rng(3).integers(0,256,(32,32,3),dtype=np.uint8)
            tf.io.write_file(str(path),tf.io.encode_png(image))
            rows=[dict(image_id='one',image_path=str(path),recording_id='lab1',group_id='lab1',
                       material='Nomex',partition='train',image_sha256=digest(path),fold='1')]
            config=dict(image_size=32,batch_size=1,seed=42,augment=True,augmentation_profile='historical-image',
                        resize_interpolation='nearest',augmentation_export_dir=str(out/'saved'),
                        augmentation_samples_per_class=5,data_threads=1,prefetch_batches=1)
            ds,_=pipeline.make_dataset(rows,config,True)
            batch,label=next(iter(ds))
            np.testing.assert_allclose(batch[0].numpy(),np.load(out/'saved/Nomex/one_augmented.npy'),rtol=0,atol=0)
            metadata=json.loads((out/'saved/Nomex/one.json').read_text())
            self.assertEqual(metadata['recording_id'],'lab1')
            self.assertEqual(metadata['partition'],'train')
            # Same augment=True config must not augment validation/inference inputs.
            validation_config=dict(config,augmentation_export_dir=str(out/'validation_exports'))
            val,_=pipeline.make_dataset(rows,validation_config,False)
            first=next(iter(val))[0].numpy()
            second=next(iter(val))[0].numpy()
            np.testing.assert_allclose(first[0],image.astype('float32')/255.,rtol=0,atol=1e-7)
            np.testing.assert_array_equal(first,second)
            self.assertFalse((out/'validation_exports').exists())


if __name__=='__main__': unittest.main()
