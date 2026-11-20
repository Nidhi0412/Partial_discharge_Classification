"""Contract tests for the corrected ResNet50 input path."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))

from dataset_manifest import digest
from training_pipeline import build_model, configure_loaded_backbone, make_dataset


class ResNetPreprocessingTests(unittest.TestCase):
    def test_native_preprocessing_matches_tensorflow_application(self):
        import tensorflow as tf

        with tempfile.TemporaryDirectory(prefix='resnet_input_') as directory:
            path = Path(directory) / 'sample.png'
            pixels = np.array([[[10, 20, 30], [40, 50, 60]],
                               [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)
            tf.io.write_file(str(path), tf.io.encode_png(pixels))
            row = dict(image_id='sample', image_path=str(path), recording_id='r1',
                       session_id='s1', material='Nomex', electrode='',
                       image_sha256=digest(path), group_id='r1', fold='1',
                       partition='validation')
            config = dict(model_type='resnet50', preprocessing_profile='resnet50_imagenet',
                          image_size=2, batch_size=1, seed=42, augment=False,
                          augmentation_profile='scalogram-image', resize_interpolation='nearest',
                          data_threads=1, prefetch_batches=1)
            dataset, _ = make_dataset([row], config, training=False)
            actual = next(iter(dataset))[0].numpy()
            expected = tf.keras.applications.resnet50.preprocess_input(
                pixels.astype('float32')[None, ...])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
            self.assertLess(float(actual.min()), 0.0)

    def test_native_preprocessing_rejects_other_models(self):
        config = dict(model_type='vgg16', preprocessing_profile='resnet50_imagenet',
                      image_size=2, batch_size=1, seed=42, augment=False,
                      augmentation_profile='scalogram-image', resize_interpolation='nearest',
                      data_threads=1, prefetch_batches=1)
        with self.assertRaisesRegex(ValueError, 'only be used with ResNet50'):
            make_dataset([dict(image_id='x', image_path='missing', recording_id='r',
                               group_id='r', material='Nomex', partition='validation')],
                         config, training=False)

    def test_warm_start_unfreezes_late_weights_and_keeps_batch_norm_frozen(self):
        import tensorflow as tf

        config = dict(model_type='resnet50', image_size=32, fine_tuning=False,
                      weights=None, dropout=.3, l2=.0001, learning_rate=.001,
                      training_profile='reviewed')
        model = build_model('resnet50', config)
        configure_loaded_backbone(tf, model, 'resnet50', trainable_layers=33,
                                  learning_rate=1e-5)
        trainable_weighted = [layer for layer in model.layers
                              if layer.trainable and layer.weights and layer.name != 'p_teflon']
        self.assertTrue(trainable_weighted)
        batch_norm = [layer for layer in model.layers
                      if isinstance(layer, tf.keras.layers.BatchNormalization)]
        self.assertTrue(batch_norm)
        self.assertTrue(all(not layer.trainable for layer in batch_norm))


if __name__ == '__main__':
    unittest.main()
