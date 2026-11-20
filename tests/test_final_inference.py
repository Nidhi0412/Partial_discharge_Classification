"""Contracts for the final Streamlit inference path."""

from io import BytesIO
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from inference import (
    MODEL_SPECS,
    agreement_label,
    decode_uploaded_image,
    preprocess_image,
    validate_checkpoint,
)


class FinalInferenceTests(unittest.TestCase):
    def test_final_checkpoints_match_frozen_hashes(self):
        for name in MODEL_SPECS:
            self.assertTrue(validate_checkpoint(name).is_file())

    def test_uploaded_image_is_rgb_and_nearest_resized(self):
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        pixels[0, 0] = [255, 0, 0]
        stream = BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        stream.seek(0)
        image, metadata = decode_uploaded_image(stream)
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertEqual(metadata["resize_interpolation"], "nearest")
        np.testing.assert_array_equal(image[0, 0], [255, 0, 0])

    def test_architecture_specific_preprocessing(self):
        pixels = np.full((224, 224, 3), [10, 20, 30], dtype=np.uint8)
        vgg = preprocess_image(pixels, "VGG16")
        np.testing.assert_allclose(vgg[0, 0, 0], np.array([10, 20, 30]) / 255.0)

        import tensorflow as tf

        expected = tf.keras.applications.resnet50.preprocess_input(
            pixels.astype("float32")[None, ...]
        )
        actual = preprocess_image(pixels, "ResNet50")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        self.assertFalse(np.allclose(actual, vgg))

    def test_agreement_is_not_probability_averaging(self):
        same = [
            {"predicted_class": "Nomex"},
            {"predicted_class": "Nomex"},
        ]
        different = [
            {"predicted_class": "Nomex"},
            {"predicted_class": "Teflon"},
        ]
        self.assertEqual(agreement_label(same), "Nomex")
        self.assertEqual(agreement_label(different), "Disagreement")


if __name__ == "__main__":
    unittest.main()
