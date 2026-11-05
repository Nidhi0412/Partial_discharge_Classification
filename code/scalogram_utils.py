"""Versioned scalogram inference utility for NEW, explicitly configured inputs.

Historical checkpoints must retain their historical transform. The new generator
uses true detail levels and is not implicitly compatible with old dc1 images.
"""
from io import BytesIO
from wavelet_pipeline import read_signal, transform, image_from_values


class ScalogramProcessor:
    def __init__(self, config):
        if config.get('schema_version') != 2:
            raise ValueError('Explicit version-2 generation config required; do not guess historical transforms')
        self.config = dict(config)

    def csv_to_signal(self, csv_path):
        _, signal, dt = read_signal(csv_path, self.config['skiprows'])
        return signal, 1.0 / dt

    def generate_scalogram(self, signal_values, sampling_freq, level, wavelet):
        import numpy as np
        values, _, metadata = transform(signal_values, 1.0 / sampling_freq, wavelet, level, self.config)
        image = image_from_values(values, self.config)
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return np.asarray(image), buffer, metadata

    def preprocess_for_model(self, img_array):
        import numpy as np
        image = np.asarray(img_array, dtype='float32') / 255.0
        if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
            raise ValueError('Expected finite RGB image matching the trained run configuration')
        return image[None, ...]
