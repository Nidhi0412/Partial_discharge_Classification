import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

class ScalogramProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.wavelet = 'db3'
        self.max_level = 5
        self.scales = np.arange(1, 128)
        self.cwt_wavelet = 'morl'

    def csv_to_signal(self, csv_file):
        """
        Parses the uploaded CSV file. 
        Expected format: Skip 21 rows, use Col 0 (Time) and Col 1 (Signal).
        """
        try:
            # Read CSV skipping 21 header rows
            df = pd.read_csv(csv_file, skiprows=21, usecols=[0, 1], header=None)
            df.columns = ['time', 'signal']
            
            # Numeric conversion and cleaning
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df['signal'] = pd.to_numeric(df['signal'], errors='coerce')
            df = df.dropna()
            
            time_values = df['time'].values
            signal_values = df['signal'].values
            
            # Calculate sampling frequency
            dt = np.mean(np.diff(time_values))
            sampling_freq = 1.0 / dt
            
            return signal_values, sampling_freq
        except Exception as e:
            raise ValueError(f"Error parsing CSV: {str(e)}")

    def generate_scalogram(self, signal_values, sampling_freq, level=1):
        """
        Generates a 224x224 'clean' scalogram image as a numpy array.
        """
        # 1. DWT up to level 5
        coeffs = pywt.wavedec(signal_values, self.wavelet, level=self.max_level)
        
        # 2. Extract Detail Coefficients (Level 1 is index 1, Level 2 is index 2, etc.)
        # Based on generate_clean_scalograms.py, the Loop is level in range(1, 6)
        # and coeffs[level] is used. So Level 1 corresponds to coeffs[1].
        detail_coeffs = coeffs[level]
        
        # 3. CWT
        cwt_coeffs, _ = pywt.cwt(detail_coeffs, self.scales, self.cwt_wavelet, 1.0/sampling_freq)
        
        # 4. Generate 'Clean' Image (No axes, no labels)
        plt.figure(figsize=(self.target_size[0]/100, self.target_size[1]/100), dpi=100)
        try:
            plt.imshow(np.abs(cwt_coeffs), aspect='auto', cmap='jet', origin='lower')
            plt.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            
            # Convert to PIL and then to numpy array for model
            image = Image.open(buf).convert('RGB')
            image = image.resize(self.target_size)
            img_array = np.array(image)
            
            return img_array, buf
        finally:
            plt.close()

    def preprocess_for_model(self, img_array):
        """
        Normalization as per training (rescale 1/255).
        """
        x = img_array.astype('float32') / 255.0
        x = np.expand_dims(x, axis=0) # Add batch dimension
        return x
