import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clean_scalograms.log'),
        logging.StreamHandler()
    ]
)

def load_signal(file_path):
    """
    Load signal data from CSV file, skipping the first 21 rows.
    First column is time data, second column is sensor 1 data.
    """
    try:
        # Read CSV file, skipping first 21 rows and using only first two columns
        df = pd.read_csv(file_path, skiprows=21, usecols=[0, 1], header=None)
        
        # Convert columns to numeric, handling any non-numeric values
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # Time column
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # Signal column
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Extract time and signal values
        time_values = df.iloc[:, 0].values  # First column is time
        signal_values = df.iloc[:, 1].values  # Second column is sensor 1 data
        
        # Calculate sampling frequency from time data
        sampling_freq = 1 / np.mean(np.diff(time_values))
        
        return time_values, signal_values, sampling_freq
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return None, None, None

def save_clean_scalogram(cwt_coeffs, extent, output_path):
    """
    Save a clean scalogram without axes, titles, or colorbars.
    Args:
        cwt_coeffs: CWT coefficients
        extent: Extent of the plot
        output_path: Path to save the image
    """
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    try:
        plt.imshow(np.abs(cwt_coeffs), extent=extent, aspect='auto', cmap='jet', origin='lower')
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    finally:
        plt.close()

def process_signal(file_path, level):
    """
    Process a single signal file and generate CWT coefficients for specified level.
    Args:
        file_path: Path to the CSV file
        level: Decomposition level (1-5)
    Returns:
        cwt_coeffs, freqs, detail_coeffs, sampling_freq
    """
    # Load and preprocess signal
    time_values, signal_values, sampling_freq = load_signal(file_path)
    if time_values is None:
        return None, None, None, None
    
    # Wavelet decomposition
    wavelet = 'db3'
    max_level = 5
    coeffs = pywt.wavedec(signal_values, wavelet, level=max_level)
    
    # Get detail coefficients for current level
    detail_coeffs = coeffs[level]
    
    # Generate CWT using Morlet wavelet
    scales = np.arange(1, 128)
    cwt_coeffs, freqs = pywt.cwt(detail_coeffs, scales, 'morl', 1/sampling_freq)
    
    return cwt_coeffs, freqs, detail_coeffs, sampling_freq

def save_coefficients(detail_coeffs, output_path):
    """
    Save detail coefficients to CSV file.
    Args:
        detail_coeffs: List of detail coefficients
        output_path: Path to save the CSV file
    """
    coeffs_dict = {}
    maxlen = max(len(dc) for dc in detail_coeffs)
    for i, dc in enumerate(detail_coeffs):
        padded = np.full(maxlen, np.nan)
        padded[:len(dc)] = dc
        coeffs_dict[f'dc{i+1}'] = padded
    df = pd.DataFrame(coeffs_dict)
    df.to_csv(output_path, index=False)

def process_dataset(input_dir, output_dir, class_labels):
    """
    Process all CSV files and generate clean scalograms organized by decomposition levels.
    Args:
        input_dir (str): Directory containing CSV files
        output_dir (str): Directory to save clean scalogram images
        class_labels (list): List of class labels
    """
    logging.info(f"Starting preprocessing with input_dir: {input_dir}")
    
    # Calculate total files for progress tracking
    total_files = sum(len([f for f in os.listdir(os.path.join(input_dir, label)) if f.endswith('.csv')]) 
                     for label in class_labels)
    
    # Create progress bar
    pbar = tqdm(total=total_files * 5, desc="Processing files")
    
    # Process each decomposition level
    for level in range(1, 6):  # dc1 to dc5
        level_dir = f"dc{level}"
        logging.info(f"Processing {level_dir}")
        
        for label in class_labels:
            try:
                class_input_dir = os.path.join(input_dir, label)
                class_output_dir = os.path.join(output_dir, level_dir, label)
                coeffs_dir = os.path.join(output_dir, "coeffs", label)
                os.makedirs(class_output_dir, exist_ok=True)
                os.makedirs(coeffs_dir, exist_ok=True)
                
                for file_name in os.listdir(class_input_dir):
                    if file_name.endswith('.csv'):
                        try:
                            file_path = os.path.join(class_input_dir, file_name)
                            file_stem = os.path.splitext(file_name)[0]
                            
                            # Process signal
                            cwt_coeffs, freqs, detail_coeffs, sampling_freq = process_signal(file_path, level)
                            if cwt_coeffs is None:
                                continue
                            
                            # Save scalogram
                            extent = [0, len(detail_coeffs) / sampling_freq, freqs[-1]/1e6, freqs[0]/1e6]
                            output_path = os.path.join(class_output_dir, f"{file_stem}.png")
                            save_clean_scalogram(cwt_coeffs, extent, output_path)
                            
                            # Save coefficients
                            coeffs_path = os.path.join(coeffs_dir, f"{file_stem}_detailed_coeffs.csv")
                            save_coefficients([detail_coeffs], coeffs_path)
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            logging.error(f"Error processing {file_name}: {str(e)}")
                            continue
                            
            except Exception as e:
                logging.error(f"Error processing class {label}: {str(e)}")
                continue
    
    pbar.close()
    logging.info(f"Completed preprocessing. Output saved to: {output_dir}")

def main():
    """
    Main function to run the preprocessing pipeline.
    """
    # Define input and output directories
    input_dir = "Data for Classification"
    output_dir = "scalograms_cleaned"  # New output directory
    
    # Define class labels
    class_labels = ['NN100sig', 'NF100sig', 'TN100sig', 'TF100sig']
    
    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        logging.warning(f"Output directory {output_dir} already exists. Contents will be overwritten.")
    else:
        os.makedirs(output_dir)
    
    # Process the dataset
    process_dataset(input_dir, output_dir, class_labels)

if __name__ == "__main__":
    main()