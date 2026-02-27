import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import traceback

bad_shape_files = []

def load_signal(file_path, skip_rows=20):
    """
    Load signal data from CSV file, skipping the first 20 rows.
    First column is time data, second column is sensor 1 data.
    """
    global bad_shape_files
    try:
        print(f"\nDebug: Loading file {file_path}")
        
        # Read CSV file, skipping first 20 rows and using only first two columns
        df = pd.read_csv(file_path, skiprows=21, usecols=[0, 1], header=None)
        if df.shape != (10000, 2):
            bad_shape_files.append((file_path, df.shape))
        print(f"Debug: DataFrame shape: {df.shape}")
        print(f"Debug: First few rows of data:\n{df.head()}")
        print(f"Debug: Data types:\n{df.dtypes}")
        
        # Convert columns to numeric, handling any non-numeric values
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # Time column
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # Signal column
        
        # Remove any rows with NaN values
        df = df.dropna()
        print(f"Debug: DataFrame shape after cleaning: {df.shape}")
        
        # Extract time and signal values
        time_values = df.iloc[:, 0].values  # First column is time
        signal_values = df.iloc[:, 1].values  # Second column is sensor 1 data
        
        print(f"Debug: Time values range: {time_values.min()} to {time_values.max()}")
        print(f"Debug: Signal values range: {signal_values.min()} to {signal_values.max()}")
        
        # Calculate sampling frequency from time data
        time_diff = np.diff(time_values)
        print(f"Debug: Time differences range: {time_diff.min()} to {time_diff.max()}")
        sampling_freq = 1 / np.mean(time_diff)
        print(f"Debug: Calculated sampling frequency: {sampling_freq} Hz")
        
        return time_values, signal_values, sampling_freq
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        print("Debug: Full error details:")
        print(traceback.format_exc())
        return None, None, None

def generate_scalogram(signal_data, time_values, output_path):
    """
    Generate a true scalogram (2D time-frequency representation) from the level 1 detail coefficients.
    """
    try:
        print(f"\nDebug: Generating scalogram for {output_path}")
        # Calculate sampling frequency from time column
        sampling_freq = 1 / np.mean(np.diff(time_values))
        print(f"Debug: Sampling frequency: {sampling_freq} Hz")

        # 1. DWT up to level 5
        wavelet = 'db3'
        max_level = 5
        coeffs = pywt.wavedec(signal_data, wavelet, level=max_level)
        detail_coeffs_level1 = coeffs[1]

        # 2. CWT on level 1 detail coefficients
        scales = np.arange(1, 128)
        cwt_coeffs, freqs = pywt.cwt(detail_coeffs_level1, scales, 'morl', 1/sampling_freq)

        # 3. Plot the scalogram (2D)
        plt.figure(figsize=(10, 6))
        extent = [0, len(detail_coeffs_level1) / sampling_freq, freqs[-1]/1e6, freqs[0]/1e6]
        plt.imshow(np.abs(cwt_coeffs), extent=extent, aspect='auto', cmap='jet', origin='lower')
        plt.title('Scalogram (CWT of Level 1 Detail Coefficients)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (MHz)')
        plt.colorbar(label='Magnitude')
        plt.savefig(output_path)
        plt.close()
        print(f"Debug: Scalogram saved successfully to {output_path}")
        return True
    except Exception as e:
        print(f"Error generating scalogram: {str(e)}")
        print("Debug: Full error details:")
        print(traceback.format_exc())
        return False

def process_folder(input_folder, output_base_dir, class_folder):
    """
    Process all CSV files in a folder and generate scalograms for all levels.
    """
    try:
        print(f"\nDebug: Processing folder {input_folder}")
        # Get all CSV files
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        print(f"Debug: Found {len(csv_files)} CSV files")
        for file_name in csv_files:
            print(f"\nProcessing {file_name}...")
            file_path = os.path.join(input_folder, file_name)
            time_values, signal_values, sampling_freq = load_signal(file_path)
            if time_values is not None and signal_values is not None and len(time_values) == 10000:
                file_stem = os.path.splitext(file_name)[0]
                save_all_detail_coeffs_and_scalograms(signal_values, time_values, output_base_dir, class_folder, file_stem)
            else:
                print(f"Skipping {file_name} due to loading error or bad shape")
    except Exception as e:
        print(f"Error processing folder {input_folder}: {str(e)}")
        print("Debug: Full error details:")
        print(traceback.format_exc())

def check_data_points(input_base_dir):
    """
    Check all CSV files and identify those with 1000 datapoints.
    """
    print("\nChecking files with 1000 datapoints:")
    print("-" * 50)
    
    folders = ["TN100sig", "TF100sig", "NN100sig", "NF100sig"]
    files_with_1000 = []
    
    for folder in folders:
        input_folder = os.path.join(input_base_dir, folder)
        if not os.path.exists(input_folder):
            print(f"Warning: Folder {input_folder} does not exist!")
            continue
            
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        print(f"\nChecking {folder}:")
        print(f"Total CSV files in {folder}: {len(csv_files)}")
        
        for file_name in csv_files:
            file_path = os.path.join(input_folder, file_name)
            try:
                df = pd.read_csv(file_path, skiprows=21, usecols=[0, 1], header=None)
                if len(df) == 1000:
                    files_with_1000.append((folder, file_name))
                    print(f"  {file_name}: {len(df)} datapoints")
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
    
    print("\nSummary:")
    print("-" * 50)
    print(f"Total files with 1000 datapoints: {len(files_with_1000)}")
    print("\nFiles with 1000 datapoints:")
    for folder, file_name in files_with_1000:
        print(f"{folder}/{file_name}")
    print("-" * 50)
    return files_with_1000

def save_all_detail_coeffs_and_scalograms(signal_data, time_values, output_base_dir, class_folder, file_stem):
    sampling_freq = 1 / np.mean(np.diff(time_values))
    wavelet = 'db3'
    max_level = 5
    coeffs = pywt.wavedec(signal_data, wavelet, level=max_level)
    detail_coeffs = coeffs[1:]  # [dc1, dc2, ..., dc5]

    # Save all detail coefficients in one CSV (in the class folder)
    coeffs_dict = {}
    maxlen = max(len(dc) for dc in detail_coeffs)
    for i, dc in enumerate(detail_coeffs):
        padded = np.full(maxlen, np.nan)
        padded[:len(dc)] = dc
        coeffs_dict[f'dc{i+1}'] = padded
    df = pd.DataFrame(coeffs_dict)
    coeffs_dir = os.path.join(output_base_dir, "coeffs", class_folder)
    os.makedirs(coeffs_dir, exist_ok=True)
    df.to_csv(os.path.join(coeffs_dir, f"{file_stem}_detailed_coeffs.csv"), index=False)

    # Save scalograms for each level in the appropriate folder
    for i, dc in enumerate(detail_coeffs):
        level_dir = os.path.join(output_base_dir, f"dc{i+1}", class_folder)
        os.makedirs(level_dir, exist_ok=True)
        scales = np.arange(1, 128)
        cwt_coeffs, freqs = pywt.cwt(dc, scales, 'morl', 1/sampling_freq)
        plt.figure(figsize=(10, 6))
        extent = [0, len(dc) / sampling_freq, freqs[-1]/1e6, freqs[0]/1e6]
        plt.imshow(np.abs(cwt_coeffs), extent=extent, aspect='auto', cmap='jet', origin='lower')
        plt.title(f'Scalogram (CWT of Level {i+1} Detail Coefficients)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (MHz)')
        plt.colorbar(label='Magnitude')
        plt.savefig(os.path.join(level_dir, f"{file_stem}.png"))
        plt.close()

def main():
    try:
        # Define input and output base directories using absolute path
        input_base_dir = r"D:\Python Project\Data for Classification"
        output_base_dir = r"D:\Python Project\Scalograms"
        
        print(f"Debug: Input directory: {input_base_dir}")
        print(f"Debug: Output directory: {output_base_dir}")
        
        # First check for files with 1000 datapoints
        files_with_1000 = check_data_points(input_base_dir)
        
        # Ask user if they want to continue with processing
        if files_with_1000:
            print("\nFound files with 1000 datapoints. These files should have 10000 datapoints.")
            response = input("Do you want to continue processing? (y/n): ")
            if response.lower() != 'y':
                print("Processing stopped.")
                return
        
        # List contents of input directory
        print("\nContents of input directory:")
        try:
            for item in os.listdir(input_base_dir):
                item_path = os.path.join(input_base_dir, item)
                if os.path.isdir(item_path):
                    print(f"Directory: {item}")
                    # List contents of subdirectory
                    for subitem in os.listdir(item_path):
                        print(f"  - {subitem}")
                else:
                    print(f"File: {item}")
        except Exception as e:
            print(f"Error listing directory contents: {str(e)}")
            print("Debug: Full error details:")
            print(traceback.format_exc())
        
        # Define folder names
        folders = ["TN100sig", "TF100sig", "NN100sig", "NF100sig"]
        
        # Process each folder
        for folder in folders:
            print(f"\nProcessing {folder}...")
            input_folder = os.path.join(input_base_dir, folder)
            # No need for output_folder, pass output_base_dir and class_folder
            if not os.path.exists(input_folder):
                print(f"Warning: Input folder {input_folder} does not exist!")
                continue
            process_folder(input_folder, output_base_dir, folder)
            print(f"Completed processing {folder}")
            
        # After all processing
        print("\nFiles that do NOT have shape (10000, 2):")
        print("-" * 50)
        if bad_shape_files:
            for fname, shape in bad_shape_files:
                print(f"{fname}  shape: {shape}")
        else:
            print("All files have shape (10000, 2).")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print("Debug: Full error details:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()