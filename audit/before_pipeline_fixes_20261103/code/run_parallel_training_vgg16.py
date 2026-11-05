import os
import sys
import psutil
import argparse
from datetime import datetime
import tensorflow as tf

def check_system_resources():
    """Check available system resources and determine safe number of parallel tasks."""
    print("Checking system resources...\n")
    
    # Get system memory info
    memory = psutil.virtual_memory()
    total_ram = memory.total / (1024**3)  # Convert to GB
    available_ram = memory.available / (1024**3)
    
    print("System Resources:")
    print(f"Total RAM: {total_ram:.1f} GB")
    print(f"Available RAM: {available_ram:.1f} GB\n")
    
    # Calculate safe number of parallel tasks
    # Assuming each training process needs ~4GB RAM
    safe_tasks = max(1, int(available_ram / 4))
    print(f"Safe number of parallel tasks: {safe_tasks}\n")
    
    return safe_tasks

def get_training_config():
    """Get training configuration for VGG16."""
    print("Selected dataset for training:")
    print("2A_Electrode_Classifier_Nomex (VGG16 with cross-validation and specialized augmentations)")
    print("Using fresh training with cross-validation\n")
    training_params = {
        'model_type': 'vgg16',
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 1e-5,
        'early_stopping_patience': 10,
        'n_splits': 5,
        'dropout_rate': 0.5,
        'l2_reg': 0.01
    }
    print("Training parameters:\n")
    print("2A_Electrode_Classifier_Nomex:")
    print("  Model: VGG16 with cross-validation and specialized augmentations")
    print(f"  Epochs: {training_params['epochs']}")
    print(f"  Batch size: {training_params['batch_size']}")
    print(f"  Learning rate: {training_params['learning_rate']}")
    print(f"  Early stopping patience: {training_params['early_stopping_patience']}")
    print(f"  Cross-validation folds: {training_params['n_splits']}")
    print(f"  Dropout rate: {training_params['dropout_rate']}")
    print(f"  L2 regularization: {training_params['l2_reg']}")
    print("  Fine-tuning: Enabled")
    print("  Cross-validation: Enabled")
    print("  Specialized augmentations: Enabled")
    print("\nSafety Measures:")
    print("1. Running only 2A_Electrode_Classifier_Nomex dataset")
    print("2. Using VGG16 model with cross-validation")
    print("3. Batch size set to 16 to save memory")
    print("4. Single worker to prevent overload")
    print("5. System resource checks enabled")
    print("6. Cross-validation for better evaluation")
    print("7. Specialized augmentations for scalograms")
    return training_params

def main():
    safe_tasks = check_system_resources()
    
    # Choose model type here
    model_type = "resnet50"  # or "vgg16"; you can set this dynamically or via input
    
    # Set training parameters based on model_type
    if model_type == "vgg16":
        training_params = {
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 1e-5,
            'early_stopping_patience': 10,
            'n_splits': 5,
            'dropout_rate': 0.5,
            'l2_reg': 0.01
        }
    elif model_type == "resnet50":
        training_params = {
            'epochs': 40,
            'batch_size': 16,
            'learning_rate': 5e-5,
            'early_stopping_patience': 10,
            'n_splits': 5,
            'dropout_rate': 0.5,
            'l2_reg': 0.01
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    response = input("\nDo you want to proceed with VGG16 training for 2A_Electrode_Classifier_Nomex? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        sys.exit(0)
    print("\nLaunching VGG16 training for 2A_Electrode_Classifier_Nomex in current terminal...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("models", f"2A_Electrode_Classifier_Nomex_VGG16_CV_Aug_{timestamp}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "Classifications_data", "2A_Electrode_Classifier_Nomex")
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory not found: {data_dir}")
        print("Please ensure the data directory exists at the correct location.")
        sys.exit(1)
    train_dir = os.path.join(data_dir, "Train")
    if not os.path.exists(train_dir):
        print(f"\nError: Train directory not found: {train_dir}")
        print("Please ensure the Train directory exists in the data directory.")
        sys.exit(1)
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')])
    if not class_names:
        print(f"\nError: No class directories found in {train_dir}")
        print("Please ensure there are class subdirectories in the Train directory.")
        sys.exit(1)
    print("\nVerifying train directory structure:")
    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            num_files = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"{class_name}: {num_files} images found in {class_dir}")
        else:
            print(f"Warning: {class_name} directory not found at {class_dir}")
    cmd = [
        sys.executable,
        "model_training_vgg16.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--model_type", model_type,
        "--epochs", str(training_params['epochs']),
        "--batch_size", str(training_params['batch_size']),
        "--learning_rate", str(training_params['learning_rate']),
        "--early_stopping_patience", str(training_params['early_stopping_patience']),
        "--n_splits", str(training_params['n_splits']),
        "--dropout_rate", str(training_params['dropout_rate']),
        "--l2_reg", str(training_params['l2_reg'])
    ]
    print("\nStarting training in current terminal...")
    print(f"Data directory: {data_dir}")
    print(f"Results will be saved in: {output_dir}")
    print("\nTips:")
    print("1. This terminal will show VGG16 training progress")
    print("2. Press Ctrl+C to stop training if needed")
    print("3. Monitor your system resources (CPU, RAM) while training")
    print("4. Cross-validation results will be saved separately for each fold")
    print("5. Augmentation samples will be saved for visualization")
    print("\nNote: We're running the 2A_Electrode_Classifier_Nomex with VGG16 using:")
    print("- Cross-validation (5 folds)")
    print("- Specialized scalogram augmentations")
    print("- Enhanced regularization")
    print("- Fresh training (no previous weights)")
    os.execv(sys.executable, cmd)

if __name__ == "__main__":
    main() 
    