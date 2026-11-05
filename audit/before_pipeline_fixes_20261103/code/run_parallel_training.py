import os
import sys
import psutil
import argparse
from datetime import datetime

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
    """Get training configuration based on available datasets."""
    print("Selected dataset for training:")
    print("Electrode_classifier (RESNET50 with cross-validation and specialized augmentations)")
    print("Using fresh training with cross-validation\n")
    
    # Training parameters for new approach
    training_params = {
        'model_type': 'resnet50',
        'epochs': 40,
        'batch_size': 16,
        'learning_rate': 5e-5,
        'early_stopping_patience': 10,
        'n_splits': 5,
        'dropout_rate': 0.5,
        'l2_reg': 0.01
    }
    
    print("Training parameters:\n")
    print("Electrode_classifier:")
    print("  Model: RESNET50 with cross-validation and specialized augmentations")
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
    print("1. Running only Electrode_classifier dataset")
    print("2. Using RESNET50 model with cross-validation")
    print("3. Batch size set to 16 to save memory")
    print("4. Single worker to prevent overload")
    print("5. System resource checks enabled")
    print("6. Cross-validation for better evaluation")
    print("7. Specialized augmentations for scalograms")
    
    return training_params

def main():
    # Check system resources
    safe_tasks = check_system_resources()
    
    # Get training configuration
    training_params = get_training_config()
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with RESNET50 training for Electrode_classifier? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    print("\nLaunching RESNET50 training for Electrode_classifier in current terminal...")
    
    # Create timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("models", f"Electrode_classifier_CV_Aug_{timestamp}")
    
    # Use absolute path for data directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "Classifications_data", "Electrode_classifier")
    
    # Verify data directory exists
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory not found: {data_dir}")
        print("Please ensure the data directory exists at the correct location.")
        sys.exit(1)
    
    # Verify train directory exists
    train_dir = os.path.join(data_dir, "Train")
    if not os.path.exists(train_dir):
        print(f"\nError: Train directory not found: {train_dir}")
        print("Please ensure the Train directory exists in the data directory.")
        sys.exit(1)
    
    # Get class names from train directory
    class_names = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d)) 
                         and not d.startswith('.')])
    
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
    
    # Verify model_training.py exists
    model_training_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_training.py")
    if not os.path.exists(model_training_path):
        print(f"\nError: model_training.py not found at {model_training_path}")
        print("Please ensure model_training.py is in the same directory as run_parallel_training.py")
        sys.exit(1)

    # Verify output directory is writable
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"\nError: Cannot write to output directory {output_dir}")
        print(f"Error details: {str(e)}")
        sys.exit(1)

    print("\nStarting training in current terminal...")
    print(f"Data directory: {data_dir}")
    print(f"Results will be saved in: {output_dir}")
    print(f"Using model training script: {model_training_path}")
    
    print("\nTips:")
    print("1. This terminal will show RESNET50 training progress")
    print("2. Press Ctrl+C to stop training if needed")
    print("3. Monitor your system resources (CPU, RAM) while training")
    print("4. Cross-validation results will be saved separately for each fold")
    print("5. Augmentation samples will be saved for visualization")
    
    print("\nNote: We're running the Electrode_classifier with RESNET50 using:")
    print("- Cross-validation (5 folds)")
    print("- Specialized scalogram augmentations")
    print("- Enhanced regularization")
    print("- Fresh training (no previous weights)")
    
    # Construct the training command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_training.py"),
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--model_type", "resnet50",
        "--epochs", str(training_params['epochs']),
        "--batch_size", str(training_params['batch_size']),
        "--learning_rate", str(training_params['learning_rate']),
        "--early_stopping_patience", str(training_params['early_stopping_patience']),
        "--n_splits", str(training_params['n_splits']),
        "--dropout_rate", str(training_params['dropout_rate']),
        "--l2_reg", str(training_params['l2_reg']),
        "--num_workers", "1"  # Single worker for safety
    ]
    
    # Execute the training command
    os.execv(sys.executable, cmd)

if __name__ == "__main__":
    main() 
    