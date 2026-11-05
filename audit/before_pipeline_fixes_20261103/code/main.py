import os
import shutil
from sklearn.model_selection import train_test_split
from preprocessing import process_dataset
from My_Project.model_training_resnet50 import (create_vgg16_model, create_resnet50_model,
                          train_model, evaluate_model, plot_confusion_matrix)

# Define paths and parameters
DATA_DIR = "Data for Classification"
PROCESSED_DIR = "processed_data"
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DIR, "test")

# Class labels
CLASS_LABELS = ['NN100sig', 'NF100sig', 'TN100sig', 'TF100sig']

def setup_directories():
    """Create necessary directories for the pipeline."""
    for dir_path in [PROCESSED_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

def split_dataset():
    """Split the processed data into train, validation, and test sets."""
    for label in CLASS_LABELS:
        # Create class directories
        for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            os.makedirs(os.path.join(split_dir, label), exist_ok=True)
        
        # Get all files for this class
        class_dir = os.path.join(PROCESSED_DIR, label)
        files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        
        # Split files into train, validation, and test sets
        train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        # Move files to respective directories
        for file_list, target_dir in [(train_files, TRAIN_DIR), 
                                    (val_files, VAL_DIR), 
                                    (test_files, TEST_DIR)]:
            for file_name in file_list:
                src = os.path.join(class_dir, file_name)
                dst = os.path.join(target_dir, label, file_name)
                shutil.move(src, dst)

def main():
    # Setup directories
    setup_directories()
    
    # Process raw signals into scalograms
    print("Processing signals into scalograms...")
    process_dataset(DATA_DIR, PROCESSED_DIR, CLASS_LABELS)
    
    # Split dataset
    print("Splitting dataset into train, validation, and test sets...")
    split_dataset()
    
    # Train and evaluate VGG16
    print("\nTraining VGG16 model...")
    vgg16_model = create_vgg16_model(num_classes=len(CLASS_LABELS))
    vgg16_history, class_indices = train_model(vgg16_model, TRAIN_DIR, VAL_DIR)
    
    print("\nEvaluating VGG16 model...")
    vgg16_cm, vgg16_report = evaluate_model(vgg16_model, TEST_DIR, CLASS_LABELS)
    plot_confusion_matrix(vgg16_cm, CLASS_LABELS, "VGG16")
    
    print("\nVGG16 Classification Report:")
    print(vgg16_report)
    
    # Train and evaluate ResNet50
    print("\nTraining ResNet50 model...")
    resnet50_model = create_resnet50_model(num_classes=len(CLASS_LABELS))
    resnet50_history, _ = train_model(resnet50_model, TRAIN_DIR, VAL_DIR)
    
    print("\nEvaluating ResNet50 model...")
    resnet50_cm, resnet50_report = evaluate_model(resnet50_model, TEST_DIR, CLASS_LABELS)
    plot_confusion_matrix(resnet50_cm, CLASS_LABELS, "ResNet50")
    
    print("\nResNet50 Classification Report:")
    print(resnet50_report)

if __name__ == "__main__":
    main()