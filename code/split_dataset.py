import os
import shutil
import random
from pathlib import Path

def create_directory_structure(class_dir):
    """Create train and test directories inside each class directory."""
    train_dir = os.path.join(class_dir, 'train')
    test_dir = os.path.join(class_dir, 'test')
    
    # Create train and test directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    return train_dir, test_dir

def split_dataset(source_dir, test_split=0.2, seed=42):
    """
    Split the dataset into train and test sets within each class directory.
    
    Args:
        source_dir (str): Source directory containing class folders
        test_split (float): Proportion of images to use for testing
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    # Get all class directories (folders that don't start with '.')
    class_dirs = [d for d in os.listdir(source_dir) 
                 if os.path.isdir(os.path.join(source_dir, d)) 
                 and not d.startswith('.')]
    
    print(f"\nProcessing dataset: {os.path.basename(source_dir)}")
    print(f"Found {len(class_dirs)} classes: {', '.join(class_dirs)}")
    
    # Process each class
    for class_name in class_dirs:
        class_dir = os.path.join(source_dir, class_name)
        train_dir, test_dir = create_directory_structure(class_dir)
        
        # Get all images for this class
        images = [f for f in os.listdir(class_dir) 
                 if f.endswith('.png') and not f.startswith('.')]
        random.shuffle(images)
        
        # Calculate split index
        split_idx = int(len(images) * (1 - test_split))
        
        # Split into train and test
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Move images to respective directories
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_dir, img)
            shutil.move(src, dst)
            
        for img in test_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(test_dir, img)
            shutil.move(src, dst)
        
        print(f"\nClass: {class_name}")
        print(f"Total images: {len(images)}")
        print(f"Training images: {len(train_images)}")
        print(f"Testing images: {len(test_images)}")
        print(f"Train directory: {train_dir}")
        print(f"Test directory: {test_dir}")

def process_all_datasets():
    """Process all specified datasets."""
    # Define all dataset paths
    datasets = [
        '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/Classifications_data/Full_4Class_Classifier',
        '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/Classifications_data/2A_Electrode_Classifier_Nomex',
        '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/Classifications_data/2B_Electrode_Classifier_Teflon',
        '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/Classifications_data/Electrode_classifier'
    ]
    
    # Process each dataset
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            print("\n" + "="*80)
            print(f"Processing dataset: {dataset_path}")
            print("="*80)
            split_dataset(dataset_path)
        else:
            print(f"\nWarning: Dataset path does not exist: {dataset_path}")

def main():
    # Process all datasets
    print("Starting dataset splitting for all classifiers...")
    process_all_datasets()
    print("\nAll datasets have been processed!")

if __name__ == "__main__":
    main() 