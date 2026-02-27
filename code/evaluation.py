import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURATION ---
# Updated paths to match crossval_aggregate_evaluation.py
PARENT_DIR = '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/models/CV_Aug_20250529_210046/ResNet50'
MODEL_DIR = os.path.join(PARENT_DIR, 'fold_2')  # Example: evaluating fold 2
BEST_WEIGHTS = os.path.join(MODEL_DIR, 'best_weights.h5')
HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.json')
TEST_DIR = '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/Classifications_data/Insulation_classifier/Test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ['Nomex', 'Teflon']


# --- LOAD MODEL ---
def load_best_model(fold_dir):
    model_path = os.path.join(fold_dir, 'model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    return model

# --- FINE-TUNING INSPECTION ---
def inspect_fine_tuning(model, fold_dir=None):
    """
    Print and log which layers are trainable (i.e., were fine-tuned) in the model.
    Optionally save this info to a file in the fold directory.
    """
    trainable_layers = [(i, layer.name) for i, layer in enumerate(model.layers) if layer.trainable]
    total_layers = len(model.layers)
    n_trainable = len(trainable_layers)
    print(f"\n=== Fine-Tuning Inspection ===")
    print(f"Total layers in model: {total_layers}")
    print(f"Trainable (fine-tuned) layers: {n_trainable}")
    if n_trainable > 0:
        print("Indices and names of trainable layers:")
        for idx, name in trainable_layers:
            print(f"  {idx}: {name}")
        print("\nConclusion: This model WAS fine-tuned.\n")
    else:
        print("All layers are frozen (no fine-tuning performed).\n")
    # Optionally save to file
    if fold_dir:
        with open(os.path.join(fold_dir, 'fine_tuning_report.txt'), 'w') as f:
            f.write(f"Total layers: {total_layers}\n")
            f.write(f"Trainable layers: {n_trainable}\n")
            if n_trainable > 0:
                f.write("Indices and names of trainable layers:\n")
                for idx, name in trainable_layers:
                    f.write(f"  {idx}: {name}\n")
                f.write("\nConclusion: This model WAS fine-tuned.\n")
            else:
                f.write("All layers are frozen (no fine-tuning performed).\n")

# --- LOAD TEST DATA ---
def get_test_generator():
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found at {TEST_DIR}")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    print(f"\nTest set loaded: {test_generator.samples} images")
    print(f"Class mapping: {test_generator.class_indices}")
    return test_generator

# --- EVALUATION ---
def evaluate_model(model, test_generator, fold_dir):
    print(f"\nEvaluating model in {fold_dir}...")
    # Fine-tuning inspection
    inspect_fine_tuning(model, fold_dir)
    # Predict
    y_pred_probs = model.predict(test_generator)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Print metrics with better formatting
    print("\n=== Model Performance Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    # Classification report with better formatting
    print("\n=== Classification Report ===")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Save detailed classification report
    with open(os.path.join(fold_dir, 'detailed_classification_report.txt'), 'w') as f:
        f.write("=== Detailed Classification Report ===\n\n")
        f.write(f"Model: {os.path.basename(fold_dir)}\n")
        f.write(f"Total samples: {len(y_true)}\n\n")
        f.write("Per-class metrics:\n")
        for class_name in CLASS_NAMES:
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {report[class_name]['precision']:.4f}\n")
            f.write(f"  Recall:    {report[class_name]['recall']:.4f}\n")
            f.write(f"  F1-score:  {report[class_name]['f1-score']:.4f}\n")
            f.write(f"  Support:   {report[class_name]['support']}\n")
        f.write("\nOverall metrics:\n")
        f.write(f"  Accuracy:  {acc:.4f}\n")
        f.write(f"  Precision: {prec:.4f}\n")
        f.write(f"  Recall:    {rec:.4f}\n")
        f.write(f"  F1-score:  {f1:.4f}\n")
    
    # Enhanced confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar=False, annot_kws={'size': 18})
    plt.xlabel('Predicted', fontsize=18, labelpad=10)
    plt.ylabel('True', fontsize=18, labelpad=10)
    plt.title(f'Confusion Matrix - {os.path.basename(fold_dir)}', fontsize=20, pad=20)
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix in multiple formats
    for ext in ['png', 'jpg', 'tiff']:
        plt.savefig(os.path.join(fold_dir, f'confusion_matrix.{ext}'), 
                   dpi=300, bbox_inches='tight')
    plt.close()
    
    return acc, prec, rec, f1, cm

# --- LEARNING CURVE ---
def plot_learning_curves(fold_dirs):
    print("\nGenerating learning curves for all folds...")
    
    # Colors for different folds
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Set global font settings
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    # Create figure for individual fold plots
    fig_individual, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Create figure for combined plot
    fig_combined, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Store all histories for combined plot
    all_histories = []
    
    for idx, fold_dir in enumerate(fold_dirs):
        history_path = os.path.join(fold_dir, 'training_history.json')
        if not os.path.exists(history_path):
            print(f"Warning: No training history found for {fold_dir}")
            continue
            
        with open(history_path, 'r') as f:
            history = json.load(f)
            all_histories.append(history)
        
        # Plot individual fold
        fold_name = os.path.basename(fold_dir)
        epochs = range(1, len(history['accuracy']) + 1)
        
        # Accuracy
        ax1.plot(epochs, history['accuracy'], 
                label=f'{fold_name} Train', 
                color=colors[idx], 
                linestyle='-', 
                linewidth=2.5,
                alpha=0.7)
        ax1.plot(epochs, history['val_accuracy'], 
                label=f'{fold_name} Val', 
                color=colors[idx], 
                linestyle='--', 
                linewidth=2.5,
                alpha=0.7)
        
        # Loss
        ax2.plot(epochs, history['loss'], 
                label=f'{fold_name} Train', 
                color=colors[idx], 
                linestyle='-', 
                linewidth=2.5,
                alpha=0.7)
        ax2.plot(epochs, history['val_loss'], 
                label=f'{fold_name} Val', 
                color=colors[idx], 
                linestyle='--', 
                linewidth=2.5,
                alpha=0.7)
    
    # Customize individual plots
    for ax, title, ylabel in [(ax1, 'Learning Curves - Accuracy', 'Accuracy'),
                             (ax2, 'Learning Curves - Loss', 'Loss')]:
        ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=18, pad=20, fontweight='bold')
        ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='both', which='major', labelsize=14)
        # Make axis lines thicker
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        # Remove grid
        ax.grid(False)
    
    # Calculate and plot combined metrics
    if all_histories:
        # Get the maximum number of epochs
        max_epochs = max(len(h['accuracy']) for h in all_histories)
        epochs = range(1, max_epochs + 1)
        
        # Calculate mean and std for each epoch
        acc_mean = np.mean([h['accuracy'] + [h['accuracy'][-1]] * (max_epochs - len(h['accuracy'])) 
                           for h in all_histories], axis=0)
        acc_std = np.std([h['accuracy'] + [h['accuracy'][-1]] * (max_epochs - len(h['accuracy'])) 
                         for h in all_histories], axis=0)
        val_acc_mean = np.mean([h['val_accuracy'] + [h['val_accuracy'][-1]] * (max_epochs - len(h['val_accuracy'])) 
                              for h in all_histories], axis=0)
        val_acc_std = np.std([h['val_accuracy'] + [h['val_accuracy'][-1]] * (max_epochs - len(h['val_accuracy'])) 
                            for h in all_histories], axis=0)
        
        loss_mean = np.mean([h['loss'] + [h['loss'][-1]] * (max_epochs - len(h['loss'])) 
                            for h in all_histories], axis=0)
        loss_std = np.std([h['loss'] + [h['loss'][-1]] * (max_epochs - len(h['loss'])) 
                          for h in all_histories], axis=0)
        val_loss_mean = np.mean([h['val_loss'] + [h['val_loss'][-1]] * (max_epochs - len(h['val_loss'])) 
                               for h in all_histories], axis=0)
        val_loss_std = np.std([h['val_loss'] + [h['val_loss'][-1]] * (max_epochs - len(h['val_loss'])) 
                             for h in all_histories], axis=0)
        
        # Plot combined metrics with confidence intervals
        # Accuracy
        ax3.plot(epochs, acc_mean, label='Train', color='blue', linewidth=2.5)
        ax3.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, 
                        color='blue', alpha=0.2)
        ax3.plot(epochs, val_acc_mean, label='Validation', color='red', linewidth=2.5)
        ax3.fill_between(epochs, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, 
                        color='red', alpha=0.2)
        
        # Loss
        ax4.plot(epochs, loss_mean, label='Train', color='blue', linewidth=2.5)
        ax4.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, 
                        color='blue', alpha=0.2)
        ax4.plot(epochs, val_loss_mean, label='Validation', color='red', linewidth=2.5)
        ax4.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, 
                        color='red', alpha=0.2)
        
        # Customize combined plots
        for ax, title, ylabel in [(ax3, 'Combined Learning Curves - Accuracy', 'Accuracy'),
                                 (ax4, 'Combined Learning Curves - Loss', 'Loss')]:
            ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
            ax.set_title(title, fontsize=18, pad=20, fontweight='bold')
            ax.legend(fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=14)
            # Make axis lines thicker
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            # Remove grid
            ax.grid(False)
    
    # Add super titles with increased font size
    fig_individual.suptitle('Learning Curves for All Folds', fontsize=20, y=1.05, fontweight='bold')
    fig_combined.suptitle('Combined Learning Curves with Confidence Intervals', fontsize=20, y=1.05, fontweight='bold')
    
    # Save individual fold plots
    fig_individual.tight_layout()
    for ext in ['png', 'jpg', 'tiff']:
        fig_individual.savefig(os.path.join(PARENT_DIR, f'learning_curves_all_folds.{ext}'), 
                             dpi=300, bbox_inches='tight')
    
    # Save combined plot
    fig_combined.tight_layout()
    for ext in ['png', 'jpg', 'tiff']:
        fig_combined.savefig(os.path.join(PARENT_DIR, f'learning_curves_combined.{ext}'), 
                           dpi=300, bbox_inches='tight')
    
    plt.close('all')

# --- DATASET SIZE REPORTING ---
def print_dataset_sizes(train_dir, test_dir, aug_dir=None, val_split=0.2, img_size=(224,224), batch_size=32, save_dir=None):
    print("\n=== Dataset Size Report ===")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)
    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    aug_count = None
    if aug_dir and os.path.exists(aug_dir):
        aug_count = 0
        for root, dirs, files in os.walk(aug_dir):
            aug_count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # Print
    print(f"Training set:   {train_gen.samples} images")
    print(f"Validation set: {val_gen.samples} images")
    print(f"Test set:       {test_gen.samples} images")
    print(f"Number of classes: {train_gen.num_classes}")
    if aug_count is not None:
        print(f"Augmented dataset size (saved images): {aug_count}")
    # Save to file
    if save_dir:
        report_path = os.path.join(save_dir, 'dataset_size_report.txt')
        with open(report_path, 'w') as f:
            f.write("=== Dataset Size Report ===\n")
            f.write(f"Training set:   {train_gen.samples} images\n")
            f.write(f"Validation set: {val_gen.samples} images\n")
            f.write(f"Test set:       {test_gen.samples} images\n")
            f.write(f"Number of classes: {train_gen.num_classes}\n")
            if aug_count is not None:
                f.write(f"Augmented dataset size (saved images): {aug_count}\n")
            f.write("\n")

# --- MAIN ---
if __name__ == '__main__':
    print("\n=== Model Evaluation and Visualization ===")
    
    # Infer train directory from TEST_DIR
    TRAIN_DIR = os.path.join(os.path.dirname(TEST_DIR), 'Train')
    # Optionally look for augmented samples in parent directory
    AUG_DIR = os.path.join(os.path.dirname(PARENT_DIR), 'augmented_samples')
    print_dataset_sizes(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        aug_dir=AUG_DIR,
        val_split=0.2,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        save_dir=PARENT_DIR
    )
    
    # Get all fold directories
    fold_dirs = [os.path.join(PARENT_DIR, d) for d in os.listdir(PARENT_DIR) 
                if d.startswith('fold_') and os.path.isdir(os.path.join(PARENT_DIR, d))]
    fold_dirs.sort()
    
    # Filter out empty folds
    valid_fold_dirs = []
    for fold_dir in fold_dirs:
        model_path = os.path.join(fold_dir, 'model.h5')
        if os.path.exists(model_path):
            valid_fold_dirs.append(fold_dir)
        else:
            print(f"Skipping {fold_dir} (no model.h5 found)")
    
    if not valid_fold_dirs:
        print("No valid folds found for evaluation!")
        exit(1)
    
    print(f"\nFound {len(valid_fold_dirs)} valid folds: {[os.path.basename(d) for d in valid_fold_dirs]}")
    
    try:
        # Get test generator once
        test_generator = get_test_generator()
        
        # Evaluate each fold
        for fold_dir in valid_fold_dirs:
            model = load_best_model(fold_dir)
            metrics = evaluate_model(model, test_generator, fold_dir)
        
        # Generate learning curves for all folds
        plot_learning_curves(valid_fold_dirs)
        
        print("\n=== Evaluation Complete ===")
        print(f"Results saved in:")
        print(f"- Individual fold directories for per-fold results")
        print(f"- {PARENT_DIR} for combined learning curves:")
        print(f"  - learning_curves_all_folds.png/jpg/tiff (individual folds)")
        print(f"  - learning_curves_combined.png/jpg/tiff (combined with confidence intervals)")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise 