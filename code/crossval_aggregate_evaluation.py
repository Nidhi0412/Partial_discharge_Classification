import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURATION ---
PARENT_DIR = '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/models/Insulation_classifier_CV_Aug_20250529_210044/CV_Aug_20250529_210046/ResNet50/CV_Aug_20250529_210046/ResNet50'  # Directory containing fold_1, fold_2, ...
TEST_DIR = '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/Classifications_data/Insulation_classifier/Test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ['Nomex', 'Teflon']  # Update as needed

# --- LOAD TEST DATA ---
def get_test_generator():
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    return test_generator

# --- EVALUATION ---
def evaluate_fold(fold_dir, test_generator):
    model_path = os.path.join(fold_dir, 'model.h5')
    if not os.path.exists(model_path):
        print(f"Model not found in {fold_dir}, skipping.")
        return None
    model = load_model(model_path)
    y_pred_probs = model.predict(test_generator)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    # Save classification report
    with open(os.path.join(fold_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False, annot_kws={'size': 18})
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('True', fontsize=18)
    plt.title(f'Confusion Matrix - {os.path.basename(fold_dir)}', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    for ext in ['png', 'jpg', 'tiff']:
        plt.savefig(os.path.join(fold_dir, f'confusion_matrix.{ext}'), dpi=300)
    plt.close()
    return {
        'fold': os.path.basename(fold_dir),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'report': report
    }

# --- MAIN ---
if __name__ == '__main__':
    if not os.path.exists(PARENT_DIR):
        raise FileNotFoundError(f'PARENT_DIR does not exist: {PARENT_DIR}')
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f'TEST_DIR does not exist: {TEST_DIR}')
    print("\n--- Cross-Validation Aggregate Evaluation ---")
    test_generator = get_test_generator()
    fold_dirs = [os.path.join(PARENT_DIR, d) for d in os.listdir(PARENT_DIR) if d.startswith('fold_') and os.path.isdir(os.path.join(PARENT_DIR, d))]
    fold_dirs.sort()
    results = []
    for fold_dir in fold_dirs:
        model_path = os.path.join(fold_dir, 'model.h5')
        if not os.path.exists(model_path):
            print(f"Skipping {fold_dir} (no model.h5 found or directory is empty).")
            continue
        print(f"Evaluating {fold_dir} ...")
        res = evaluate_fold(fold_dir, test_generator)
        if res:
            results.append(res)
    if not results:
        print("No folds were evaluated. Please check that model.h5 exists in at least one fold directory.")
        exit(1)
    # Aggregate results
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'report'} for r in results])
    summary = df.describe().loc[['mean', 'std']]
    print("\nPer-fold results:")
    print(df)
    print("\nSummary (mean ± std):")
    print(summary)
    # Save to CSV and TXT
    df.to_csv(os.path.join(PARENT_DIR, 'crossval_metrics.csv'), index=False)
    with open(os.path.join(PARENT_DIR, 'crossval_summary.txt'), 'w') as f:
        f.write('Per-fold results:\n')
        f.write(df.to_string(index=False))
        f.write('\n\nSummary (mean ± std):\n')
        f.write(summary.to_string())
    print(f"\nAggregate results saved in {PARENT_DIR}")