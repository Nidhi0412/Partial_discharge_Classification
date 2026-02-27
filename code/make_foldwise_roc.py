import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import shutil

# --- CONFIGURATION DEFAULTS ---
# These can be overridden by command line arguments
DEFAULT_ROOT = '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project'
OUTPUT_SUBDIR = 'Result_insulation'  # Where to save the final figures

# styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

def load_fold_data(model_name, root_dir, folds=[2, 3, 4, 5]):
    """
    Load y_true and y_prob for specified folds.
    Returns a dictionary: {fold_num: {'y_true': array, 'y_prob': array}}
    """
    model_dir = os.path.join(root_dir, 'outputs', model_name) # e.g. .../outputs/VGG16
    fold_data = {}
    
    print(f"Loading data for {model_name} from: {model_dir}")
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return {}

    for fold in folds:
        fold_subdir = os.path.join(model_dir, f'fold_{fold}')
        
        # Construct possible filenames
        # Based on user check: y_true_foldX.npy, y_prob_foldX.npy
        y_true_path = os.path.join(fold_subdir, f'y_true_fold{fold}.npy')
        y_prob_path = os.path.join(fold_subdir, f'y_prob_fold{fold}.npy')
        
        if os.path.exists(y_true_path) and os.path.exists(y_prob_path):
            try:
                y_true = np.load(y_true_path)
                y_prob = np.load(y_prob_path)
                fold_data[fold] = {'y_true': y_true, 'y_prob': y_prob}
                print(f"  Fold {fold}: Loaded {len(y_true)} samples.")
            except Exception as e:
                print(f"  Fold {fold}: Error loading .npy files - {e}")
        else:
            print(f"  Fold {fold}: Missing files (Checked {y_true_path})")
            
    return fold_data

def generate_plot(model_name, fold_data, save_dir):
    """
    Generate and save the fold-wise ROC plot.
    """
    if not fold_data:
        print(f"No data loaded for {model_name}. Skipping plot.")
        return

    plt.figure(figsize=(10, 8))
    
    aucs = []
    summary_records = []
    sample_counts = []
    
    # For Mean Curve Calculation
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    
    # Define colors for folds to be distinct
    colors = {2: 'blue', 3: 'green', 4: 'red', 5: 'purple'}
    
    for fold in sorted(fold_data.keys()):
        y_true = fold_data[fold]['y_true']
        y_prob = fold_data[fold]['y_prob']
        sample_counts.append(len(y_true))
        
        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interpolate TPR for mean curve calculation
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        # Record metrics
        summary_records.append({
            'Model': model_name,
            'Fold': fold,
            'AUC': roc_auc,
            'Samples': len(y_true)
        })
        
        # Plot individual fold
        label = f'Fold-{fold} (AUC={roc_auc:.4f})'
        plt.plot(fpr, tpr, lw=2, label=label, color=colors.get(fold, 'black'), alpha=0.8)

    # Calculate Mean AUC and n
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_n = int(np.mean(sample_counts))
    
    # Calculate Mean TPR
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    
    # Highlight the Area (Fill)
    plt.fill_between(mean_fpr, mean_tpr, color='grey', alpha=0.2, label='Mean Area')
    
    # Add baseline
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Styling
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title(f'Fold-wise ROC Curves – {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    
    # Annotation text (Top-Left inside plot)
    annotation_text = (
        f"Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}\n"
        f"n = {mean_n} per fold (Folds 2–5)\n"
        f"Fold-1 excluded (missing data)"
    )
    
    plt.text(0.05, 0.82, annotation_text, 
             transform=plt.gca().transAxes, 
             fontsize=11, fontweight='medium', 
             verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Save
    os.makedirs(save_dir, exist_ok=True)
    base_filename = f"roc_foldwise_{model_name}"
    
    save_formats = ['png', 'jpg', 'tiff']
    for fmt in save_formats:
        save_path = os.path.join(save_dir, f"{base_filename}.{fmt}")
        # TIFF needs pillow, ensuring compatibility or falling back is good practice but standard matplotlib supports it usually
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    plt.close()
    
    return summary_records

def main():
    parser = argparse.ArgumentParser(description='Generate Fold-wise ROC Plots for Insulation Classifier')
    parser.add_argument('--root', type=str, default=DEFAULT_ROOT, help='Root directory of the project')
    parser.add_argument('--model', type=str, default='both', choices=['VGG16', 'ResNet50', 'both'], help='Model to process')
    
    args = parser.parse_args()
    
    # Determine which models to run
    if args.model == 'both':
        models_to_run = ['VGG16', 'ResNet50']
    else:
        models_to_run = [args.model]
        
    output_dir = os.path.join(args.root, OUTPUT_SUBDIR)
    all_summaries = []
    
    print("=== Fold-wise ROC Generation Started ===")
    
    for model in models_to_run:
        print(f"\nProcessing {model}...")
        fold_data = load_fold_data(model, args.root)
        summary = generate_plot(model, fold_data, output_dir)
        if summary:
            all_summaries.extend(summary)
            
    # Generate Combined Comparison Plot (Mean vs Mean)
    if 'VGG16' in models_to_run and 'ResNet50' in models_to_run:
        print("\nGenerating Combined VGG16 vs ResNet50 Comparison Plot...")
        generate_comparison_plot(args.root, output_dir)
        # GENERATE THE ALL-FOLDS PLOT
        print("Generating ALL FOLDS Combined Plot...")
        generate_all_folds_comparison(args.root, output_dir)
            
    # Save global summary csv
    if all_summaries:
        df = pd.DataFrame(all_summaries)
        csv_path = os.path.join(output_dir, 'auc_foldwise_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSummary saved to: {csv_path}")
        print(df)
    else:
        print("\nNo data found for any model.")

def generate_comparison_plot(root_dir, save_dir):
    """Generates Mean ROC of VGG16 vs ResNet50."""
    models = ['VGG16', 'ResNet50']
    colors = {'VGG16': 'blue', 'ResNet50': 'red'}
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 8))
    for model in models:
        fold_data = load_fold_data(model, root_dir)
        if not fold_data: continue
        tprs = []
        aucs = []
        for fold in fold_data:
            y_true = fold_data[fold]['y_true']
            y_prob = fold_data[fold]['y_prob']
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        label = f'{model} (AUC = {mean_auc:.4f} ± {std_auc:.4f})'
        plt.plot(mean_fpr, mean_tpr, color=colors[model], label=label, lw=2.5, alpha=0.9)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[model], alpha=0.1)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', label='Chance', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('Performance Comparison: VGG16 vs ResNet50', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12)
    base_filename = "roc_comparison_vgg16_resnet50"
    for fmt in ['png', 'jpg', 'tiff']:
        plt.savefig(os.path.join(save_dir, f"{base_filename}.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_folds_comparison(root_dir, save_dir):
    """Generates messy but requested plot with ALL 8 FOLDS."""
    models = ['VGG16', 'ResNet50']
    # Color scheme: VGG=Blues, ResNet=Reds
    styles = {
        'VGG16': {'cmap': plt.cm.Blues, 'linestyle': '-'},
        'ResNet50': {'cmap': plt.cm.Reds, 'linestyle': '--'}
    }
    
    plt.figure(figsize=(12, 10)) # Bigger figure for all lines
    
    for model in models:
        fold_data = load_fold_data(model, root_dir)
        if not fold_data: continue
        
        # We have folds 2,3,4,5. 
        # Map them to color intensities (0.4 to 1.0)
        folds = sorted(fold_data.keys())
        norm = plt.Normalize(min(folds), max(folds))
        
        for fold in folds:
            y_true = fold_data[fold]['y_true']
            y_prob = fold_data[fold]['y_prob']
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Distinct color for each fold within model family
            # Using discrete colors is safer than cmap logic sometimes
            if model == 'VGG16':
                color = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c'][fold-2] # Paired colors
                color = plt.cm.Blues(0.3 + (fold-2)*0.2)
            else:
                color = plt.cm.Reds(0.3 + (fold-2)*0.2)

            label = f'{model} Fold-{fold} (AUC={roc_auc:.4f})'
            plt.plot(fpr, tpr, label=label, color=color, 
                     linestyle=styles[model]['linestyle'], lw=2)

    plt.plot([0, 1], [0, 1], linestyle=':', lw=2, color='grey', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('All Folds Comparison: VGG16 vs ResNet50', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10, ncol=1) # Single column legend
    
    base_filename = "roc_all_folds_combined"
    for fmt in ['png', 'jpg', 'tiff']:
        plt.savefig(os.path.join(save_dir, f"{base_filename}.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
