import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
import glob
import pandas as pd

class GradCAM:
    def __init__(self, model, layer_name="conv5_block3_out"):
        """
        Initialize Grad-CAM with the model and target layer.
        For ResNet50, we typically use the last convolutional layer.
        """
        self.model = model
        self.layer_name = layer_name
        self.grad_model = Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
    
    def compute_heatmap(self, img_array, class_idx=None, eps=1e-8):
        """
        Compute the Grad-CAM heatmap for the input image.
        """
        # Get the gradient of the class output with respect to the feature map
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_output = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_output, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by their mean gradient
        conv_outputs = conv_outputs.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Generate heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + eps)
        
        return heatmap
    
    def overlay_heatmap(self, img_path, heatmap, alpha=0.4):
        """
        Overlay the heatmap on the original image.
        """
        # Load and preprocess the original image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        output = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
        return output

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Preprocess image for model input.
    """
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_best_fold(parent_dir):
    """
    Determine the best performing fold based on cross-validation metrics.
    """
    print("\nAnalyzing cross-validation metrics to find best fold...")
    
    # Load cross-validation metrics
    metrics_path = os.path.join(parent_dir, 'crossval_metrics.csv')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Cross-validation metrics not found at {metrics_path}")
    
    # Read metrics
    df = pd.read_csv(metrics_path)
    
    # Calculate mean of all metrics for each fold
    df['mean_score'] = df[['accuracy', 'precision', 'recall', 'f1']].mean(axis=1)
    
    # Find best fold
    best_fold = df.loc[df['mean_score'].idxmax()]
    print(f"\nBest performing fold: {best_fold['fold']}")
    print(f"Metrics:")
    print(f"  Accuracy:  {best_fold['accuracy']:.4f}")
    print(f"  Precision: {best_fold['precision']:.4f}")
    print(f"  Recall:    {best_fold['recall']:.4f}")
    print(f"  F1-score:  {best_fold['f1']:.4f}")
    
    return best_fold['fold']

def visualize_gradcam(model_path, test_dir, output_dir, num_samples=5):
    """
    Generate Grad-CAM visualizations for sample images from each class.
    IEEE journal standard format:
    - High resolution (300 DPI)
    - JPG format
    - Professional formatting
    - Clear, readable text
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set IEEE-style plotting parameters
    plt.rcParams.update({
        'font.family': 'serif',  # Use serif font instead of Times New Roman
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Load model
    print(f"\nLoading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model)
    
    # Store predictions and heatmaps for overall analysis
    all_predictions = []
    all_heatmaps = []
    all_true_labels = []
    
    # Process each class
    for class_name in ['Nomex', 'Teflon']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue
        
        # Get sample images
        img_paths = glob.glob(os.path.join(class_dir, '*.jpg')) + \
                   glob.glob(os.path.join(class_dir, '*.png'))
        img_paths = img_paths[:num_samples]
        
        print(f"\nProcessing {len(img_paths)} samples from {class_name}")
        
        for idx, img_path in enumerate(img_paths):
            # Preprocess image
            img_array = preprocess_image(img_path)
            
            # Get model prediction
            pred = model.predict(img_array)
            pred_class = "Nomex" if pred[0][0] < 0.5 else "Teflon"
            confidence = 1 - pred[0][0] if pred_class == "Nomex" else pred[0][0]
            
            # Store for overall analysis
            all_predictions.append(pred[0][0])
            all_true_labels.append(0 if class_name == "Nomex" else 1)
            
            # Compute heatmap
            heatmap = grad_cam.compute_heatmap(img_array)
            all_heatmaps.append(heatmap)
            
            # Overlay heatmap on original image
            output = grad_cam.overlay_heatmap(img_path, heatmap)
            
            # Create visualization with IEEE-standard formatting
            fig = plt.figure(figsize=(15, 5), dpi=300)
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            plt.title(f'Original Image\nTrue: {class_name}', 
                     fontsize=14, fontweight='bold', pad=10)
            plt.axis('off')
            
            # Heatmap
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap, cmap='jet')
            plt.title('Grad-CAM Heatmap', 
                     fontsize=14, fontweight='bold', pad=10)
            plt.axis('off')
            
            # Overlay
            plt.subplot(1, 3, 3)
            plt.imshow(output)
            plt.title(f'Overlay\nPred: {pred_class} ({confidence:.2%})', 
                     fontsize=14, fontweight='bold', pad=10)
            plt.axis('off')
            
            # Adjust layout
            plt.tight_layout(pad=2.0)
            
            # Save visualization in IEEE format
            output_path = os.path.join(output_dir, 
                                     f'gradcam_{class_name}_{idx+1}.jpg')
            plt.savefig(output_path, 
                       format='jpg',
                       dpi=300,
                       bbox_inches='tight')
            plt.close()
            
            print(f"Saved IEEE-format visualization for {os.path.basename(img_path)}")
    
    # Generate overall analysis with IEEE formatting
    print("\nGenerating overall analysis...")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_heatmaps = np.array(all_heatmaps)
    
    # Calculate mean heatmap for each class
    nomex_heatmaps = all_heatmaps[all_true_labels == 0]
    teflon_heatmaps = all_heatmaps[all_true_labels == 1]
    
    mean_nomex_heatmap = np.mean(nomex_heatmaps, axis=0)
    mean_teflon_heatmap = np.mean(teflon_heatmaps, axis=0)
    
    # Plot mean heatmaps with IEEE formatting
    fig = plt.figure(figsize=(12, 5), dpi=300)
    
    # Mean Nomex heatmap
    plt.subplot(1, 2, 1)
    plt.imshow(mean_nomex_heatmap, cmap='jet')
    plt.title('Mean Grad-CAM Heatmap - Nomex', 
             fontsize=16, fontweight='bold', pad=10)
    plt.axis('off')
    
    # Mean Teflon heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(mean_teflon_heatmap, cmap='jet')
    plt.title('Mean Grad-CAM Heatmap - Teflon', 
             fontsize=16, fontweight='bold', pad=10)
    plt.axis('off')
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save mean heatmaps in IEEE format
    plt.savefig(os.path.join(output_dir, 'mean_heatmaps.jpg'), 
                format='jpg',
                dpi=300,
                bbox_inches='tight')
    plt.close()
    
    # Save analysis summary with IEEE-style formatting
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("=== Grad-CAM Analysis Summary ===\n\n")
        f.write(f"Model: {os.path.basename(os.path.dirname(model_path))}\n")
        f.write(f"Total samples analyzed: {len(all_predictions)}\n")
        f.write(f"Nomex samples: {np.sum(all_true_labels == 0)}\n")
        f.write(f"Teflon samples: {np.sum(all_true_labels == 1)}\n\n")
        
        # Calculate mean confidence for each class
        nomex_conf = 1 - all_predictions[all_true_labels == 0]
        teflon_conf = all_predictions[all_true_labels == 1]
        
        f.write("Mean prediction confidence:\n")
        f.write(f"Nomex: {np.mean(nomex_conf):.2%}\n")
        f.write(f"Teflon: {np.mean(teflon_conf):.2%}\n")
        
        # Calculate mean heatmap intensity
        f.write("\nMean heatmap intensity:\n")
        f.write(f"Nomex: {np.mean(mean_nomex_heatmap):.4f}\n")
        f.write(f"Teflon: {np.mean(mean_teflon_heatmap):.4f}\n")
        
        # Add IEEE-style figure captions
        f.write("\n=== Figure Captions ===\n\n")
        f.write("Fig. 1. Individual Grad-CAM visualizations showing the model's attention patterns ")
        f.write("for sample images from each class. Each row shows the original image, ")
        f.write("the Grad-CAM heatmap, and the overlay of the heatmap on the original image.\n\n")
        
        f.write("Fig. 2. Mean Grad-CAM heatmaps for Nomex and Teflon classes, ")
        f.write("showing the average attention patterns across all samples in each class. ")
        f.write("The intensity of the heatmap indicates the importance of different regions ")
        f.write("in the model's decision-making process.\n")

def main():
    # Configuration
    PARENT_DIR = '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/models/Insulation_classifier_CV_Aug_20250529_210044/CV_Aug_20250529_210046/ResNet50/CV_Aug_20250529_210046/ResNet50'
    TEST_DIR = '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/Classifications_data/Insulation_classifier/Test'
    
    try:
        # Get best performing fold
        best_fold = get_best_fold(PARENT_DIR)
        MODEL_PATH = os.path.join(PARENT_DIR, best_fold, 'model.h5')
        OUTPUT_DIR = os.path.join(PARENT_DIR, 'gradcam_results')
        
        print("\n=== Grad-CAM Visualization for Insulation Classifier ===")
        print(f"Using best performing fold: {best_fold}")
        print(f"Model: {MODEL_PATH}")
        print(f"Test data: {TEST_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        
        visualize_gradcam(MODEL_PATH, TEST_DIR, OUTPUT_DIR, num_samples=5)
        print("\nVisualization complete! Check the results in:", OUTPUT_DIR)
        print("\nAnalysis files generated:")
        print("- Individual Grad-CAM visualizations for each sample")
        print("- mean_heatmaps.jpg (average attention maps for each class)")
        print("- analysis_summary.txt (detailed analysis of model behavior)")
        
    except Exception as e:
        print(f"\nError during visualization: {str(e)}")
        raise

if __name__ == '__main__':
    main() 