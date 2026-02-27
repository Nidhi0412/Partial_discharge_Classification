import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from datetime import datetime

def create_vgg16_model():
    base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def create_resnet50_model():
    base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def find_last_conv_layer(model):
    """
    Automatically search for the last convolutional layer.
    """
    for layer in reversed(model.layers):
        # Check if layer is 4D output (batch, height, width, channels)
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not find a convolutional layer in the model.")

def make_gradcam_heatmap(model, img_array, layer_name, class_index=None):
    """
    Compute Grad-CAM heatmap for a specific class index.
    """
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        if predictions.shape[-1] == 1:
            # Binary Sigmoid
            score = predictions[0][0]
        else:
            # Multi-class Softmax
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            score = predictions[0][class_index]

    grads = tape.gradient(score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and Normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    
    return heatmap.numpy(), predictions.numpy().flatten()

def overlay_heatmap_with_peak(img_path, heatmap, alpha=0.6, cmap='jet'):
    """
    Create an overlay image with a peak marker using PIL.
    """
    # Load original image
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    
    # Safe normalization for visualization
    if np.max(heatmap) == 0:
        heatmap_norm = heatmap # All zeros
    else:
        heatmap_norm = heatmap # Already 0-1 from make_gradcam_heatmap logic? 
        # Wait, make_gradcam returns normalized 0-1.
        # But let's be safe.
        pass
        
    # Resize heatmap (which is a 2D numpy array 0..1) to original size
    # We use PIL for resizing
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize((width, height), resample=Image.BILINEAR)
    
    # Apply colormap
    # matplotlib colormap returns float 0..1 RGBA
    colormap = cm.get_cmap(cmap)
    heatmap_array = np.array(heatmap_img) / 255.0
    heatmap_colored = colormap(heatmap_array) 
    
    # Convert back to PIL RGB (drop alpha from colormap)
    heatmap_colored_uint8 = np.uint8(255 * heatmap_colored[:, :, :3])
    heatmap_pil = Image.fromarray(heatmap_colored_uint8)
    
    # Blend
    # Image.blend expects both images to be same mode/size. alpha=0.0 -> first, 1.0 -> second
    # We want original * (1-alpha) + heatmap * alpha
    overlay = Image.blend(img, heatmap_pil, alpha=alpha)
    
    # Find Peak (Argmax) in the resized heatmap array
    heatmap_resized_arr = np.array(heatmap_img)
    peak_idx = np.unravel_index(np.argmax(heatmap_resized_arr, axis=None), heatmap_resized_arr.shape)
    peak_y, peak_x = peak_idx # numpy is (row, col) ie (y, x)
    max_val = heatmap_resized_arr.max() / 255.0
    
    # Draw Marker - Only if there is non-zero activation? 
    # Or always draw the argmax (which defaults to 0,0 if flat)?
    # If flat 0, argmax is usually (0,0). Drawing there is confusing.
    # Let's Skip drawing if max_val is 0.
    
    if max_val > 0.01:
        draw = ImageDraw.Draw(overlay)
        r_outer = 6
        r_inner = 3
        # Black outline circle
        draw.ellipse([(peak_x - r_outer, peak_y - r_outer), (peak_x + r_outer, peak_y + r_outer)], outline='black', width=2)
        # White fill circle
        draw.ellipse([(peak_x - r_inner, peak_y - r_inner), (peak_x + r_inner, peak_y + r_inner)], fill='white', outline='white')
    
    return img, heatmap_pil, overlay, (peak_x, peak_y), max_val

def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)
    img_array = img_array / 255.0 # Rescale 1./255
    return img_array

import csv
from scipy import ndimage

def calculate_metrics(heatmap):
    """
    Calculate analysis metrics from the heatmap.
    """
    # 1. Peak Location (Argmax) & Value
    min_val, max_val, min_loc, max_loc = (0, heatmap.max(), (0,0), np.unravel_index(heatmap.argmax(), heatmap.shape)) # Removed cv2 dependency
    peak_idx = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    peak_y, peak_x = peak_idx
    
    # 2. Area Covered (Threshold > 0.5 of max)
    # Normalize to 0-1
    norm_map = heatmap / (heatmap.max() + 1e-7)
    area_pixels = np.sum(norm_map > 0.5)
    total_pixels = heatmap.size
    area_pct = (area_pixels / total_pixels) * 100
    
    # 3. Centroid (Center of Mass)
    try:
        cy, cx = ndimage.center_of_mass(norm_map)
    except:
        cy, cx = 0, 0
    
    return {
        'max_activation': heatmap.max(),
        'peak_x': int(peak_x),
        'peak_y': int(peak_y),
        'centroid_x': float(cx),
        'centroid_y': float(cy),
        'area_pct': area_pct
    }

def save_combined(original, heatmap_only, overlay, save_path_prefix, caption, true_label, pred_label):
    """
    Save separate images AND a combined side-by-side view with True Label annotation.
    """
    # Save individual
    original.save(f"{save_path_prefix}_orig.png")
    heatmap_only.save(f"{save_path_prefix}_heatmap.png")
    
    w, h = overlay.size
    banner_height = 40
    
    # 1. Annotate Overlay (Prediction)
    overlay_with_text = Image.new('RGB', (w, h + banner_height), (255, 255, 255))
    overlay_with_text.paste(overlay, (0, banner_height))
    draw_ov = ImageDraw.Draw(overlay_with_text)
    try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except: 
        try: font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except: font = ImageFont.load_default()

    draw_ov.text((10, 10), caption, fill='black', font=font)
    overlay_with_text.save(f"{save_path_prefix}_overlay.png")

    # 2. Create Combined Side-by-Side
    total_w = w * 3
    combined = Image.new('RGB', (total_w, h + banner_height), (255, 255, 255))
    
    combined.paste(original, (0, banner_height))
    combined.paste(heatmap_only, (w, banner_height))
    combined.paste(overlay, (w*2, banner_height))
    
    draw = ImageDraw.Draw(combined)
    
    # Determine Status Color
    is_correct = (true_label == pred_label)
    status_color = "green" if is_correct else "red"
    status_text = "CORRECT" if is_correct else "WRONG"
    
    # Title 1: Original Image + True Label
    true_text = f"True: {true_label} ({status_text})"
    draw.text((10, 10), true_text, fill=status_color, font=font)
    
    # Title 2: Heatmap
    draw.text((w + 10, 10), "Grad-CAM Heatmap", fill='black', font=font)
    
    # Title 3: Prediction Details
    draw.text((w*2 + 10, 10), caption, fill='black', font=font)
    
    combined.save(f"{save_path_prefix}_combined.png")

def main():
    parser = argparse.ArgumentParser(description='Generate Report-Ready Grad-CAM Triplets')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .h5 model file')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to directory containing test images')
    parser.add_argument('--output_dir', type=str, default='gradcam_outputs', help='Root output directory')
    parser.add_argument('--fold', type=str, default='fold_X', help='Current fold name')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Model Architecture')
    parser.add_argument('--layer', type=str, default=None, help='Target Conv Layer')
    parser.add_argument('--n_per_class', type=int, default=5, help='Number of samples to process per class')
    parser.add_argument('--alpha', type=float, default=0.6, help='Overlay transparency')
    args = parser.parse_args()
    
    # 1. Load Model (Rebuild + Load Weights)
    print(f"Loading weights from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print("Error: Model path does not exist.")
        return

    try:
        if args.arch == 'vgg16':
            model = create_vgg16_model()
        else:
            model = create_resnet50_model()
        print(f"Built {args.arch} architecture. Loading weights...")
        model.load_weights(args.model_path)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # 2. Determine Layer
    layer_name = args.layer
    if layer_name is None:
        if 'resnet' in args.arch.lower():
            layer_name = 'conv5_block3_out'
        elif 'vgg' in args.arch.lower():
            layer_name = 'block5_conv3'
        try:
            model.get_layer(layer_name)
        except ValueError:
            print(f"Default layer '{layer_name}' not found. Auto-detecting...")
            layer_name = find_last_conv_layer(model)
    print(f"Using Conv Layer: {layer_name}")

    # 3. Prepare classes & CSV
    classes = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    classes.sort()
    
    csv_rows = []
    
    # 4. Process Images
    for class_name in classes:
        print(f"\nProcessing Class: {class_name}")
        class_dir = os.path.join(args.input_dir, class_name)
        images = glob(os.path.join(class_dir, '*.png')) + glob(os.path.join(class_dir, '*.jpg'))
        
        if len(images) > args.n_per_class:
            selected_images = np.random.choice(images, args.n_per_class, replace=False)
        else:
            selected_images = images
            
        for img_path in selected_images:
            fname = os.path.basename(img_path)
            prefix = os.path.splitext(fname)[0]
            
            # Preprocess
            img_array = preprocess_image(img_path)
            
            if class_name.lower() == 'teflon':
                target_idx = 1
            else:
                target_idx = 0
                
            heatmap, predictions = make_gradcam_heatmap(model, img_array, layer_name, class_index=target_idx)
            
            if len(predictions) == 1:
                pred_prob = predictions[0]
                pred_label = classes[1] if pred_prob > 0.5 else classes[0]
                confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
            else:
                pred_idx = np.argmax(predictions)
                pred_label = classes[pred_idx]
                confidence = predictions[pred_idx]
                
            # Create Overlay & Get Peak Logic
            # Note: The overlay function does the resizing. We need the resized stats for the CSV to match pixels
            orig, heat, overlay, peak, max_activation = overlay_heatmap_with_peak(img_path, heatmap, alpha=args.alpha)
            (peak_x, peak_y) = peak
            
            # Use ndimage on the *resized* heatmap used in overlay for consistent metrics
            # heat is the PIL image of the heatmap. Convert to array [0..255]
            heat_arr = np.array(heat.convert('L')) # Grayscale for analysis
            
            # Recalculate metrics on the full-size heatmap
            norm_heat = heat_arr / 255.0
            area_cov = (np.sum(norm_heat > 0.5) / norm_heat.size) * 100
            try:
                cy, cx = ndimage.center_of_mass(norm_heat)
            except:
                cy, cx = 0, 0
            
            # Save artifacts
            output_subdir = os.path.join(args.output_dir, args.arch, args.fold, class_name)
            os.makedirs(output_subdir, exist_ok=True)
            save_path_prefix = os.path.join(output_subdir, prefix)
            caption = f"Pred: {pred_label} ({confidence:.2f}) | Peak: {peak}"
            
            save_combined(orig, heat, overlay, save_path_prefix, caption, true_label=class_name, pred_label=pred_label)
            
            print(f"  {fname}: Peak={peak}, Area={area_cov:.1f}%, Center=({cx:.1f}, {cy:.1f})")
            
            # Log to CSV
            csv_rows.append({
                'filename': fname,
                'true_class': class_name,
                'pred_class': pred_label,
                'confidence': confidence,
                'peak_x': peak_x,
                'peak_y': peak_y,
                'center_x': round(cx, 1),
                'center_y': round(cy, 1),
                'area_covered_pct': round(area_cov, 1),
                'max_activation': round(max_activation, 4)
            })

    # Save CSV
    if csv_rows:
        csv_path = os.path.join(args.output_dir, f"{args.arch}_{args.fold}_analysis.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nAnalysis CSV saved to: {csv_path}")

    print(f"Done. Images saved to {args.output_dir}")

if __name__ == '__main__':
    main()
