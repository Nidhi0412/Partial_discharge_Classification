
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.ndimage

class GradCAM:
    def __init__(self, model, layer_name="block5_conv3"):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
    
    def compute_heatmap(self, img_array, class_idx=None, eps=1e-8):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_output = predictions[:, class_idx]
        
        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + eps)
        return heatmap

    def overlay_heatmap(self, img_path, heatmap, alpha=0.4):
        img = plt.imread(img_path)
        if img.max() > 1.0: img = img / 255.0
        
        # Remove alpha channel if present
        if img.shape[-1] == 4:
            img = img[:, :, :3]
            
        heatmap_rescaled = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_rescaled]
        
        zoom_factors = [img.shape[0]/heatmap.shape[0], img.shape[1]/heatmap.shape[1], 1]
        jet_heatmap = scipy.ndimage.zoom(jet_heatmap, zoom_factors, order=1)
        
        output = (1 - alpha) * img + alpha * jet_heatmap
        return output

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16 as VGG16_Base

def build_vgg16_architecture():
    base_model = VGG16_Base(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def main():
    MODEL_PATH = '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/Result_insulation/VGG16/VGG16/fold_2/model.h5'
    WEIGHTS_PATH = '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/Result_insulation/VGG16/VGG16/fold_2/best_weights.h5'
    TEST_DIR = '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/Classifications_data/Insulation_classifier/Test'
    OUTPUT_DIR = '/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/My_Project/outputs/research_assets/gradcam'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading weights from {WEIGHTS_PATH}")
    model = build_vgg16_architecture()
    model.load_weights(WEIGHTS_PATH)
    grad_cam = GradCAM(model)

    classes = ['Nomex', 'Teflon']
    for cls in classes:
        cls_dir = os.path.join(TEST_DIR, cls)
        samples = [f for f in os.listdir(cls_dir) if f.endswith(('.png', '.jpg'))][:2]
        
        for i, sample in enumerate(samples):
            img_path = os.path.join(cls_dir, sample)
            img_array = preprocess_image(img_path)
            
            # Predict
            pred = model.predict(img_array)[0][0]
            pred_class = 'Teflon' if pred > 0.5 else 'Nomex'
            conf = pred if pred > 0.5 else 1-pred
            
            # Heatmap
            heatmap = grad_cam.compute_heatmap(img_array)
            output = grad_cam.overlay_heatmap(img_path, heatmap)
            
            # Save
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            img_orig = plt.imread(img_path)
            plt.imshow(img_orig)
            plt.title(f'Original: {cls}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(output)
            plt.title(f'Grad-CAM: {pred_class} ({conf:.2%})')
            plt.axis('off')
            
            save_path = os.path.join(OUTPUT_DIR, f'vgg16_gradcam_{cls}_{i+1}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
