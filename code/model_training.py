from keras.applications import VGG16, ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime
import os
import argparse
import multiprocessing as mp
from functools import partial
import json
from sklearn.model_selection import KFold, StratifiedKFold

def normalize_image(img):
    """Normalize image to [0,1] range."""
    return img / 255.0

def get_dataset_info(data_dir):
    """
    Get information about the dataset including number of classes and class names.
    """
    # Find train directory (case-insensitive)
    train_folder = next((d for d in os.listdir(data_dir) if d.lower() == 'train'), None)
    if not train_folder:
        raise ValueError(f"Could not find train folder in {data_dir}")
    
    train_path = os.path.join(data_dir, train_folder)
    
    # Get class names from the train directory
    class_names = sorted([d for d in os.listdir(train_path) 
                         if os.path.isdir(os.path.join(train_path, d)) 
                         and not d.startswith('.')])
    
    num_classes = len(class_names)
    is_binary = num_classes == 2
    
    print(f"\nDataset Information:")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {', '.join(class_names)}")
    print(f"Classification type: {'Binary' if is_binary else 'Multi-class'}")
    
    return num_classes, class_names, is_binary

def get_model_params(is_binary, num_classes):
    """
    Get model parameters with enhanced regularization and learning rate scheduling.
    """
    if is_binary:
        return {
            'dropout_rate': 0.5,  # Increased dropout
            'learning_rate': 5e-5,  # Reduced learning rate
            'batch_size': 32,
            'epochs': 40,  # Increased epochs since we have early stopping
            'early_stopping_patience': 12,  # Increased patience
            'lr_reduce_patience': 6,  # Increased patience for learning rate reduction
            'label_mode': 'binary',
            'activation': 'sigmoid',
            'loss': 'binary_crossentropy'
        }
    else:
        # Adjust parameters based on number of classes
        dropout_rate = min(0.6, 0.5 + (num_classes * 0.02))
        learning_rate = max(1e-5, 5e-5 / num_classes)
        
        return {
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': 32,
            'epochs': 50,
            'early_stopping_patience': 15,  # Increased patience for multi-class
            'lr_reduce_patience': 8,  # Increased patience for multi-class
            'label_mode': 'categorical',
            'activation': 'softmax',
            'loss': 'categorical_crossentropy'
        }

def get_data_augmentation():
    """
    Get enhanced data augmentation configuration with specialized techniques for scalograms.
    """
    # Standard image augmentations with reduced intensity for scalograms
    standard_augmentations = [
        tf.keras.layers.RandomRotation(0.1),  # Reduced from 0.2
        tf.keras.layers.RandomZoom(0.1),      # Reduced from 0.2
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomBrightness(0.1),  # Reduced from 0.2
        tf.keras.layers.RandomContrast(0.1),    # Reduced from 0.2
    ]
    
    # Custom augmentation layers for scalograms
    class FrequencyMasking(tf.keras.layers.Layer):
        def __init__(self, max_mask_width=0.15, **kwargs):  # Reduced from 0.2
            super().__init__(**kwargs)
            self.max_mask_width = max_mask_width
        
        def call(self, inputs, training=None):
            if not training:
                return inputs
            
            # Get image dimensions
            batch_size = tf.shape(inputs)[0]
            height = tf.shape(inputs)[1]
            width = tf.shape(inputs)[2]
            
            # Randomly select frequency bands to mask
            max_mask_width = tf.cast(tf.cast(height, tf.float32) * self.max_mask_width, tf.int32)
            mask_width = tf.random.uniform([], 1, max_mask_width, dtype=tf.int32)
            mask_start = tf.random.uniform([], 0, height - mask_width, dtype=tf.int32)
            
            # Create mask with smooth edges
            mask = tf.ones_like(inputs)
            for i in range(mask_width):
                alpha = tf.cast(i, tf.float32) / tf.cast(mask_width, tf.float32)
                mask = tf.tensor_scatter_nd_update(
                    mask,
                    tf.stack([
                        tf.repeat(tf.range(batch_size), width),
                        tf.tile(tf.ones([width], dtype=tf.int32) * (mask_start + i), [batch_size]),
                        tf.tile(tf.range(width), [batch_size])
                    ], axis=1),
                    tf.ones([batch_size * width, 3]) * (1 - alpha)
                )
            
            return inputs * mask
    
    class TimeMasking(tf.keras.layers.Layer):
        def __init__(self, max_mask_width=0.15, **kwargs):  # Reduced from 0.2
            super().__init__(**kwargs)
            self.max_mask_width = max_mask_width
        
        def call(self, inputs, training=None):
            if not training:
                return inputs
            
            # Get image dimensions
            batch_size = tf.shape(inputs)[0]
            height = tf.shape(inputs)[1]
            width = tf.shape(inputs)[2]
            
            # Randomly select time segments to mask
            max_mask_width = tf.cast(tf.cast(width, tf.float32) * self.max_mask_width, tf.int32)
            mask_width = tf.random.uniform([], 1, max_mask_width, dtype=tf.int32)
            mask_start = tf.random.uniform([], 0, width - mask_width, dtype=tf.int32)
            
            # Create mask with smooth edges
            mask = tf.ones_like(inputs)
            for i in range(mask_width):
                alpha = tf.cast(i, tf.float32) / tf.cast(mask_width, tf.float32)
                mask = tf.tensor_scatter_nd_update(
                    mask,
                    tf.stack([
                        tf.repeat(tf.range(batch_size), height),
                        tf.tile(tf.range(height), [batch_size]),
                        tf.tile(tf.ones([height], dtype=tf.int32) * (mask_start + i), [batch_size])
                    ], axis=1),
                    tf.ones([batch_size * height, 3]) * (1 - alpha)
                )
            
            return inputs * mask
    
    class PhaseShift(tf.keras.layers.Layer):
        def __init__(self, max_shift=0.2, **kwargs):
            super().__init__(**kwargs)
            self.max_shift = max_shift
        
        def call(self, inputs, training=None):
            if not training:
                return inputs
            
            batch_size = tf.shape(inputs)[0]
            width = tf.shape(inputs)[2]
            
            # Random phase shift with smooth transition
            shift = tf.random.uniform([], -self.max_shift, self.max_shift)
            width_float = tf.cast(width, tf.float32)
            shift_pixels = tf.cast(width_float * shift, tf.int32)
            
            # Apply circular shift with smooth edges
            shifted = tf.roll(inputs, shift=shift_pixels, axis=2)
            
            # Blend original and shifted images at edges
            blend_width = tf.cast(tf.cast(width, tf.float32) * 0.1, tf.int32)
            blend_mask = tf.ones_like(inputs)
            for i in range(blend_width):
                alpha = tf.cast(i, tf.float32) / tf.cast(blend_width, tf.float32)
                blend_mask = tf.tensor_scatter_nd_update(
                    blend_mask,
                    tf.stack([
                        tf.repeat(tf.range(batch_size), tf.shape(inputs)[1]),
                        tf.tile(tf.range(tf.shape(inputs)[1]), [batch_size]),
                        tf.tile(tf.ones([tf.shape(inputs)[1]], dtype=tf.int32) * i, [batch_size])
                    ], axis=1),
                    tf.ones([batch_size * tf.shape(inputs)[1], 3]) * alpha
                )
            
            return inputs * blend_mask + shifted * (1 - blend_mask)
    
    class FrequencyStretch(tf.keras.layers.Layer):
        def __init__(self, max_stretch=0.2, **kwargs):
            super().__init__(**kwargs)
            self.max_stretch = max_stretch
        
        def call(self, inputs, training=None):
            if not training:
                return inputs
            
            # Random stretch factor
            stretch = tf.random.uniform([], 1 - self.max_stretch, 1 + self.max_stretch)
            
            # Resize with smooth interpolation
            height_float = tf.cast(tf.shape(inputs)[1], tf.float32)
            new_height = tf.cast(height_float * stretch, tf.int32)
            resized = tf.image.resize(inputs, [new_height, tf.shape(inputs)[2]], 
                                     method=tf.image.ResizeMethod.BICUBIC)
            
            # Center crop or pad with smooth edges
            if stretch > 1:
                start = (new_height - tf.shape(inputs)[1]) // 2
                return resized[:, start:start + tf.shape(inputs)[1], :, :]
            else:
                pad_top = (tf.shape(inputs)[1] - new_height) // 2
                pad_bottom = tf.shape(inputs)[1] - new_height - pad_top
                return tf.pad(resized, [[0, 0], [pad_top, pad_bottom], [0, 0], [0, 0]], 
                              mode='SYMMETRIC')
    
    class ScaledGaussianNoise(tf.keras.layers.Layer):
        def __init__(self, stddev=0.05, **kwargs):  # Reduced from 0.1
            super().__init__(**kwargs)
            self.stddev = stddev
        
        def call(self, inputs, training=None):
            if not training:
                return inputs
            
            # Add noise scaled by input magnitude
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs + noise * tf.abs(inputs)  # Scale noise by input magnitude
    
    # Create augmentation model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = inputs
    
    # Apply standard augmentations
    for layer in standard_augmentations:
        x = layer(x)
    
    # Apply scalogram-specific augmentations with reduced intensity
    x = FrequencyMasking(max_mask_width=0.15)(x)
    x = TimeMasking(max_mask_width=0.15)(x)
    x = PhaseShift(max_shift=0.15)(x)
    x = FrequencyStretch(max_stretch=0.15)(x)
    
    # Add scaled noise
    x = ScaledGaussianNoise(stddev=0.05)(x)
    
    return tf.keras.Model(inputs, x, name='scalogram_augmentation')

def augment_dataset(dataset, augmentation_model):
    """
    Apply augmentation to the dataset while preserving tensor shapes.
    """
    def augment(image, label):
        # The image comes in as a batch, so we need to handle it properly
        # Remove batch dimension temporarily for augmentation
        image_shape = tf.shape(image)
        batch_size = image_shape[0]
        
        # Reshape to apply augmentation to each image in the batch
        images = tf.reshape(image, [-1, 224, 224, 3])
        
        # Apply augmentation
        try:
            print("Data augmentation started.")
            augmented = augmentation_model(images, training=True)
            print("Data augmentation finished.")
            
            # Reshape back to original batch size
            augmented = tf.reshape(augmented, [batch_size, 224, 224, 3])
            
            return augmented, label
        except Exception as e:
            print(f"Error during data augmentation: {e}")
            return image, label
    
    # Apply augmentation while preserving shapes
    augmented_dataset = dataset.map(
        augment,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return augmented_dataset

class ScalogramDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
    """Custom data generator that combines standard and scalogram-specific augmentations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create augmentation model for scalogram-specific augmentations
        self.augmentation_model = get_data_augmentation()
        # Store individual augmentation layers
        self.freq_masking = self.augmentation_model.layers[-4]
        self.time_masking = self.augmentation_model.layers[-3]
        self.phase_shift = self.augmentation_model.layers[-2]
        self.freq_stretch = self.augmentation_model.layers[-1]
        # Initialize storage
        self._last_batch = None
        self._last_augmentations = None
    
    def apply_scalogram_augmentations(self, x):
        """Apply scalogram-specific augmentations to a batch of images."""
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        # Store original batch
        self._last_batch = x.numpy() if hasattr(x, 'numpy') else x
        
        try:
            # Apply each augmentation separately
            augmentations = {}
            
            # Frequency masking
            freq_masked = self.freq_masking(x, training=True)
            augmentations['freq_masked'] = freq_masked.numpy() if hasattr(freq_masked, 'numpy') else freq_masked
            
            # Time masking
            time_masked = self.time_masking(x, training=True)
            augmentations['time_masked'] = time_masked.numpy() if hasattr(time_masked, 'numpy') else time_masked
            
            # Phase shift
            phase_shifted = self.phase_shift(x, training=True)
            augmentations['phase_shifted'] = phase_shifted.numpy() if hasattr(phase_shifted, 'numpy') else phase_shifted
            
            # Frequency stretch
            freq_stretched = self.freq_stretch(x, training=True)
            augmentations['freq_stretched'] = freq_stretched.numpy() if hasattr(freq_stretched, 'numpy') else freq_stretched
            
            # Store all augmentations
            self._last_augmentations = augmentations
            
            return freq_stretched
            
        except Exception as e:
            print(f"Warning: Error in scalogram augmentation: {str(e)}")
            return x
    
    def flow_from_directory(self, directory, *args, **kwargs):
        """Create a generator that applies both standard and scalogram augmentations."""
        # Get the standard generator
        generator = super().flow_from_directory(directory, *args, **kwargs)
        
        # Create a custom generator class that wraps the standard generator
        class AugmentedGenerator(tf.keras.preprocessing.image.DirectoryIterator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.parent_generator = generator
                self.scalogram_generator = self.image_data_generator
            
            def __next__(self):
                # Get batch from standard generator
                x, y = next(self.parent_generator)
                
                # Apply scalogram augmentations
                x = self.scalogram_generator.apply_scalogram_augmentations(x)
                
                return x, y
            
            @property
            def last_batch(self):
                return self.scalogram_generator._last_batch
            
            @property
            def last_augmentations(self):
                return self.scalogram_generator._last_augmentations
        
        # Create and return the custom generator
        return AugmentedGenerator(
            directory,
            self,
            target_size=generator.target_size,
            color_mode=generator.color_mode,
            classes=generator.classes,
            class_mode=generator.class_mode,
            batch_size=generator.batch_size,
            shuffle=generator.shuffle,
            seed=generator.seed,
            save_to_dir=generator.save_to_dir,
            save_prefix=generator.save_prefix,
            save_format=generator.save_format,
            follow_links=generator.follow_links,
            subset=generator.subset,
            interpolation=generator.interpolation,
            dtype=generator.dtype
        )

def get_data_generator(augment=True):
    """Get data generator with or without augmentation."""
    if augment:
        return ScalogramDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            validation_split=0.2,
            fill_mode='nearest'
        )
    else:
        return tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

def verify_saved_samples(aug_dir):
    """
    Verify that augmented samples were saved correctly and print statistics.
    """
    print("\nVerifying saved augmented samples...")
    
    # Check if directory exists
    if not os.path.exists(aug_dir):
        print(f"Error: Augmentation directory not found: {aug_dir}")
        return False
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(aug_dir) if os.path.isdir(os.path.join(aug_dir, d)) and d.startswith('class_')]
    
    if not class_dirs:
        print("Error: No class directories found")
        return False
    
    print(f"\nFound {len(class_dirs)} class directories:")
    for class_dir in class_dirs:
        class_path = os.path.join(aug_dir, class_dir)
        # Count different types of augmented images
        aug_types = {
            'original': len([f for f in os.listdir(class_path) if f.startswith('original_')]),
            'standard_aug': len([f for f in os.listdir(class_path) if f.startswith('standard_aug_')]),
            'freq_masked': len([f for f in os.listdir(class_path) if f.startswith('freq_masked_')]),
            'time_masked': len([f for f in os.listdir(class_path) if f.startswith('time_masked_')]),
            'phase_shifted': len([f for f in os.listdir(class_path) if f.startswith('phase_shifted_')]),
            'freq_stretched': len([f for f in os.listdir(class_path) if f.startswith('freq_stretched_')])
        }
        
        print(f"\n{class_dir}:")
        for aug_type, count in aug_types.items():
            print(f"  {aug_type}: {count} images")
        
        # Verify that we have samples for each augmentation type
        if any(count == 0 for count in aug_types.values()):
            print(f"Warning: Some augmentation types are missing in {class_dir}")
            return False
    
    print("\nAll augmentation types verified successfully!")
    return True

def save_augmented_samples(generator, output_dir, num_samples=5):
    """Save sample augmented images to visualize both standard and scalogram-specific augmentations."""
    # Create directory for augmented samples with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aug_dir = os.path.join(output_dir, f'augmented_samples_{timestamp}')
    os.makedirs(aug_dir, exist_ok=True)
    
    print(f"\nSaving augmented samples to: {aug_dir}")
    
    # Get class names and ensure proper mapping
    original_class_indices = generator.class_indices
    # Sort class names to ensure consistent ordering
    sorted_classes = sorted(original_class_indices.keys())
    # Create consistent mapping
    class_indices = {name: idx for idx, name in enumerate(sorted_classes)}
    class_names = {idx: name for name, idx in class_indices.items()}
    
    print("\nClass mapping:")
    for idx, name in sorted(class_names.items()):
        print(f"Class {idx}: {name}")
    
    # Create class directories first
    for class_name in sorted_classes:
        class_dir = os.path.join(aug_dir, f'class_{class_name}')
        os.makedirs(class_dir, exist_ok=True)
        print(f"Created directory for class {class_name}: {class_dir}")
    
    # Initialize samples_per_class with zeros first
    samples_per_class = {idx: 0 for idx in class_names.keys()}
    
    # Calculate total samples per class from the generator's directory
    train_dir = generator.directory
    for class_name in sorted_classes:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            num_files = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_idx = class_indices[class_name]
            samples_per_class[class_idx] = min(num_files, num_samples)
            print(f"Found {num_files} images for class {class_name}, will save {samples_per_class[class_idx]} samples")
    
    # Verify that all classes have samples
    if any(count == 0 for count in samples_per_class.values()):
        print("\nWarning: Some classes have no samples:")
        for class_idx in sorted(class_names.keys()):
            if samples_per_class[class_idx] == 0:
                print(f"Class {class_names[class_idx]}: No samples found in {os.path.join(train_dir, class_names[class_idx])}")
        return aug_dir
    
    print("\nSaving samples per class:")
    for class_idx in sorted(class_names.keys()):
        print(f"Class {class_names[class_idx]}: {samples_per_class[class_idx]} samples")
    
    # Reset generator to ensure we start from the beginning
    generator.reset()
    
    # Track saved samples per class
    saved_samples = {class_idx: 0 for class_idx in class_names.keys()}
    
    try:
        # Get a batch to verify generator is working
        print("\nVerifying generator...")
        test_batch = next(generator)
        print(f"Successfully loaded batch of shape: {test_batch[0].shape}")
        generator.reset()  # Reset again after test
        warned_last_batch = False
        warned_last_aug = False
        while any(saved_samples[class_idx] < samples_per_class[class_idx] for class_idx in class_names.keys()):
            try:
                # Get next batch
                images, labels = next(generator)
                
                # Convert labels to numpy if they aren't already
                if hasattr(labels, 'numpy'):
                    labels = labels.numpy()
                
                # Convert labels to our consistent indices
                if len(class_indices) == 2:
                    # For binary classification, ensure labels are 0 and 1
                    labels = labels.astype(int)
                else:
                    # For multi-class, map to our consistent indices
                    new_labels = np.zeros_like(labels)
                    for old_idx, class_name in original_class_indices.items():
                        new_labels[labels == old_idx] = class_indices[class_name]
                    labels = new_labels
                
                # Process each class in the batch
                for class_idx in sorted(class_names.keys()):
                    if saved_samples[class_idx] >= samples_per_class[class_idx]:
                        continue
                    
                    # Get indices for this class
                    class_indices_batch = np.where(labels == class_idx)[0]
                    
                    for idx in class_indices_batch:
                        if saved_samples[class_idx] >= samples_per_class[class_idx]:
                            break
                        
                        try:
                            # Get class name and directory
                            class_name = class_names[class_idx]
                            class_dir = os.path.join(aug_dir, f'class_{class_name}')
                            
                            # Defensive check for last_batch
                            if hasattr(generator, 'last_batch') and generator.last_batch is not None:
                                orig_img = generator.last_batch[idx]
                                orig_path = os.path.join(class_dir, f'original_{saved_samples[class_idx]}.png')
                                tf.keras.preprocessing.image.save_img(
                                    orig_path,
                                    (orig_img * 255).astype(np.uint8)
                                )
                                print(f"Saved original image for {class_name}: {orig_path}")
                            else:
                                if not warned_last_batch:
                                    print("Warning: generator does not have last_batch attribute or it is None. Skipping saving original images.")
                                    warned_last_batch = True
                            # Save standard augmented image
                            std_aug_path = os.path.join(class_dir, f'standard_aug_{saved_samples[class_idx]}.png')
                            tf.keras.preprocessing.image.save_img(
                                std_aug_path,
                                (images[idx] * 255).astype(np.uint8)
                            )
                            print(f"Saved standard augmented image for {class_name}: {std_aug_path}")
                            # Defensive check for last_augmentations
                            if hasattr(generator, 'last_augmentations') and generator.last_augmentations is not None:
                                for aug_type, aug_images in generator.last_augmentations.items():
                                    aug_img = aug_images[idx]
                                    aug_path = os.path.join(class_dir, f'{aug_type}_{saved_samples[class_idx]}.png')
                                    tf.keras.preprocessing.image.save_img(
                                        aug_path,
                                        (aug_img * 255).astype(np.uint8)
                                    )
                                    print(f"Saved {aug_type} image for {class_name}: {aug_path}")
                            else:
                                if not warned_last_aug:
                                    print("Warning: generator does not have last_augmentations attribute or it is None. Skipping saving scalogram-specific augmentations.")
                                    warned_last_aug = True
                            saved_samples[class_idx] += 1
                            print(f"Progress for {class_name}: {saved_samples[class_idx]}/{samples_per_class[class_idx]} samples saved")
                            
                        except Exception as e:
                            print(f"Warning: Error saving augmentations for {class_name} image {idx}: {str(e)}")
                            continue
                
            except StopIteration:
                print("\nReached end of dataset. Resetting generator...")
                generator.reset()
                continue
            except Exception as e:
                print(f"Warning: Error processing batch: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error during augmentation saving: {str(e)}")
    
    # Print final summary
    print(f"\nSaved augmented samples to: {aug_dir}")
    print("\nFinal summary of saved samples:")
    for class_idx in sorted(class_names.keys()):
        print(f"Class {class_names[class_idx]}: {saved_samples[class_idx]}/{samples_per_class[class_idx]} samples saved")
    
    # Verify saved samples
    if verify_saved_samples(aug_dir):
        print("\nAll augmented samples saved and verified successfully!")
    else:
        print("\nWarning: Some issues were found with saved samples. Please check the output directory.")
    
    return aug_dir

def create_vgg16_model(num_classes, params):
    """
    Create VGG16 model with adaptive classification head.
    Feature extraction only - no fine-tuning for now.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(params['dropout_rate'])(x)
    
    # Add intermediate dense layer for multi-class problems
    if num_classes > 2:
        x = Dense(512, activation='relu')(x)
        x = Dropout(params['dropout_rate'])(x)
    
    # Output layer with appropriate activation
    predictions = Dense(num_classes if num_classes > 2 else 1, 
                       activation=params['activation'])(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss=params['loss'],
        metrics=['accuracy']
    )
    
    return model

def create_resnet50_model(num_classes, params, load_weights=None):
    """
    Create ResNet50 model with enhanced regularization and dropout.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Fine-tuning: Unfreeze fewer layers if fine-tuning is enabled
    if params.get('fine_tuning', False):
        # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False
        
        # Unfreeze only the last 15 layers (reduced from 30)
        for layer in base_model.layers[-15:]:
            layer.trainable = True
        
        print("\nFine-tuning enabled:")
        print("Last 15 layers of ResNet50 are trainable")
        print("Rest of the layers are frozen")
    else:
        # Freeze all layers if no fine-tuning
        for layer in base_model.layers:
            layer.trainable = False
        print("\nFeature extraction only (no fine-tuning)")
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Add L2 regularization to dense layers
    regularizer = tf.keras.regularizers.l2(0.01)  # L2 regularization
    
    # Enhanced dropout configuration
    dropout_rate = params.get('dropout_rate', 0.5)
    print(f"\nUsing dropout rate: {dropout_rate}")
    
    # First dropout layer
    x = Dropout(dropout_rate)(x)
    
    # Add intermediate dense layer with regularization and dropout
    if num_classes > 2:
        x = Dense(512, activation='relu', 
                 kernel_regularizer=regularizer,
                 bias_regularizer=regularizer)(x)
        x = Dropout(dropout_rate)(x)  # Additional dropout after dense layer
        
        # Add another dense layer for more capacity with regularization
        x = Dense(256, activation='relu',
                 kernel_regularizer=regularizer,
                 bias_regularizer=regularizer)(x)
        x = Dropout(dropout_rate * 0.8)(x)  # Slightly reduced dropout for second layer
    
    # Output layer with regularization
    predictions = Dense(num_classes if num_classes > 2 else 1, 
                       activation=params['activation'],
                       kernel_regularizer=regularizer,
                       bias_regularizer=regularizer)(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Load previous weights if provided
    if load_weights and os.path.exists(load_weights):
        print(f"\nLoading previous weights from: {load_weights}")
        try:
            model.load_weights(load_weights)
            print("Successfully loaded previous weights")
            
            # Use an even lower learning rate when loading weights
            if params.get('fine_tuning', False):
                print("Using reduced learning rate for fine-tuning loaded weights")
                learning_rate = params['learning_rate'] * 0.25  # Reduced from 0.5
            else:
                learning_rate = params['learning_rate']
        except Exception as e:
            print(f"Warning: Could not load weights: {str(e)}")
            print("Continuing with ImageNet weights")
            learning_rate = params['learning_rate']
    else:
        learning_rate = params['learning_rate']
    
    # Use a lower learning rate for fine-tuning if not loading weights
    if params.get('fine_tuning', False) and not load_weights:
        learning_rate = params['learning_rate']  # Removed the 2x multiplier
    
    # Use Adam optimizer with gradient clipping
    optimizer = Adam(
        learning_rate=learning_rate,
        clipnorm=1.0  # Add gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss=params['loss'],
        metrics=['accuracy']
    )
    
    return model

def get_callbacks(fold_dir, params):
    """
    Create and return a list of callbacks for model training.
    
    Args:
        fold_dir (str): Directory to save model checkpoints and logs
        params (dict): Training parameters dictionary
        
    Returns:
        list: List of Keras callbacks
    """
    callbacks = [
        # Save best model weights
        ModelCheckpoint(
            filepath=os.path.join(fold_dir, 'best_weights.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=params['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=params['lr_reduce_patience'],
            min_lr=1e-7,
            verbose=1
        ),
        
        # Log training progress
        CSVLogger(
            os.path.join(fold_dir, 'training_log.csv'),
            append=False
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(fold_dir, 'logs'),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks

def get_data_generator(augment=True):
    """
    Get data generator with or without augmentation.
    """
    if augment:
        # Create base data generator with augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            validation_split=0.2,
            fill_mode='nearest'
        )
    else:
        # Create base data generator without augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
    
    return datagen

def load_dataset(data_dir, batch_size=32, img_size=(224, 224), validation_split=0.2, augment=True):
    """Load and preprocess the dataset with proper train/test split."""
    print("\nLoading dataset from:", data_dir)
    
    # Define the train and test directories
    train_dir = os.path.join(data_dir, 'Train')
    test_dir = os.path.join(data_dir, 'Test')
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise ValueError(f"Train or Test directory not found in {data_dir}")
    
    # Get class names from train directory
    class_names = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    
    if not class_names:
        raise ValueError(f"No class directories found in {train_dir}")
    
    print(f"\nFound {len(class_names)} classes: {', '.join(class_names)}")
    
    # Create data generators
    train_datagen = get_data_generator(augment=augment)
    test_datagen = get_data_generator(augment=False)  # No augmentation for test data
    
    # Load training data
    print("\nLoading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary' if len(class_names) == 2 else 'categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Load validation data
    print("\nLoading validation data...")
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary' if len(class_names) == 2 else 'categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # Load test data
    print("\nLoading test data...")
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary' if len(class_names) == 2 else 'categorical',
        shuffle=False
    )
    
    # Get dataset information
    num_train_samples = sum(len(files) for _, _, files in os.walk(train_dir))
    num_test_samples = sum(len(files) for _, _, files in os.walk(test_dir))
    
    print(f"\nDataset Information:")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {', '.join(class_names)}")
    print(f"Classification type: {'Binary' if len(class_names) == 2 else 'Multi-class'}")
    print(f"\nTraining samples: {num_train_samples}")
    print(f"Test samples: {num_test_samples}")
    print(f"Total samples: {num_train_samples + num_test_samples}")
    
    # Get image paths for cross-validation
    image_paths = []
    labels = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(train_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)
    
    return train_generator, validation_generator, test_generator, image_paths, labels, class_names

def train_model_with_cross_validation(
    model_fn,
    image_paths,
    labels,
    params,
    output_dir,
    class_names,
    test_generator=None
):
    """Train model using k-fold cross-validation."""
    print("\nStarting cross-validation training...")
    
    if not image_paths:
        raise ValueError("No images found in any class directory")
    
    print(f"\nTotal images found: {len(image_paths)}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Create StratifiedKFold for cross-validation
    kfold = StratifiedKFold(
        n_splits=params['n_splits'],
        shuffle=True,
        random_state=42
    )
    
    # Initialize metrics storage
    cv_metrics = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }
    
    fold_histories = []
    
    # Get base data generator
    train_datagen = get_data_generator(augment=True)
    val_datagen = get_data_generator(augment=False)
    
    # Create base output directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f'CV_Aug_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Infer train_dir from the first image path
    train_dir = os.path.dirname(os.path.dirname(image_paths[0]))
    print(f"\nUsing train directory: {train_dir}")
    
    # Train for each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths, labels)):
        print(f"\nFold {fold + 1}/{params['n_splits']}")
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")
        
        try:
            # Create fold directory
            fold_dir = os.path.join(experiment_dir, 'ResNet50', f'fold_{fold + 1}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # Create and compile model for this fold
            model = model_fn(len(class_names), params)
            
            # Create data generators for this fold
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(224, 224),
                batch_size=params['batch_size'],
                class_mode='binary' if len(class_names) == 2 else 'categorical',
                subset='training',
                shuffle=True,
                seed=42
            )
            
            val_generator = val_datagen.flow_from_directory(
                train_dir,
                target_size=(224, 224),
                batch_size=params['batch_size'],
                class_mode='binary' if len(class_names) == 2 else 'categorical',
                subset='validation',
                shuffle=False,
                seed=42
            )
            
            # Get callbacks for this fold
            fold_callbacks = get_callbacks(fold_dir, params)
    
    # Train the model
            print(f"\nTraining fold {fold + 1}")
    history = model.fit(
                        train_generator,
                        validation_data=val_generator,
                        epochs=params['epochs'],
                        callbacks=fold_callbacks,
                        verbose=1
            )
            
            # Store metrics
            cv_metrics['accuracy'].append(history.history['accuracy'])
            cv_metrics['val_accuracy'].append(history.history['val_accuracy'])
            cv_metrics['loss'].append(history.history['loss'])
            cv_metrics['val_loss'].append(history.history['val_loss'])
            fold_histories.append(history)
        
            # Save fold-specific model
            model_path = os.path.join(fold_dir, 'model.h5')
            model.save(model_path)
            print(f"Saved model for fold {fold + 1} to: {model_path}")
            
            # Save training history
            history_path = os.path.join(fold_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history.history, f, indent=4)
            
            # Clear session to free memory
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"Error in fold {fold + 1}: {str(e)}")
        continue
    
    # Save cross-validation summary
    cv_summary = {
        'metrics': {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
            for metric, values in cv_metrics.items()
        },
        'num_folds_completed': len(fold_histories),
        'timestamp': timestamp
    }
    
    summary_path = os.path.join(experiment_dir, 'ResNet50', 'cv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=4)
    
    return cv_metrics, fold_histories

def create_dataset_from_paths(image_paths, labels, batch_size):
    """
    Create a tf.data.Dataset from image paths and labels.
    """
    def load_and_preprocess_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train_model(model, train_dir, validation_dir, model_name, params, model_dir):
    """
    Main training function with cross-validation support.
    """
    # Create timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{model_name}_CV_{timestamp}"
    
    # Update model directory with experiment name
    model_dir = os.path.join(model_dir, experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Get class names
    class_names = sorted(os.listdir(train_dir))
    print(f"\nClasses: {class_names}")
    
    # Train with cross-validation
    cv_metrics, fold_histories = train_model_with_cross_validation(
        model,
        train_dir,
        model_name,
        params,
        model_dir,
        class_names,
        test_generator=None
    )
    
    # Save final configuration
    config = {
        'model_name': model_name,
        'experiment_name': experiment_name,
        'num_classes': len(class_names),
        'class_names': class_names,
        'cross_validation_metrics': cv_metrics,
        'training_params': params,
        'timestamp': timestamp
    }
    
    with open(os.path.join(model_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    return cv_metrics, fold_histories, class_names

def evaluate_model(model, test_dir, class_names, model_name, model_dir, batch_size=32):
    """
    Evaluate the model and generate performance metrics.
    """
    # Find test directory (case-insensitive)
    test_folder = next((d for d in os.listdir(test_dir) if d.lower() == 'test'), None)
    if not test_folder:
        raise ValueError(f"Could not find test folder in {test_dir}")
    
    test_path = os.path.join(test_dir, test_folder)
    
    # Print class distribution in test set
    print("\nClass distribution in test set:")
    for class_name in class_names:
        class_path = os.path.join(test_path, class_name)
        if os.path.exists(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"{class_name}: {num_images} images")
        else:
            print(f"Warning: {class_name} directory not found in test set")
    
    # Create test dataset
    test_dataset = image_dataset_from_directory(
        test_path,
        validation_split=None,
        seed=42,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode='categorical',
        class_names=class_names,  # Explicitly set class names
        shuffle=False
    ).map(lambda x, y: (normalize_image(x), y))
    
    # Print dataset information
    print(f"\nTest dataset class names: {test_dataset.class_names}")
    print(f"Number of batches: {len(test_dataset)}")
    
    # Get predictions
    predictions = model.predict(test_dataset)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = []
    for _, labels in test_dataset:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)
    y_true = np.argmax(y_true, axis=1)
    
    # Print prediction distribution
    print("\nPrediction distribution:")
    for i, class_name in enumerate(class_names):
        pred_count = np.sum(y_pred == i)
        true_count = np.sum(y_true == i)
        print(f"{class_name}:")
        print(f"  True samples: {true_count}")
        print(f"  Predicted samples: {pred_count}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names,
                                 digits=4)
    
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    plt.close()
    
    return cm, report

def plot_training_history(history, model_name, model_dir):
    """
    Plot and save training history.
    
    Args:
        history: Training history object
        model_name (str): Name of the model
        model_dir (str): Directory to save plots
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.close()

def compare_models(vgg16_history, resnet50_history, model_dir):
    """
    Compare training histories of both models.
    
    Args:
        vgg16_history: VGG16 training history
        resnet50_history: ResNet50 training history
        model_dir (str): Directory to save comparison plots
    """
    plt.figure(figsize=(12, 4))
    
    # Compare accuracy
    plt.subplot(1, 2, 1)
    plt.plot(vgg16_history.history['val_accuracy'], label='VGG16')
    plt.plot(resnet50_history.history['val_accuracy'], label='ResNet50')
    plt.title('Model Comparison - Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Compare loss
    plt.subplot(1, 2, 2)
    plt.plot(vgg16_history.history['val_loss'], label='VGG16')
    plt.plot(resnet50_history.history['val_loss'], label='ResNet50')
    plt.title('Model Comparison - Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'model_comparison.png'))
    plt.close()

def train_model_parallel(model_type, data_dir, num_classes, params, output_dir):
    """
    Train a single model with cross-validation and specialized augmentations.
    """
    print(f"\nTraining {model_type} model with cross-validation...")
    
    # Create model directory
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset
    num_classes, class_names, is_binary = get_dataset_info(data_dir)
    train_generator, validation_generator, test_generator, image_paths, labels, class_names = load_dataset(
        data_dir,
        batch_size=params['batch_size'],
        img_size=params['img_size'],
        augment=True
    )
    
    # Save initial augmented samples
    aug_dir = save_augmented_samples(
        train_generator,
        model_dir,
        num_samples=5
    )
    print(f"\nSaved initial augmented samples in: {aug_dir}")
    
    # Create fresh model (no weight loading)
    if model_type == 'VGG16':
        model_fn = lambda p: create_vgg16_model(num_classes, p)
    else:
        model_fn = lambda p: create_resnet50_model(num_classes, p)
    
    # Ensure params has all required fields
    training_params = {
        'img_size': params['img_size'],
        'batch_size': params['batch_size'],
        'epochs': params['epochs'],
        'learning_rate': params['learning_rate'],
        'early_stopping_patience': params['early_stopping_patience'],
        'lr_reduce_patience': params.get('lr_reduce_patience', params['early_stopping_patience'] // 2),
        'n_splits': params['n_splits'],
        'dropout_rate': params['dropout_rate'],
        'l2_reg': params['l2_reg'],
        'fine_tuning': params.get('fine_tuning', True),
        'activation': params.get('activation', 'sigmoid' if num_classes == 2 else 'softmax'),
        'loss': params.get('loss', 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy')
    }
    
    # Train with cross-validation
    cv_metrics, fold_histories = train_model_with_cross_validation(
        model_fn=model_fn,
        image_paths=image_paths,
        labels=labels,
        params=training_params,
        output_dir=model_dir,
        class_names=class_names,
        test_generator=test_generator
    )
    
    # Plot training history for each fold
    for fold, history in enumerate(fold_histories):
        plot_training_history(
            history,
            f"{model_type}_fold_{fold+1}",
            os.path.join(model_dir, 'cross_validation')
        )
    
    return model_type, cv_metrics, fold_histories

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train classification models with cross-validation and specialized augmentations')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to the dataset directory containing class folders')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save model outputs (default: models)')
    parser.add_argument('--model_type', type=str, choices=['vgg16', 'resnet50', 'both'], default='both',
                      help='Which model to train (default: both)')
    parser.add_argument('--num_workers', type=int, default=2,
                      help='Number of parallel workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                      help='Learning rate (default: 1e-5)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                      help='Patience for early stopping (default: 10)')
    parser.add_argument('--n_splits', type=int, default=5,
                      help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                      help='Dropout rate for regularization (default: 0.5)')
    parser.add_argument('--l2_reg', type=float, default=0.01,
                      help='L2 regularization factor (default: 0.01)')
    args = parser.parse_args()
    
    # Get dataset information
    num_classes, class_names, is_binary = get_dataset_info(args.data_dir)
    train_dir = os.path.join(args.data_dir, 'Train')
    
    # Create timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"CV_Aug_{timestamp}"
    
    # Create output directory with experiment name
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhanced model parameters with regularization
    params = {
        'img_size': (224, 224),
        'is_binary': is_binary,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'early_stopping_patience': args.early_stopping_patience,
        'dropout_rate': args.dropout_rate,
        'l2_reg': args.l2_reg,
        'n_splits': args.n_splits,  # Ensure n_splits is included in params
        'fine_tuning': True,
        'experiment_name': experiment_name,
        'label_mode': 'binary' if is_binary else 'categorical',
        'activation': 'sigmoid' if is_binary else 'softmax',
        'loss': 'binary_crossentropy' if is_binary else 'categorical_crossentropy',
        'lr_reduce_patience': args.early_stopping_patience // 2,
        'augmentation_intensity': 0.15,
        'use_cross_validation': True,
        'use_specialized_augmentation': True,
        'use_regularization': True
    }
    
    # Save experiment configuration
    config_path = os.path.join(output_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'parameters': params,
            'dataset_info': {
                'num_classes': num_classes,
                'class_names': class_names,
                'is_binary': is_binary
            }
        }, f, indent=4)
    
    print(f"\nStarting new experiment: {experiment_name}")
    print("Configuration:")
    print(f"- Cross-validation folds: {args.n_splits}")
    print(f"- Dropout rate: {args.dropout_rate}")
    print(f"- L2 regularization: {args.l2_reg}")
    print(f"- Specialized scalogram augmentations: Enabled")
    print(f"- Fine-tuning: Enabled")
    
    # Determine which models to train
    models_to_train = []
    if args.model_type in ['vgg16', 'both']:
        models_to_train.append('VGG16')
    if args.model_type in ['resnet50', 'both']:
        models_to_train.append('ResNet50')
    
    cv_metrics = {}
    
    if len(models_to_train) > 1:
        print(f"\nTraining {len(models_to_train)} models in parallel...")
        # Create a pool of workers
        with mp.Pool(processes=min(len(models_to_train), args.num_workers)) as pool:
            # Prepare arguments for parallel training
            train_func = partial(
                train_model_parallel,
                data_dir=args.data_dir,
                num_classes=num_classes,
                params=params,
                output_dir=output_dir
            )
            
            # Run training in parallel
            results = pool.map(train_func, models_to_train)
            
            # Process results
            for model_type, metrics, histories in results:
                cv_metrics[model_type] = metrics
                
                # Save cross-validation results
                cv_dir = os.path.join(output_dir, model_type, 'cross_validation')
                os.makedirs(cv_dir, exist_ok=True)
                with open(os.path.join(cv_dir, 'cv_summary.txt'), 'w') as f:
                    f.write(f"=== {model_type} Cross-Validation Summary ===\n\n")
                    for metric, values in metrics.items():
                        mean_value = np.mean(values)
                        std_value = np.std(values)
                        f.write(f"{metric}: {mean_value:.4f} ± {std_value:.4f}\n")
            
            # Compare models if both were trained
            if len(models_to_train) == 2:
                comparison_dir = os.path.join(output_dir, 'model_comparison')
                os.makedirs(comparison_dir, exist_ok=True)
                
                # Save model comparison metrics
                with open(os.path.join(comparison_dir, 'model_comparison.txt'), 'w') as f:
                    f.write("=== Model Comparison ===\n\n")
                    for model_type in models_to_train:
                        f.write(f"\n{model_type}:\n")
                        for metric, values in cv_metrics[model_type].items():
                            mean_value = np.mean(values)
                            std_value = np.std(values)
                            f.write(f"{metric}: {mean_value:.4f} ± {std_value:.4f}\n")
    else:
        # Train single model
        model_type = models_to_train[0]
        _, metrics, _ = train_model_parallel(
            model_type,
            args.data_dir,
            num_classes,
            params,
            output_dir
        )
        
        cv_metrics[model_type] = metrics
        
        # Save results
        cv_dir = os.path.join(output_dir, model_type, 'cross_validation')
        os.makedirs(cv_dir, exist_ok=True)
        with open(os.path.join(cv_dir, 'cv_summary.txt'), 'w') as f:
            f.write(f"=== {model_type} Cross-Validation Summary ===\n\n")
            for metric, values in metrics.items():
                mean_value = np.mean(values)
                std_value = np.std(values)
                f.write(f"{metric}: {mean_value:.4f} ± {std_value:.4f}\n")
    
    print(f"\nExperiment completed. Results saved in: {output_dir}")
    print("\nCross-validation metrics:")
    for model_type in models_to_train:
        print(f"\n{model_type}:")
        for metric, values in cv_metrics[model_type].items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            print(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")

if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)
    main()