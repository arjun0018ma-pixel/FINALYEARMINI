"""
Lightweight CNN Training for Microcrack Detection
This script trains a compact CNN suitable for ESP32 deployment via TensorFlow Lite.

Dataset: 40,000 images (20,000 Negative, 20,000 Positive)
Split: 80% train (32,000), 10% val (4,000), 10% test (4,000)
Input: 96x96 RGB
Output: Binary classification (Negative/Positive)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from pathlib import Path
import json

# ===========================
# Configuration
# ===========================
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
OUTPUT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"

IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
CLASS_NAMES = ["Negative", "Positive"]
SEED = 42

# Reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Create results directory
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ===========================
# Device Setup
# ===========================
print("=" * 60)
print("DEVICE CONFIGURATION")
print("=" * 60)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU(s) detected: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    device_type = "GPU"
else:
    print("✗ No GPU detected, using CPU")
    device_type = "CPU"

print(f"TensorFlow version: {tf.__version__}")
print()

# ===========================
# Dataset Loading & Preparation
# ===========================
print("=" * 60)
print("DATASET LOADING")
print("=" * 60)

def get_image_files_by_class():
    """Get lists of image file paths for each class (no loading)."""
    valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    negative_files = []
    positive_files = []
    
    print("Scanning Negative class...")
    neg_dir = DATASET_PATH / "Negative"
    for img_file in sorted(os.listdir(neg_dir)):
        if os.path.splitext(img_file)[1].lower() in valid_formats:
            negative_files.append(str(neg_dir / img_file))
    
    print(f"  Found: {len(negative_files)} images")
    
    print("Scanning Positive class...")
    pos_dir = DATASET_PATH / "Positive"
    for img_file in sorted(os.listdir(pos_dir)):
        if os.path.splitext(img_file)[1].lower() in valid_formats:
            positive_files.append(str(pos_dir / img_file))
    
    print(f"  Found: {len(positive_files)} images")
    
    return negative_files, positive_files

# Get file paths (don't load images yet)
negative_files, positive_files = get_image_files_by_class()

# Subsample per class to keep CPU training time reasonable
PER_CLASS_LIMIT = 2500
if len(negative_files) > PER_CLASS_LIMIT:
    negative_files = negative_files[:PER_CLASS_LIMIT]
if len(positive_files) > PER_CLASS_LIMIT:
    positive_files = positive_files[:PER_CLASS_LIMIT]

# Create label lists
negative_labels = [0] * len(negative_files)
positive_labels = [1] * len(positive_files)

# Combine and shuffle file paths with labels
all_files = negative_files + positive_files
all_labels = negative_labels + positive_labels

indices = np.random.permutation(len(all_files))
all_files = [all_files[i] for i in indices]
all_labels = [all_labels[i] for i in indices]

print()
print("Class Distribution:")
print(f"  Negative: {len(negative_files)} ({len(negative_files)/len(all_files)*100:.1f}%)")
print(f"  Positive: {len(positive_files)} ({len(positive_files)/len(all_files)*100:.1f}%)")
print()

# Calculate splits (80/10/10)
total = len(all_files)
train_size = int(total * 0.8)
val_size = int(total * 0.1)

train_files = all_files[:train_size]
train_labels = all_labels[:train_size]

val_files = all_files[train_size:train_size + val_size]
val_labels = all_labels[train_size:train_size + val_size]

test_files = all_files[train_size + val_size:]
test_labels = all_labels[train_size + val_size:]

print("=" * 60)
print("DATASET SPLITTING")
print("=" * 60)
print(f"Training set:   {len(train_files)} images")
print(f"Validation set: {len(val_files)} images")
print(f"Test set:       {len(test_files)} images")
print(f"Total:          {len(all_files)} images")
print()

# Calculate class weights
class_counts = np.array([len(negative_files), len(positive_files)])
if class_counts[0] != class_counts[1]:
    print("Class Imbalance Detected!")
    class_weights = {0: class_counts[1] / (2 * class_counts[0]),
                     1: class_counts[0] / (2 * class_counts[1])}
    print(f"  Applying class weights: {class_weights}")
else:
    print("✓ Classes are balanced")
    class_weights = None

print()

# Function to load image at inference time
def load_image_fn(filepath, label):
    """Load and preprocess image."""
    try:
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    except:
        # Return empty image on error (will be filtered out)
        return tf.zeros((IMG_SIZE, IMG_SIZE, 3)), label

# Load test set into memory for evaluation (smaller subset)
print("Loading test set into memory (for evaluation)...")
X_test = []
y_test = []
for filepath, label in zip(test_files, test_labels):
    try:
        img = keras.preprocessing.image.load_img(
            filepath,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='rgb'
        )
        img_array = keras.preprocessing.image.img_to_array(img) / 255.0
        X_test.append(img_array)
        y_test.append(label)
    except Exception as e:
        print(f"  Warning: Failed to load {filepath}: {e}")

X_test = np.array(X_test)
y_test = np.array(y_test)
print(f"✓ Test set loaded: {len(X_test)} images")
print()

# ===========================
# Data Augmentation
# ===========================
print("=" * 60)
print("DATA AUGMENTATION")
print("=" * 60)

# NOTE: Aggressive augmentation (rotation/translation/brightness/contrast)
# destroys the subtle class signal in these microcrack images and prevents
# the model from learning (accuracy stays ~50%). Kept to identity so the
# model can learn the distinguishing features. Flip is harmless but not
# required, so augmentation is disabled.
data_augmentation = lambda x, training=True: x

print("✓ Augmentation disabled (identity) - aggressive augmentation blocks learning")
print("  - Rotation: OFF")
print("  - Translation: OFF")
print("  - Horizontal flip: OFF")
print("  - Brightness: OFF")
print("  - Contrast: OFF")
print()

# ===========================
# Build Lightweight CNN
# ===========================
print("=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)

def build_lightweight_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Build a lightweight CNN suitable for embedded deployment.

    NOTE: BatchNormalization layers are intentionally OMITTED. An earlier
    version used BatchNorm and the model froze at ~50% accuracy / val_acc 0.
    This was traced to a BatchNorm training/inference mismatch in this Keras
    version. Removing BatchNorm makes the model learn reliably (~92% val).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(2, activation='softmax')
    ])
    
    return model

model = build_lightweight_cnn()
model.summary()

# Calculate model size
total_params = model.count_params()
print(f"\nModel Parameters: {total_params:,}")
print()

# ===========================
# Compile Model
# ===========================
print("=" * 60)
print("MODEL COMPILATION")
print("=" * 60)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss = keras.losses.SparseCategoricalCrossentropy()
metrics = [keras.metrics.SparseCategoricalAccuracy()]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print("✓ Model compiled with:")
print(f"  - Optimizer: Adam (lr=1e-3, decayed on plateau)")
print(f"  - Loss: SparseCategoricalCrossentropy")
print(f"  - Metrics: SparseCategoricalAccuracy")
print()

# ===========================
# Callbacks
# ===========================
checkpoint_path = OUTPUT_DIR / "microcrack_cnn.keras"

callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(checkpoint_path),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ===========================
# Training
# ===========================
print("=" * 60)
print("TRAINING")
print("=" * 60)
print()

# Use simpler tf.io approach for better compatibility
@tf.function
def load_and_preprocess_image(filepath):
    """Load and preprocess image using tf.io operations."""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img

def create_dataset(file_paths, labels, augment=True):
    """Create a tf.data.Dataset with proper typing."""
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.shuffle(buffer_size=min(1000, len(file_paths)))
    
    def load_fn(filepath, label):
        img = load_and_preprocess_image(filepath)
        if augment:
            img = data_augmentation(img, training=True)
        return img, label
    
    dataset = dataset.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

print("Creating training dataset...")
train_dataset = create_dataset(np.array(train_files), np.array(train_labels), augment=True)

print("Creating validation dataset...")
val_dataset = create_dataset(np.array(val_files), np.array(val_labels), augment=False)

print("✓ Datasets created successfully")
print()

# Train
print("Starting training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights if class_weights else None,
    verbose=1
)

print()
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
print(f"Best model saved to: {checkpoint_path}")
print()

# ===========================
# Evaluation on Test Set
# ===========================
print("=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

# Load best model
model = keras.models.load_model(str(checkpoint_path))

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

# Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print()

tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix:")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")
print()

# Save metrics
metrics_dict = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1),
    'confusion_matrix': {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    },
    'model_params': int(total_params),
    'device': device_type
}

with open(RESULTS_DIR / "cnn_metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)

print(f"Metrics saved to: {RESULTS_DIR / 'cnn_metrics.json'}")
print()

# ===========================
# Plot Training History
# ===========================
print("=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

try:
    import matplotlib.pyplot as plt
    
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CNN Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS_DIR / 'training_accuracy.png', dpi=100, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR / 'training_accuracy.png'}")
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS_DIR / 'training_loss.png', dpi=100, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR / 'training_loss.png'}")
    
    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.colorbar()
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')
    
    class_labels = ['Negative', 'Positive']
    plt.xticks([0, 1], class_labels)
    plt.yticks([0, 1], class_labels)
    
    plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=100, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR / 'confusion_matrix.png'}")
    
    plt.close('all')
    print()
    
except ImportError:
    print("✗ matplotlib not available, skipping plots")
    print()

# ===========================
# Summary
# ===========================
print("=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"Model saved:        {checkpoint_path}")
print(f"Model size:         ~{total_params * 4 / 1024 / 1024:.2f} MB (before quantization)")
print(f"Test Accuracy:      {accuracy:.4f}")
print(f"Test Precision:     {precision:.4f}")
print(f"Test Recall:        {recall:.4f}")
print(f"Test F1-Score:      {f1:.4f}")
print(f"Device:             {device_type}")
print()
print("Next step: Run convert_tflite.py to convert to TensorFlow Lite format")
print("=" * 60)
