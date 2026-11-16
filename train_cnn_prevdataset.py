"""
Lightweight CNN training on the PREVIOUS dataset (Negative / Positive).
Runs alongside the original train_cnn.py without touching any of its files:
every artifact is written to a new, prefixed path.

Dataset: dataset/Negative + dataset/Positive (40,000 images available)
Subset : PER_CLASS_LIMIT per class (default 2,500) for CPU-friendly runtime
Input  : 48x48 RGB
Output : binary classification (Negative / Positive)

New outputs (none of the original files are overwritten):
  embedded_ml/prev_microcrack_cnn.keras
  results/prev_run/prev_cnn_metrics.json
  results/prev_run/prev_training_accuracy.png
  results/prev_run/prev_training_loss.png
  results/prev_run/prev_confusion_matrix.png
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from pathlib import Path

# ===========================
# Configuration
# ===========================
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
OUTPUT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results" / "prev_run"

IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 20
PER_CLASS_LIMIT = 2500
CLASS_NAMES = ["Negative", "Positive"]
SEED = 42
TAG = "prev"

np.random.seed(SEED)
tf.random.set_seed(SEED)

RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ===========================
# Device Setup
# ===========================
print("=" * 60)
print("DEVICE CONFIGURATION (previous-dataset run)")
print("=" * 60)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU(s) detected: {len(gpus)}")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception:
        pass
    device_type = "GPU"
else:
    print("No GPU detected, using CPU")
    device_type = "CPU"

print(f"TensorFlow version: {tf.__version__}")
print()

# ===========================
# Dataset Loading
# ===========================
print("=" * 60)
print("DATASET LOADING (previous dataset)")
print("=" * 60)

VALID_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


def get_files(class_name):
    class_dir = DATASET_PATH / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {class_dir}")
    files = []
    for name in sorted(os.listdir(class_dir)):
        if os.path.splitext(name)[1].lower() in VALID_FORMATS:
            files.append(str(class_dir / name))
    print(f"  {class_name}: {len(files)} images found")
    return files


negative_files = get_files("Negative")
positive_files = get_files("Positive")

negative_files = negative_files[:PER_CLASS_LIMIT]
positive_files = positive_files[:PER_CLASS_LIMIT]

all_files = negative_files + positive_files
all_labels = [0] * len(negative_files) + [1] * len(positive_files)

perm = np.random.permutation(len(all_files))
all_files = [all_files[i] for i in perm]
all_labels = [all_labels[i] for i in perm]

total = len(all_files)
train_size = int(total * 0.8)
val_size = int(total * 0.1)

train_files, train_labels = all_files[:train_size], all_labels[:train_size]
val_files, val_labels = all_files[train_size:train_size + val_size], all_labels[train_size:train_size + val_size]
test_files, test_labels = all_files[train_size + val_size:], all_labels[train_size + val_size:]

print()
print(f"Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}  Total: {total}")
print(f"  Negative: {len(negative_files)}  Positive: {len(positive_files)}")
print()

# ===========================
# Load test set into memory
# ===========================
print("Loading test set into memory...")
X_test, y_test = [], []
for filepath, label in zip(test_files, test_labels):
    try:
        img = keras.preprocessing.image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb')
        X_test.append(keras.preprocessing.image.img_to_array(img) / 255.0)
        y_test.append(label)
    except Exception as e:
        print(f"  Warning: failed to load {filepath}: {e}")

X_test = np.array(X_test)
y_test = np.array(y_test)
print(f"Test set loaded: {len(X_test)} images")
print()

# ===========================
# Model
# ===========================
print("=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)


def build_lightweight_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Same proven architecture as the original train_cnn.py (no BatchNorm)."""
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax'),
    ])


model = build_lightweight_cnn()
model.summary()
total_params = model.count_params()
print(f"\nModel Parameters: {total_params:,}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# ===========================
# Datasets
# ===========================
@tf.function
def load_and_preprocess_image(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32) / 255.0


def make_ds(file_paths, labels, training=False):
    paths = np.array(file_paths)
    labs = np.array(labels)
    ds = tf.data.Dataset.from_tensor_slices((paths, labs))
    if training:
        ds = ds.shuffle(buffer_size=min(2000, len(paths)), seed=SEED)

    def _load(path, label):
        return load_and_preprocess_image(path), label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = make_ds(train_files, train_labels, training=True)
val_dataset = make_ds(val_files, val_labels)
print("Datasets created.")

# ===========================
# Callbacks
# ===========================
checkpoint_path = OUTPUT_DIR / "prev_microcrack_cnn.keras"

callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(checkpoint_path), monitor='val_loss', save_best_only=True, mode='min', verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
]

# ===========================
# Training
# ===========================
print("=" * 60)
print("TRAINING (previous dataset)")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

print()
print(f"Best model saved to: {checkpoint_path}")

# ===========================
# Evaluation
# ===========================
print("=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

model = keras.models.load_model(str(checkpoint_path))
y_pred = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE, verbose=0), axis=1)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report,
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

metrics_dict = {
    'run': 'previous_dataset',
    'dataset': 'dataset/Negative + dataset/Positive',
    'per_class_limit': PER_CLASS_LIMIT,
    'img_size': IMG_SIZE,
    'epochs_trained': len(history.history['loss']),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1),
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    'model_params': int(total_params),
    'device': device_type,
    'model_file': str(checkpoint_path),
}

with open(RESULTS_DIR / "prev_cnn_metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print(f"Metrics saved: {RESULTS_DIR / 'prev_cnn_metrics.json'}")

with open(RESULTS_DIR / "prev_history.json", 'w') as f:
    json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

# ===========================
# Plots
# ===========================
print("=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Previous Dataset CNN - Training/Validation Accuracy')
    plt.legend(); plt.grid(True)
    plt.savefig(RESULTS_DIR / 'prev_training_accuracy.png', dpi=100, bbox_inches='tight')

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Previous Dataset CNN - Training/Validation Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(RESULTS_DIR / 'prev_training_loss.png', dpi=100, bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Previous Dataset CNN - Confusion Matrix')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')
    plt.xticks([0, 1], CLASS_NAMES); plt.yticks([0, 1], CLASS_NAMES)
    plt.savefig(RESULTS_DIR / 'prev_confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close('all')
    print(f"Plots saved to {RESULTS_DIR}")
except ImportError:
    print("matplotlib not available, skipping plots")

print()
print("=" * 60)
print("TRAINING SUMMARY (previous dataset)")
print("=" * 60)
print(f"Model:    {checkpoint_path}")
print(f"Accuracy: {accuracy:.4f}   F1: {f1:.4f}")
print(f"Device:   {device_type}")
print("Next: run deploy_prev_to_esp32cam.py to build TFLite + headers")
print("=" * 60)
