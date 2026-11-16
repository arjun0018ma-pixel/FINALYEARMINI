"""
IMPROVED classifier (v2) on the PREVIOUS dataset (Negative / Positive).
=======================================================================
Improvement over embedded_ml/train_cnn_prevdataset.py (the "before" model):
  * More data : PER_CLASS_LIMIT (default 4000) instead of 2,500 per class.
  * Aug      : flips + brightness/contrast + rotation jitter (robustness).
  * Model    : ImageNet-pretrained MobileNetV2 (alpha=0.5, 96x96) transfer
               learning instead of a from-scratch 48x48 CNN.
  * Eval     : accuracy / precision / recall / F1 / ROC-AUC + confusion matrix.

Every artifact is a NEW file (originals untouched):
  embedded_ml/prev_microcrack_cnn_v2.keras
  results/prev_run/prev_cnn_v2_metrics.json
  results/prev_run/prev_cnn_v2_history.json
  results/prev_run/prev_cnn_v2_training.png
  results/prev_run/prev_cnn_v2_confusion_matrix.png
"""

import json
import os
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

IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 18
PER_CLASS_LIMIT = 4000
CLASS_NAMES = ["Negative", "Positive"]
SEED = 42
ALPHA = 0.5
FINE_TUNE_AT = 40     # unfreeze the last N layers after the head converges (0 = never)

np.random.seed(SEED)
tf.random.set_seed(SEED)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 60)
print("IMPROVED CNN v2 (previous dataset, MobileNetV2 transfer)")
print("=" * 60)
print(f"TensorFlow: {tf.__version__}")

# ===========================
# Dataset discovery
# ===========================
VALID_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


def get_files(class_name):
    class_dir = DATASET_PATH / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {class_dir}")
    files = [str(class_dir / n) for n in sorted(os.listdir(class_dir))
             if os.path.splitext(n)[1].lower() in VALID_FORMATS]
    print(f"  {class_name}: {len(files)} images found")
    return files


negative_files = get_files("Negative")[:PER_CLASS_LIMIT]
positive_files = get_files("Positive")[:PER_CLASS_LIMIT]

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
print(f"Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}")

# ===========================
# tf.data pipeline
# ===========================
AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _decode(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32) / 255.0


def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.10)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def make_ds(file_paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((np.array(file_paths), np.array(labels)))
    if training:
        ds = ds.shuffle(min(4000, len(file_paths)), seed=SEED)

    def _load(path, label):
        return _decode(path), tf.cast(label, tf.float32)

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


train_dataset = make_ds(train_files, train_labels, training=True)
val_dataset = make_ds(val_files, val_labels)

# ===========================
# Model: MobileNetV2 + head
# ===========================
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet', alpha=ALPHA)
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid', name="crack")(x)
model = models.Model(inputs, outputs, name="prev_microcrack_cnn_v2")
total_params = model.count_params()
print(f"Total params: {total_params:,}")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(name='acc'),
             keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='prec'),
             keras.metrics.Recall(name='rec')],
)

checkpoint_path = OUTPUT_DIR / "prev_microcrack_cnn_v2.keras"
callbacks = [
    keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor='val_loss',
                                    save_best_only=True, mode='min', verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=2, min_lr=1e-6, verbose=1),
]

print("=" * 60)
print("STAGE 1: training head (frozen encoder)")
print("=" * 60)
h1 = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS,
               callbacks=callbacks, verbose=1)

if FINE_TUNE_AT > 0:
    print("=" * 60)
    print(f"STAGE 2: fine-tuning last {FINE_TUNE_AT} encoder layers")
    print("=" * 60)
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_AT]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name='acc'),
                 keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='prec'),
                 keras.metrics.Recall(name='rec')],
    )
    h2 = model.fit(train_dataset, validation_data=val_dataset, epochs=6,
                   callbacks=callbacks, verbose=1)
else:
    h2 = None

model = keras.models.load_model(str(checkpoint_path))

# ===========================
# Evaluation
# ===========================
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)

X_test, y_test = [], []
for fp, lab in zip(test_files, test_labels):
    img = keras.preprocessing.image.load_img(fp, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb')
    X_test.append(keras.preprocessing.image.img_to_array(img) / 255.0)
    y_test.append(lab)
X_test = np.array(X_test)
y_test = np.array(y_test)

probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
y_pred = (probs >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, probs)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"Accuracy: {accuracy:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  "
      f"F1: {f1:.4f}  AUC: {auc:.4f}")

metrics_dict = {
    'run': 'previous_dataset_v2',
    'dataset': 'dataset/Negative + dataset/Positive',
    'model': f'MobileNetV2 (imagenet, alpha={ALPHA}) @ {IMG_SIZE}x{IMG_SIZE}',
    'per_class_limit': PER_CLASS_LIMIT,
    'img_size': IMG_SIZE,
    'epochs_trained': int(len(h1.history['loss']) + (len(h2.history['loss']) if h2 else 0)),
    'fine_tune_at': FINE_TUNE_AT,
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1),
    'roc_auc': float(auc),
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    'model_params': int(total_params),
    'device': 'CPU',
    'model_file': str(checkpoint_path),
}
with open(RESULTS_DIR / "prev_cnn_v2_metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)

hist = {f"s1_{k}": [float(x) for x in v] for k, v in h1.history.items()}
if h2:
    hist.update({f"s2_{k}": [float(x) for x in v] for k, v in h2.history.items()})
with open(RESULTS_DIR / "prev_cnn_v2_history.json", 'w') as f:
    json.dump(hist, f, indent=2)

# ===========================
# Plots
# ===========================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(h1.history['acc'], label='Train Acc')
    plt.plot(h1.history['val_acc'], label='Val Acc')
    if h2:
        off = len(h1.history['acc'])
        plt.plot(range(off, off + len(h2.history['acc'])), h2.history['acc'], label='Train Acc (ft)')
        plt.plot(range(off, off + len(h2.history['val_acc'])), h2.history['val_acc'], label='Val Acc (ft)')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.title('Improved CNN v2 (previous dataset) - Accuracy')
    plt.savefig(RESULTS_DIR / 'prev_cnn_v2_training.png', dpi=100, bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Improved CNN v2 - Confusion Matrix')
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')
    plt.xticks([0, 1], CLASS_NAMES); plt.yticks([0, 1], CLASS_NAMES)
    plt.savefig(RESULTS_DIR / 'prev_cnn_v2_confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close('all')
    print(f"Plots saved to {RESULTS_DIR}")
except ImportError:
    print("matplotlib not available, skipping plots")

print()
print("=" * 60)
print(f"IMPROVED CNN v2 DONE  Acc={accuracy:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
print("=" * 60)
