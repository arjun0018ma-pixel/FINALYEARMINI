import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==============================
# Configuration
# ==============================
DATASET_PATH = "dataset"   # Change if needed

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
SEED = 123

# ==============================
# Load Dataset
# ==============================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Split Validation and Test
val_batches = tf.data.experimental.cardinality(temp_ds)

test_ds = temp_ds.take(val_batches // 2)
val_ds = temp_ds.skip(val_batches // 2)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ==============================
# Data Augmentation
# ==============================

data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1)
])

# ==============================
# Build EfficientNetB0 Model
# ==============================

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(160,160,3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(160,160,3))

x = data_augmentation(inputs)

x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False)

x = GlobalAveragePooling2D()(x)

x = Dense(256, activation="relu")(x)

x = Dense(128, activation="relu")(x)

outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

# ==============================
# Compile
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# Callbacks
# ==============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_efficientnet.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# ==============================
# Train
# ==============================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# ==============================
# Evaluate
# ==============================

loss, accuracy = model.evaluate(test_ds)

print(f"\nEfficientNetB0 Test Accuracy: {accuracy*100:.2f}%")

# ==============================
# Classification Report
# ==============================

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(images, verbose=0)
    predictions = (predictions > 0.5).astype(int)

    y_true.extend(labels.numpy())
    y_pred.extend(predictions.flatten())

print(classification_report(y_true, y_pred))

# ==============================
# Accuracy Graph
# ==============================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("EfficientNetB0 Accuracy")

plt.legend()

plt.show()