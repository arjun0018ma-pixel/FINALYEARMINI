"""
Convert trained CNN to TensorFlow Lite format (Float32 and INT8 Quantized)
This creates deployment-ready models for ESP32.
"""

import os
import numpy as np
import tensorflow as tf
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
QUANTIZATION_SAMPLES = 300  # Number of images for calibration

print("=" * 60)
print("TFLITE CONVERSION SCRIPT")
print("=" * 60)
print()

# ===========================
# Load Trained Model
# ===========================
print("Loading trained Keras model...")
model_path = OUTPUT_DIR / "microcrack_cnn.keras"

if not model_path.exists():
    print(f"✗ Error: Model not found at {model_path}")
    print("Please run train_cnn.py first")
    exit(1)

model = tf.keras.models.load_model(str(model_path))
print(f"✓ Model loaded from {model_path}")
print()

# ===========================
# Convert to Float32 TFLite
# ===========================
print("=" * 60)
print("FLOAT32 CONVERSION")
print("=" * 60)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []  # No quantization for float32
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]

tflite_float_model = converter.convert()

float_model_path = OUTPUT_DIR / "microcrack_float32.tflite"
with open(str(float_model_path), 'wb') as f:
    f.write(tflite_float_model)

float_size = len(tflite_float_model)
print(f"✓ Float32 model saved: {float_model_path}")
print(f"  Size: {float_size:,} bytes ({float_size / 1024 / 1024:.2f} MB)")
print()

# ===========================
# Load Calibration Data
# ===========================
print("=" * 60)
print("PREPARING CALIBRATION DATA")
print("=" * 60)

def load_calibration_images(count=QUANTIZATION_SAMPLES):
    """Load representative images for quantization."""
    images = []
    loaded = 0
    
    # Load from both Negative and Positive
    for class_name in ['Negative', 'Positive']:
        class_path = DATASET_PATH / class_name
        if not class_path.exists():
            print(f"✗ Warning: {class_path} not found")
            continue
        
        for img_file in os.listdir(class_path):
            if loaded >= count:
                break
            
            try:
                img_path = class_path / img_file
                img = tf.keras.preprocessing.image.load_img(
                    str(img_path),
                    target_size=(IMG_SIZE, IMG_SIZE),
                    color_mode='rgb'
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                loaded += 1
            except:
                pass
    
    return np.array(images)

print(f"Loading {QUANTIZATION_SAMPLES} calibration images...")
calibration_data = load_calibration_images(QUANTIZATION_SAMPLES)
print(f"✓ Loaded {len(calibration_data)} calibration images")
print()

# ===========================
# Convert to INT8 TFLite
# ===========================
print("=" * 60)
print("INT8 QUANTIZATION CONVERSION")
print("=" * 60)

def representative_data_gen():
    """Generate representative dataset for quantization."""
    for i in range(len(calibration_data)):
        yield [calibration_data[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_data = representative_data_gen
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Converting with full integer quantization...")
try:
    tflite_quantized_model = converter.convert()
    int8_model_path = OUTPUT_DIR / "microcrack_int8.tflite"
    
    with open(str(int8_model_path), 'wb') as f:
        f.write(tflite_quantized_model)
    
    int8_size = len(tflite_quantized_model)
    print(f"✓ INT8 model saved: {int8_model_path}")
    print(f"  Size: {int8_size:,} bytes ({int8_size / 1024:.2f} KB)")
    print()
    
except Exception as e:
    print(f"✗ INT8 conversion failed: {e}")
    print("Attempting fallback to TFLITE_BUILTINS...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_data = representative_data_gen
    
    tflite_quantized_model = converter.convert()
    int8_model_path = OUTPUT_DIR / "microcrack_int8.tflite"
    
    with open(str(int8_model_path), 'wb') as f:
        f.write(tflite_quantized_model)
    
    int8_size = len(tflite_quantized_model)
    print(f"✓ Quantized model saved (dynamic range): {int8_model_path}")
    print(f"  Size: {int8_size:,} bytes ({int8_size / 1024:.2f} KB)")
    print()

# ===========================
# Verify TFLite Models
# ===========================
print("=" * 60)
print("TFLITE VERIFICATION")
print("=" * 60)

# Float32 verification
try:
    interpreter_float = tf.lite.Interpreter(str(float_model_path))
    interpreter_float.allocate_tensors()
    
    input_details = interpreter_float.get_input_details()
    output_details = interpreter_float.get_output_details()
    
    print("Float32 Model:")
    print(f"  Input shape:  {input_details[0]['shape']}")
    print(f"  Input type:   {input_details[0]['dtype'].__name__}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output type:  {output_details[0]['dtype'].__name__}")
    print(f"  Status: ✓ Valid")
    print()
    
except Exception as e:
    print(f"✗ Float32 model verification failed: {e}")
    print()

# INT8 verification
try:
    interpreter_int8 = tf.lite.Interpreter(str(int8_model_path))
    interpreter_int8.allocate_tensors()
    
    input_details = interpreter_int8.get_input_details()
    output_details = interpreter_int8.get_output_details()
    
    print("INT8 Model:")
    print(f"  Input shape:  {input_details[0]['shape']}")
    print(f"  Input type:   {input_details[0]['dtype'].__name__}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output type:  {output_details[0]['dtype'].__name__}")
    print(f"  Status: ✓ Valid")
    print()
    
except Exception as e:
    print(f"✗ INT8 model verification failed: {e}")
    print()

# ===========================
# Save Conversion Summary
# ===========================
print("=" * 60)
print("CONVERSION SUMMARY")
print("=" * 60)

summary = {
    'float32': {
        'file': str(float_model_path),
        'size_bytes': int(float_size),
        'size_kb': float(float_size / 1024),
        'size_mb': float(float_size / 1024 / 1024)
    },
    'int8': {
        'file': str(int8_model_path),
        'size_bytes': int(int8_size),
        'size_kb': float(int8_size / 1024),
        'size_mb': float(int8_size / 1024 / 1024)
    },
    'compression_ratio': float(float_size / int8_size),
    'calibration_samples': QUANTIZATION_SAMPLES
}

with open(RESULTS_DIR / "tflite_conversion_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Float32 Model Size: {float_size / 1024 / 1024:.2f} MB ({float_size:,} bytes)")
print(f"INT8 Model Size:    {int8_size / 1024:.2f} KB ({int8_size:,} bytes)")
print(f"Compression Ratio:  {float_size / int8_size:.2f}x smaller")
print()
print(f"Summary saved to: {RESULTS_DIR / 'tflite_conversion_summary.json'}")
print()

print("=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)
print("Next step: Run test_tflite.py to verify inference")
print("=" * 60)
