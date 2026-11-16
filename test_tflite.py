"""
Test TensorFlow Lite Model Inference
Compares Keras CNN accuracy with TFLite (Float32 and INT8) accuracy.
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
BATCH_SIZE = 64
CLASS_NAMES = ["Negative", "Positive"]

print("=" * 60)
print("TFLITE INFERENCE TEST")
print("=" * 60)
print()

# ===========================
# Load Test Data
# ===========================
print("Loading test dataset...")

def load_test_images():
    """Load and prepare test data (last 10% of dataset)."""
    images_neg = []
    images_pos = []
    
    # Load Negative
    for img_file in sorted(os.listdir(DATASET_PATH / "Negative")):
        try:
            img_path = DATASET_PATH / "Negative" / img_file
            img = tf.keras.preprocessing.image.load_img(
                str(img_path),
                target_size=(IMG_SIZE, IMG_SIZE),
                color_mode='rgb'
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images_neg.append(img_array)
        except:
            pass
    
    # Load Positive
    for img_file in sorted(os.listdir(DATASET_PATH / "Positive")):
        try:
            img_path = DATASET_PATH / "Positive" / img_file
            img = tf.keras.preprocessing.image.load_img(
                str(img_path),
                target_size=(IMG_SIZE, IMG_SIZE),
                color_mode='rgb'
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images_pos.append(img_array)
        except:
            pass
    
    images_neg = np.array(images_neg)
    images_pos = np.array(images_pos)
    
    # Split to get test set (last 10%)
    test_split = 0.1
    neg_test_count = int(len(images_neg) * test_split)
    pos_test_count = int(len(images_pos) * test_split)
    
    X_test = np.concatenate([
        images_neg[-neg_test_count:],
        images_pos[-pos_test_count:]
    ])
    y_test = np.concatenate([
        np.zeros(neg_test_count),
        np.ones(pos_test_count)
    ])
    
    return X_test, y_test

X_test, y_test = load_test_images()
print(f"✓ Test set loaded: {len(X_test)} images")
print(f"  Negative: {np.sum(y_test == 0)}")
print(f"  Positive: {np.sum(y_test == 1)}")
print()

# ===========================
# Test Keras Model
# ===========================
print("=" * 60)
print("KERAS CNN INFERENCE")
print("=" * 60)

keras_model_path = OUTPUT_DIR / "microcrack_cnn.keras"
if keras_model_path.exists():
    keras_model = tf.keras.models.load_model(str(keras_model_path))
    
    y_pred_keras = np.argmax(keras_model.predict(X_test), axis=1)
    keras_accuracy = np.mean(y_pred_keras == y_test)
    
    print(f"Keras Model Accuracy: {keras_accuracy:.4f}")
    print()
else:
    print("✗ Keras model not found")
    keras_accuracy = None

# ===========================
# Test Float32 TFLite
# ===========================
print("=" * 60)
print("FLOAT32 TFLITE INFERENCE")
print("=" * 60)

float_model_path = OUTPUT_DIR / "microcrack_float32.tflite"

if not float_model_path.exists():
    print(f"✗ Model not found: {float_model_path}")
else:
    try:
        interpreter = tf.lite.Interpreter(str(float_model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape:  {input_details[0]['shape']}")
        print(f"Input type:   {input_details[0]['dtype'].__name__}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type:  {output_details[0]['dtype'].__name__}")
        print()
        
        # Inference
        y_pred_float32 = []
        confidences_float32 = []
        
        for img in X_test:
            img_batch = np.expand_dims(img, 0).astype(input_details[0]['dtype'])
            
            interpreter.set_tensor(input_details[0]['index'], img_batch)
            interpreter.invoke()
            
            output = interpreter.get_tensor(output_details[0]['index'])
            pred_class = np.argmax(output[0])
            confidence = np.max(output[0])
            
            y_pred_float32.append(pred_class)
            confidences_float32.append(confidence)
        
        y_pred_float32 = np.array(y_pred_float32)
        confidences_float32 = np.array(confidences_float32)
        float32_accuracy = np.mean(y_pred_float32 == y_test)
        
        print(f"Float32 TFLite Accuracy: {float32_accuracy:.4f}")
        print(f"Avg Confidence: {np.mean(confidences_float32):.4f}")
        print()
        
    except Exception as e:
        print(f"✗ Float32 TFLite inference failed: {e}")
        print()
        float32_accuracy = None

# ===========================
# Test INT8 TFLite
# ===========================
print("=" * 60)
print("INT8 TFLITE INFERENCE")
print("=" * 60)

int8_model_path = OUTPUT_DIR / "microcrack_int8.tflite"

if not int8_model_path.exists():
    print(f"✗ Model not found: {int8_model_path}")
else:
    try:
        interpreter = tf.lite.Interpreter(str(int8_model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape:  {input_details[0]['shape']}")
        print(f"Input type:   {input_details[0]['dtype'].__name__}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type:  {output_details[0]['dtype'].__name__}")
        print()
        
        # Get quantization parameters
        input_scale = input_details[0]['quantization'][0]
        input_zero_point = input_details[0]['quantization'][1]
        
        output_scale = output_details[0]['quantization'][0]
        output_zero_point = output_details[0]['quantization'][1]
        
        print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
        print(f"Output quantization: scale={output_scale}, zero_point={output_zero_point}")
        print()
        
        # Inference
        y_pred_int8 = []
        confidences_int8 = []
        
        for img in X_test:
            # Quantize input
            img_quantized = (img / input_scale + input_zero_point).astype(input_details[0]['dtype'])
            img_batch = np.expand_dims(img_quantized, 0)
            
            interpreter.set_tensor(input_details[0]['index'], img_batch)
            interpreter.invoke()
            
            output_quantized = interpreter.get_tensor(output_details[0]['index'])
            # Dequantize output
            output = output_quantized.astype(np.float32)
            output = (output - output_zero_point) * output_scale
            
            pred_class = np.argmax(output[0])
            confidence = np.max(output[0])
            
            y_pred_int8.append(pred_class)
            confidences_int8.append(confidence)
        
        y_pred_int8 = np.array(y_pred_int8)
        confidences_int8 = np.array(confidences_int8)
        int8_accuracy = np.mean(y_pred_int8 == y_test)
        
        print(f"INT8 TFLite Accuracy: {int8_accuracy:.4f}")
        print(f"Avg Confidence: {np.mean(confidences_int8):.4f}")
        print()
        
    except Exception as e:
        print(f"✗ INT8 TFLite inference failed: {e}")
        print()
        int8_accuracy = None

# ===========================
# Sample Predictions
# ===========================
print("=" * 60)
print("SAMPLE PREDICTIONS (First 10 Test Images)")
print("=" * 60)
print()

if keras_accuracy is not None and float32_accuracy is not None and int8_accuracy is not None:
    print(f"{'Index':<6} {'Expected':<12} {'Keras':<12} {'Float32':<12} {'INT8':<12}")
    print("-" * 54)
    
    for i in range(min(10, len(X_test))):
        expected = CLASS_NAMES[int(y_test[i])]
        keras_pred = CLASS_NAMES[int(y_pred_keras[i])]
        float32_pred = CLASS_NAMES[int(y_pred_float32[i])]
        int8_pred = CLASS_NAMES[int(y_pred_int8[i])]
        
        print(f"{i:<6} {expected:<12} {keras_pred:<12} {float32_pred:<12} {int8_pred:<12}")
    
    print()

# ===========================
# Comparison Summary
# ===========================
print("=" * 60)
print("ACCURACY COMPARISON")
print("=" * 60)
print()

if keras_accuracy is not None:
    print(f"Keras CNN:          {keras_accuracy:.4f}")

if float32_accuracy is not None:
    print(f"Float32 TFLite:     {float32_accuracy:.4f}")
    if keras_accuracy is not None:
        diff = abs(keras_accuracy - float32_accuracy)
        print(f"  Difference from Keras: {diff:+.4f}")

if int8_accuracy is not None:
    print(f"INT8 TFLite:        {int8_accuracy:.4f}")
    if keras_accuracy is not None:
        diff = abs(keras_accuracy - int8_accuracy)
        print(f"  Difference from Keras: {diff:+.4f}")

print()

# Save comparison
comparison = {
    'test_set_size': int(len(X_test)),
    'keras_accuracy': float(keras_accuracy) if keras_accuracy else None,
    'float32_tflite_accuracy': float(float32_accuracy) if float32_accuracy else None,
    'int8_tflite_accuracy': float(int8_accuracy) if int8_accuracy else None
}

with open(RESULTS_DIR / "tflite_accuracy_comparison.json", 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"Comparison saved to: {RESULTS_DIR / 'tflite_accuracy_comparison.json'}")
print()

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("Next step: Run convert_model_to_header.py to generate ESP32 model data")
print("=" * 60)
