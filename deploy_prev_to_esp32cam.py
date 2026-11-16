"""
Deploy the PREVIOUS-DATASET model to the new ESP32-CAM firmware.
===============================================================
Takes embedded_ml/prev_microcrack_cnn.keras (trained by
train_cnn_prevdataset.py on dataset/Negative + dataset/Positive),
converts it to TFLite and writes the C headers consumed by
embedded_ml/esp32cam/src/main.cpp.

Every output is a NEW file (the original embedded_ml artifacts are untouched):
  embedded_ml/prev_microcrack_float32.tflite
  embedded_ml/prev_microcrack_int8.tflite
  embedded_ml/esp32cam/src/model_data.h
  embedded_ml/esp32cam/src/test_image.h
  results/prev_run/prev_tflite_conversion_summary.json
"""

import json
import os
import numpy as np
import tensorflow as tf
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results" / "prev_run"
ESP32CAM_SRC = SCRIPT_DIR / "esp32cam" / "src"

IMG_SIZE = 48
QUANTIZATION_SAMPLES = 300

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
ESP32CAM_SRC.mkdir(exist_ok=True, parents=True)

print("=" * 60)
print("DEPLOY PREVIOUS-DATASET MODEL -> ESP32-CAM")
print("=" * 60)

model_path = SCRIPT_DIR / "prev_microcrack_cnn.keras"
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path} (run train_cnn_prevdataset.py first)")

model = tf.keras.models.load_model(str(model_path))
print(f"Loaded model: {model_path}")


# ============================ header writers ============================
def write_model_header(tflite_bytes: bytes, out_path: Path, model_name="microcrack_model"):
    guard = f"_{out_path.stem.upper()}_H_"
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "// TensorFlow Lite model: microcrack detection (previous dataset)",
        f"// Model size: {len(tflite_bytes):,} bytes",
        "",
        "alignas(4) const unsigned char " + model_name + "_data[] = {",
    ]
    row = 16
    for i in range(0, len(tflite_bytes), row):
        chunk = tflite_bytes[i:i + row]
        s = "    " + ", ".join(f"0x{b:02x}" for b in chunk)
        if i + row < len(tflite_bytes):
            s += ","
        lines.append(s)
    lines.append("};")
    lines.append("")
    lines.append(f"const unsigned int {model_name}_data_len = {len(tflite_bytes)};")
    lines.append("")
    lines.append(f"#endif  // {guard}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {out_path.name}  ({out_path.stat().st_size/1024:.1f} KB, "
          f"symbol {model_name}_data[])")


def write_image_header(image_path: Path, out_path: Path):
    """Write the fallback frame as normalized float32 in [0,1] (model-agnostic:
    the firmware quantizes it with the model's own scale/zero_point)."""
    img = tf.keras.preprocessing.image.load_img(
        str(image_path), target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb")
    arr = (tf.keras.preprocessing.image.img_to_array(img) / 255.0).flatten().astype(np.float32)

    guard = f"_{out_path.stem.upper()}_H_"
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        f"// Fallback test frame: {image_path.name}",
        f"// Size: {IMG_SIZE}x{IMG_SIZE}x3, format: FLOAT32 in [0,1]",
        f"// Total elements: {len(arr)}",
        "",
        "const float test_image_data[] = {",
    ]
    row = 8
    for i in range(0, len(arr), row):
        chunk = arr[i:i + row]
        s = "    " + ", ".join(f"{v:.6f}f" for v in chunk)
        if i + row < len(arr):
            s += ","
        lines.append(s)
    lines.append("};")
    lines.append("")
    lines.append(f"const unsigned int test_image_data_len = {len(arr)};")
    lines.append("")
    lines.append(f"#endif  // {guard}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {out_path.name}  ({out_path.stat().st_size/1024:.1f} KB, "
          f"symbol test_image_data[] /* float */)")


# ---------------------------------------------------------------- Float32
print("\n[1/4] Float32 TFLite conversion...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
float_bytes = converter.convert()
float_path = SCRIPT_DIR / "prev_microcrack_float32.tflite"
float_path.write_bytes(float_bytes)
print(f"  {float_path.name}: {len(float_bytes):,} bytes ({len(float_bytes)/1024:.1f} KB)")

# ---------------------------------------------------------- Calibration set
print(f"\n[2/4] Loading {QUANTIZATION_SAMPLES} calibration images...")
calib = []
for class_name in ("Negative", "Positive"):
    class_dir = DATASET_PATH / class_name
    files = sorted(os.listdir(class_dir))[:QUANTIZATION_SAMPLES // 2]
    for name in files:
        try:
            img = tf.keras.preprocessing.image.load_img(
                str(class_dir / name), target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb")
            calib.append(tf.keras.preprocessing.image.img_to_array(img) / 255.0)
        except Exception:
            pass
calib = np.array(calib, dtype=np.float32)
print(f"  {len(calib)} calibration images")


def rep_gen():
    for i in range(len(calib)):
        yield [calib[i:i + 1]]


# ---------------------------------------------------------------- INT8
print("\n[3/4] INT8 TFLite conversion (full integer)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

try:
    int8_bytes = converter.convert()
    full_int8 = True
except Exception as e:
    print(f"  Full INT8 failed ({e}); falling back to dynamic-range quantization")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_bytes = converter.convert()
    full_int8 = False

int8_path = SCRIPT_DIR / "prev_microcrack_int8.tflite"
int8_path.write_bytes(int8_bytes)
print(f"  {int8_path.name}: {len(int8_bytes):,} bytes ({len(int8_bytes)/1024:.1f} KB)")

interp = tf.lite.Interpreter(str(int8_path))
interp.allocate_tensors()
in_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
in_shape, in_dtype = list(in_det["shape"]), in_det["dtype"].__name__
out_shape, out_dtype = list(out_det["shape"]), out_det["dtype"].__name__
print(f"  input : shape={in_shape} dtype={in_dtype} quant={in_det['quantization']}")
print(f"  output: shape={out_shape} dtype={out_dtype} quant={out_det['quantization']}")

# ---------------------------------------------------------------- Headers
print("\n[4/4] Generating C headers for the ESP32-CAM sketch...")
write_model_header(int8_bytes, ESP32CAM_SRC / "model_data.h", model_name="microcrack_model")

positive_dir = DATASET_PATH / "Positive"
sample = None
for name in sorted(os.listdir(positive_dir)):
    if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}:
        sample = positive_dir / name
        break
if sample is None:
    raise FileNotFoundError(f"No sample image found in {positive_dir}")
write_image_header(sample, ESP32CAM_SRC / "test_image.h")

summary = {
    "source_model": str(model_path),
    "float32": {"file": str(float_path), "size_bytes": len(float_bytes)},
    "int8": {"file": str(int8_path), "size_bytes": len(int8_bytes),
             "full_integer_quantization": full_int8},
    "compression_ratio": round(len(float_bytes) / max(len(int8_bytes), 1), 2),
    "input": {"shape": in_shape, "dtype": in_dtype,
              "scale": float(in_det["quantization"][0]),
              "zero_point": int(in_det["quantization"][1])},
    "output": {"shape": out_shape, "dtype": out_dtype,
               "scale": float(out_det["quantization"][0]),
               "zero_point": int(out_det["quantization"][1])},
    "headers": {
        "model_data.h": str(ESP32CAM_SRC / "model_data.h"),
        "test_image.h": str(ESP32CAM_SRC / "test_image.h"),
        "test_image_source": str(sample),
    },
    "calibration_samples": len(calib),
}
with open(RESULTS_DIR / "prev_tflite_conversion_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 60)
print("DEPLOY COMPLETE")
print("=" * 60)
print(f"float32: {len(float_bytes)/1024:.1f} KB   int8: {len(int8_bytes)/1024:.1f} KB   "
      f"ratio: {summary['compression_ratio']}x   full_int8={full_int8}")
print(f"Model header : {ESP32CAM_SRC / 'model_data.h'}")
print(f"Image header : {ESP32CAM_SRC / 'test_image.h'}")
print(f"Summary      : {RESULTS_DIR / 'prev_tflite_conversion_summary.json'}")
