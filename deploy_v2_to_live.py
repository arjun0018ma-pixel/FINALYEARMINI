"""
Deploy the IMPROVED models to the LIVE ESP32-CAM project (Track 5)
==================================================================
Generates the C headers used by embedded_ml/esp32cam_live/src/main.cpp:

  classifier : embedded_ml/prev_microcrack_cnn_v2.keras   (96x96, sigmoid)
  segmentation: segmentation/densecrack_unet_tiny.keras   (64x64)

Headers (new files):
  embedded_ml/esp32cam_live/src/live_model_data.h        (INT8 classifier)
  embedded_ml/esp32cam_live/src/live_test_image.h        (fallback frame)
  embedded_ml/esp32cam_live/src/live_seg_model_data.h    (INT8 tiny U-Net)
  embedded_ml/esp32cam_live/src/live_seg_test_image.h    (fallback frame)

Summary: results/live/live_deploy_summary.json
"""

import json
import os
import numpy as np
import tensorflow as tf
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
DENSECRACK = (DATASET_PATH / "DenseCrack A Segmentation Dataset of Dense Micro-C" /
              "DenseCrack" / "DenseCrack")
LIVE_SRC = SCRIPT_DIR / "esp32cam_live" / "src"
RESULTS_DIR = PROJECT_ROOT / "results" / "live"

CLS_MODEL = SCRIPT_DIR / "prev_microcrack_cnn_v2.keras"
SEG_MODEL = PROJECT_ROOT / "segmentation" / "densecrack_unet_tiny.keras"
CLS_SIZE = 96
SEG_SIZE = 64

LIVE_SRC.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 60)
print("DEPLOY IMPROVED MODELS -> LIVE ESP32-CAM")
print("=" * 60)


def write_model_header(tflite_bytes, out_path, symbol):
    guard = f"_{out_path.stem.upper()}_H_"
    lines = [f"#ifndef {guard}", f"#define {guard}", "",
             f"// TFLite model: {symbol} ({len(tflite_bytes):,} bytes)",
             f"// Size: {len(tflite_bytes):,} bytes", "",
             f"alignas(8) const unsigned char {symbol}[] = {{"]
    for i in range(0, len(tflite_bytes), 16):
        chunk = tflite_bytes[i:i + 16]
        s = "    " + ", ".join(f"0x{b:02x}" for b in chunk)
        if i + 16 < len(tflite_bytes):
            s += ","
        lines.append(s)
    lines += ["};", "", f"const unsigned int {symbol}_len = {len(tflite_bytes)};", "",
              f"#endif  // {guard}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {out_path.name} ({out_path.stat().st_size/1024:.1f} KB)")


def write_image_header(image_path, out_path, symbol, size):
    img = tf.keras.preprocessing.image.load_img(str(image_path), target_size=(size, size), color_mode="rgb")
    arr = (tf.keras.preprocessing.image.img_to_array(img) / 255.0).flatten().astype(np.float32)
    guard = f"_{out_path.stem.upper()}_H_"
    lines = [f"#ifndef {guard}", f"#define {guard}", "",
             f"// Fallback frame: {Path(image_path).name}  ({size}x{size}x3 float in [0,1])", "",
             f"const float {symbol}[] = {{"]
    for i in range(0, len(arr), 8):
        s = "    " + ", ".join(f"{v:.6f}f" for v in arr[i:i + 8])
        if i + 8 < len(arr):
            s += ","
        lines.append(s)
    lines += ["};", "", f"const unsigned int {symbol}_len = {len(arr)};", "", f"#endif  // {guard}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {out_path.name} ({out_path.stat().st_size/1024:.1f} KB)")


def to_int8(model, size, gen):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def info(path):
    interp = tf.lite.Interpreter(model_content=path if isinstance(path, bytes) else str(path))
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    return {
        "input": {"shape": [int(x) for x in i["shape"]], "dtype": i["dtype"].__name__,
                  "scale": float(i["quantization"][0]), "zero_point": int(i["quantization"][1])},
        "output": {"shape": [int(x) for x in o["shape"]], "dtype": o["dtype"].__name__,
                   "scale": float(o["quantization"][0]), "zero_point": int(o["quantization"][1])},
    }


summary = {}

# ---------------- classifier ----------------
if CLS_MODEL.exists():
    print(f"[classifier] {CLS_MODEL.name}")
    model = tf.keras.models.load_model(str(CLS_MODEL))
    calib = []
    for cls in ("Negative", "Positive"):
        for n in sorted(os.listdir(DATASET_PATH / cls))[:80]:
            if os.path.splitext(n)[1].lower() in {".jpg", ".jpeg", ".png"}:
                im = tf.keras.preprocessing.image.load_img(str(DATASET_PATH / cls / n),
                                                            target_size=(CLS_SIZE, CLS_SIZE), color_mode="rgb")
                calib.append(tf.keras.preprocessing.image.img_to_array(im) / 255.0)
    calib = np.array(calib, dtype=np.float32)

    def cls_gen():
        for x in calib:
            yield [x[None, ...]]
    cls_bytes = to_int8(model, CLS_SIZE, cls_gen)
    write_model_header(cls_bytes, LIVE_SRC / "live_model_data.h", "live_model_data")
    sample = next(DATASET_PATH.joinpath("Positive").iterdir())
    for n in sorted(os.listdir(DATASET_PATH / "Positive")):
        if os.path.splitext(n)[1].lower() in {".jpg", ".jpeg", ".png"}:
            sample = DATASET_PATH / "Positive" / n
            break
    write_image_header(sample, LIVE_SRC / "live_test_image.h", "live_test_image", CLS_SIZE)
    summary["classifier"] = {"model": str(CLS_MODEL), "size_bytes": len(cls_bytes),
                             "calibration_samples": len(calib), **info(cls_bytes)}

# ---------------- segmentation ----------------
if SEG_MODEL.exists():
    print(f"[segmentation] {SEG_MODEL.name}")
    model = tf.keras.models.load_model(str(SEG_MODEL), compile=False)
    files = sorted(p for p in (DENSECRACK / "train_image").iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})[:60]

    def seg_gen():
        for p in files:
            img = tf.image.decode_png(tf.io.read_file(str(p)), channels=3)
            img = tf.image.resize(img, (SEG_SIZE, SEG_SIZE), method='bilinear')
            yield [tf.expand_dims(tf.cast(img, tf.float32) / 255.0, 0)]
    seg_bytes = to_int8(model, SEG_SIZE, seg_gen)
    write_model_header(seg_bytes, LIVE_SRC / "live_seg_model_data.h", "live_seg_model_data")
    sample = sorted(p for p in (DENSECRACK / "test_image").iterdir()
                    if p.suffix.lower() in {".png", ".jpg"})[0]
    write_image_header(sample, LIVE_SRC / "live_seg_test_image.h", "live_seg_test_image", SEG_SIZE)
    summary["segmentation"] = {"model": str(SEG_MODEL), "size_bytes": len(seg_bytes),
                              "calibration_samples": len(files), **info(seg_bytes)}

with open(RESULTS_DIR / "live_deploy_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("=" * 60)
print("LIVE DEPLOY COMPLETE")
print("=" * 60)
for k, v in summary.items():
    print(f"{k}: {v['size_bytes']:,} bytes")
