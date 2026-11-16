"""
tinyML optimisation study (Track 3)
===================================
Measures the accuracy / size / latency trade-off for the improved crack
classifier (embedded_ml/prev_microcrack_cnn_v2.keras) across:

  1. float32                    - unquantised baseline (4 bytes/weight)
  2. int8                       - full-integer quantisation
  3. pruned50 + fine-tune + int8 - magnitude pruning retrained to recover accuracy

For each variant we report test accuracy / F1 (held-out split), TFLite size (KB),
compression vs float32 and mean host-CPU inference latency (ms).

New outputs:
  results/optimization/tradeoff.json
  results/optimization/tradeoff.md
  results/optimization/tradeoff.png
"""

import json
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
MODEL_PATH = Path(__file__).parent / "prev_microcrack_cnn_v2.keras"
RESULTS_DIR = PROJECT_ROOT / "results" / "optimization"

IMG_SIZE = 96
SEED = 42
PER_CLASS_LIMIT = 4000
PRUNE_SPARSITY = 0.5
FINETUNE_IMAGES = 4000
FINETUNE_EPOCHS = 30
FINETUNE_LR = 1e-4
BENCH_RUNS = 100
BENCH_WARMUP = 10

np.random.seed(SEED)
tf.random.set_seed(SEED)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 60)
print("tinyML OPTIMISATION STUDY")
print("=" * 60)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"{MODEL_PATH} not found (run train_cnn_prevdataset_v2.py first)")

# ---------------- reproduce the training split (same seed) ----------------
VALID = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
neg = [str(DATASET_PATH / "Negative" / n) for n in sorted(os.listdir(DATASET_PATH / "Negative"))
       if os.path.splitext(n)[1].lower() in VALID][:PER_CLASS_LIMIT]
pos = [str(DATASET_PATH / "Positive" / n) for n in sorted(os.listdir(DATASET_PATH / "Positive"))
       if os.path.splitext(n)[1].lower() in VALID][:PER_CLASS_LIMIT]
files = neg + pos
labels = [0] * len(neg) + [1] * len(pos)
perm = np.random.permutation(len(files))
files = [files[i] for i in perm]
labels = [labels[i] for i in perm]
total = len(files)
train_size = int(total * 0.8)
val_size = int(total * 0.1)
train_files, train_labels = files[:train_size], labels[:train_size]
test_files = files[train_size + val_size:]
test_labels = np.array(labels[train_size + val_size:])
print(f"Test images: {len(test_files)}")

X_test = np.zeros((len(test_files), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
for i, fp in enumerate(test_files):
    img = keras.preprocessing.image.load_img(fp, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb')
    X_test[i] = keras.preprocessing.image.img_to_array(img) / 255.0
print(f"Loaded test tensor: {X_test.shape}")


def scores(preds):
    from sklearn.metrics import accuracy_score, f1_score
    p = (preds >= 0.5).astype(int)
    return float(accuracy_score(test_labels, p)), float(f1_score(test_labels, p, zero_division=0))


def magnitude_prune(model, sparsity):
    """Zero the smallest-magnitude weights (per tensor) and return the mask so
    the same sparsity pattern can be enforced throughout fine-tuning."""
    masks = []
    pruned = tot = 0
    for w in model.weights:
        if len(w.shape) >= 2 and 'kernel' in w.name:
            arr = w.numpy()
            thr = np.percentile(np.abs(arr), sparsity * 100)
            mask = np.abs(arr) < thr
            arr[mask] = 0.0
            w.assign(arr)
            masks.append((w, mask))
            pruned += int(mask.sum()); tot += arr.size
    print(f"  effective sparsity: {pruned / max(tot, 1) * 100:.1f}% ({pruned:,}/{tot:,})")
    return masks


def apply_masks(masks):
    """Keep pruned weights exactly zero (fine-tuning must not resurrect them)."""
    for w, mask in masks:
        arr = w.numpy()
        arr[mask] = 0.0
        w.assign(arr)


class MaskCallback(keras.callbacks.Callback):
    def __init__(self, masks):
        super().__init__()
        self.masks = masks

    def on_train_batch_end(self, batch, logs=None):
        apply_masks(self.masks)


def convert(model, kind, rep=None):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if kind == "float32":
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    elif kind == "float16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    else:  # int8
        conv.optimizations = [tf.lite.Optimize.DEFAULT]

        def _rep():
            for x in rep:
                yield [x[None, ...].astype(np.float32)]
        conv.representative_dataset = _rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def benchmark(tflite_bytes):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    is_int8 = inp["dtype"] == np.int8
    x = X_test[0]
    if is_int8:
        scale, zp = inp["quantization"]
        q = np.clip(np.round(x / scale) + zp, -128, 127).astype(np.int8)[None, ...]
    else:
        q = x[None, ...].astype(np.float32)
    for _ in range(BENCH_WARMUP):
        interp.set_tensor(inp["index"], q); interp.invoke()
    t0 = time.perf_counter()
    for _ in range(BENCH_RUNS):
        interp.set_tensor(inp["index"], q); interp.invoke()
    return (time.perf_counter() - t0) / BENCH_RUNS * 1000.0, inp["dtype"].__name__


def tflite_eval(tflite_bytes):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    is_int8 = inp["dtype"] == np.int8
    preds = np.zeros(len(X_test))
    for i in range(len(X_test)):
        x = X_test[i]
        if is_int8:
            scale, zp = inp["quantization"]
            q = np.clip(np.round(x / scale) + zp, -128, 127).astype(np.int8)[None, ...]
        else:
            q = x[None, ...].astype(np.float32)
        interp.set_tensor(inp["index"], q); interp.invoke()
        y = interp.get_tensor(out["index"]).ravel()
        if out["dtype"] == np.int8:
            oscale, ozp = out["quantization"]
            y = (y.astype(np.float32) - ozp) * oscale
        preds[i] = float(y[0])
    return scores(preds)


variants = []
model = keras.models.load_model(str(MODEL_PATH))

# --- float32 ---
f32 = convert(model, "float32")
lat, dt = benchmark(f32)
a, f = tflite_eval(f32)
variants.append(dict(name="float32", accuracy=a, f1=f, size_kb=round(len(f32)/1024, 1),
                     latency_ms=round(lat, 3), dtype=dt))
print(f"float32      acc={a:.4f} size={len(f32)/1024:.1f}KB lat={lat:.3f}ms")

# --- int8 ---
calib = X_test[: min(120, len(X_test))]
i8 = convert(model, "int8", rep=calib)
lat8, dt8 = benchmark(i8)
a8, f8 = tflite_eval(i8)
variants.append(dict(name="int8", accuracy=a8, f1=f8, size_kb=round(len(i8)/1024, 1),
                     latency_ms=round(lat8, 3), dtype=dt8))
print(f"int8         acc={a8:.4f} size={len(i8)/1024:.1f}KB lat={lat8:.3f}ms")

# --- float16 (2x smaller, near-lossless; keeps float compute) ---
f16 = convert(model, "float16")
lat16, dt16 = benchmark(f16)
a16, f16s = tflite_eval(f16)
variants.append(dict(name="float16", accuracy=a16, f1=f16s, size_kb=round(len(f16)/1024, 1),
                     latency_ms=round(lat16, 3), dtype=dt16))
print(f"float16      acc={a16:.4f} size={len(f16)/1024:.1f}KB lat={lat16:.3f}ms")

# --- pruned + fine-tune + int8 ---
m = keras.models.load_model(str(MODEL_PATH))
for layer in m.layers:          # unfreeze so the whole net can recover after pruning
    layer.trainable = True
masks = magnitude_prune(m, PRUNE_SPARSITY)


def ft_load(path, label):
    img = tf.image.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32) / 255.0, tf.cast(label, tf.float32)


ft = tf.data.Dataset.from_tensor_slices(
    (np.array(train_files[:FINETUNE_IMAGES]), np.array(train_labels[:FINETUNE_IMAGES])))
ft = ft.map(ft_load, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)
schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=FINETUNE_LR, decay_steps=FINETUNE_EPOCHS * max(len(ft), 1))
m.compile(optimizer=keras.optimizers.Adam(schedule), loss=keras.losses.BinaryCrossentropy(),
          metrics=[keras.metrics.BinaryAccuracy(name='acc')])
m.fit(ft, epochs=FINETUNE_EPOCHS, verbose=1, callbacks=[MaskCallback(masks)])

# enforce the fixed mask one final time before exporting
apply_masks(masks)

p50 = convert(m, "int8", rep=calib)
latp, dtp = benchmark(p50)
ap, fp = tflite_eval(p50)
variants.append(dict(name=f"pruned{int(PRUNE_SPARSITY*100)}+int8", accuracy=ap, f1=fp,
                     size_kb=round(len(p50)/1024, 1), latency_ms=round(latp, 3), dtype=dtp,
                     finetuned_epochs=FINETUNE_EPOCHS))
print(f"pruned+int8  acc={ap:.4f} size={len(p50)/1024:.1f}KB lat={latp:.3f}ms")

# --- sparsity sweep (post-training magnitude pruning, NO retraining) ---
sweep = []
for sp in (0.0, 0.25, 0.5, 0.75):
    ms = keras.models.load_model(str(MODEL_PATH))
    if sp > 0:
        magnitude_prune(ms, sp)
    a_s, f_s = scores(ms.predict(X_test, verbose=0).ravel())
    sweep.append({"sparsity": sp, "accuracy": round(a_s, 4), "f1": round(f_s, 4)})
    print(f"sparsity {sp:.0%}: acc={a_s:.4f} f1={f_s:.4f}")

# ---------------- report ----------------
base_size, base_lat = variants[0]["size_kb"], variants[0]["latency_ms"]
for v in variants:
    v["compression_vs_float32"] = round(base_size / v["size_kb"], 2) if v["size_kb"] else None
    v["latency_speedup_vs_float32"] = round(base_lat / v["latency_ms"], 2) if v["latency_ms"] else None

report = {"model": str(MODEL_PATH), "benchmark_runs": BENCH_RUNS,
          "device": "host CPU (Windows)", "variants": variants,
          "sparsity_sweep": sweep}
with open(RESULTS_DIR / "tradeoff.json", "w") as fj:
    json.dump(report, fj, indent=2)

md = ["# tinyML optimisation trade-off", "",
      f"Model: `{MODEL_PATH.name}`  |  latency: host CPU, {BENCH_RUNS} runs", "",
      "| Variant | Accuracy | F1 | Size (KB) | Compression | Latency (ms) | Speedup |",
      "|---|---|---|---|---|---|---|"]
for v in variants:
    md.append(f"| {v['name']} | {v['accuracy']:.4f} | {v['f1']:.4f} | {v['size_kb']} | "
              f"{v['compression_vs_float32']}x | {v['latency_ms']} | {v['latency_speedup_vs_float32']}x |")
md += ["", "## Sparsity sweep (post-training magnitude pruning, no retraining)", "",
       "| Sparsity | Accuracy | F1 |", "|---|---|---|"]
for s in sweep:
    md.append(f"| {s['sparsity']:.0%} | {s['accuracy']:.4f} | {s['f1']:.4f} |")
(RESULTS_DIR / "tradeoff.md").write_text("\n".join(md), encoding="utf-8")
print(f"Wrote {RESULTS_DIR / 'tradeoff.md'}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [v["name"] for v in variants]
    sizes = [v["size_kb"] for v in variants]
    lats = [v["latency_ms"] for v in variants]
    accs = [v["accuracy"] for v in variants]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax1.bar(x - 0.2, sizes, 0.4, label="Size (KB)", color="#4c72b0")
    ax2 = ax1.twinx()
    ax2.plot(x, lats, "o-", color="#dd8452", label="Latency (ms)")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15)
    ax1.set_ylabel("TFLite size (KB)"); ax2.set_ylabel("Latency (ms)")
    ax1.set_title("Model size / latency trade-off")
    for i, a in enumerate(accs):
        ax1.text(i, sizes[i], f"acc\n{a:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tradeoff.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {RESULTS_DIR / 'tradeoff.png'}")
except ImportError:
    print("matplotlib not available")

print("=" * 60)
print("OPTIMISATION STUDY COMPLETE")
print("=" * 60)
