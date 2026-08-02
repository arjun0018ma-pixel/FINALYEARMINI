import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import pandas as pd
import time

# ===============================
# Configuration
# ===============================

MODEL_PATH = "vit_microcrack.pth"
CLASS_NAMES = ["Negative", "Positive"]  # must match training dataset.classes order
IMAGE_SIZE = 224

ACCENT_FAIL = "#FF4D4D"     # defect detected
ACCENT_PASS = "#3ECF8E"     # no defect
BG_PRIMARY = "#0A0E13"
BG_PANEL = "#12181F"
BORDER = "#232C36"
TEXT_MUTED = "#7C8894"
TEXT_MAIN = "#E4E9ED"

st.set_page_config(
    page_title="Microcrack Inspection System",
    page_icon="◈",
    layout="wide"
)

# ===============================
# Global styling
# ===============================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {BG_PRIMARY};
    background-image:
        linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
    background-size: 32px 32px;
}}

.system-header {{
    border-bottom: 1px solid {BORDER};
    padding-bottom: 18px;
    margin-bottom: 28px;
}}
.eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 3px;
    color: {TEXT_MUTED};
    text-transform: uppercase;
}}
.system-title {{
    font-size: 32px;
    font-weight: 700;
    color: {TEXT_MAIN};
    margin-top: 4px;
    letter-spacing: -0.5px;
}}

.panel {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 20px 24px;
}}
.panel-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    margin-bottom: 10px;
}}

.verdict {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 22px;
    letter-spacing: 1px;
    padding: 14px 18px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.verdict-fail {{
    background-color: rgba(255,77,77,0.08);
    border: 1px solid rgba(255,77,77,0.4);
    color: {ACCENT_FAIL};
}}
.verdict-pass {{
    background-color: rgba(62,207,142,0.08);
    border: 1px solid rgba(62,207,142,0.4);
    color: {ACCENT_PASS};
}}

.readout-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: {TEXT_MUTED};
}}
.readout-value span {{
    color: {TEXT_MAIN};
    font-weight: 600;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

[data-testid="stMetric"] {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 14px 18px;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 11px;
}}

[data-testid="stFileUploaderDropzone"] {{
    background-color: {BG_PANEL};
    border: 1px dashed {BORDER};
}}

.stTabs [data-baseweb="tab"] {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    letter-spacing: 1px;
    text-transform: uppercase;
}}

hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)

# ===============================
# Model loading
# ===============================

@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, processor


@st.cache_resource
def get_transform(_processor):
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=_processor.image_mean, std=_processor.image_std)
    ])


def predict(image, model, transform):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(pixel_values=tensor).logits
        probs = F.softmax(outputs, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    return CLASS_NAMES[pred_idx], probs.tolist()


def gauge_html(confidence_pct, is_fail):
    # Pure CSS conic-gradient dial -- no external JS component to fetch,
    # so this can't fail the way the Plotly component asset load did.
    color = ACCENT_FAIL if is_fail else ACCENT_PASS
    angle = confidence_pct * 3.6  # percent -> degrees
    return f"""
    <div style="display:flex; justify-content:center; padding:10px 0 4px;">
        <div style="
            position:relative;
            width:180px; height:180px;
            border-radius:50%;
            background: conic-gradient({color} {angle}deg, {BORDER} {angle}deg 360deg);
            display:flex; align-items:center; justify-content:center;
        ">
            <div style="
                width:138px; height:138px;
                border-radius:50%;
                background:{BG_PANEL};
                display:flex; flex-direction:column;
                align-items:center; justify-content:center;
            ">
                <div style="font-family:'JetBrains Mono'; font-size:34px; font-weight:700; color:{TEXT_MAIN};">
                    {confidence_pct:.1f}%
                </div>
                <div style="font-family:'JetBrains Mono'; font-size:10px; letter-spacing:2px; color:{TEXT_MUTED}; margin-top:2px;">
                    CONFIDENCE
                </div>
            </div>
        </div>
    </div>
    """


# ===============================
# Sidebar
# ===============================

with st.sidebar:
    st.markdown('<div class="eyebrow">SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown('<div class="system-title" style="font-size:20px;">Inspection Node</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="panel-label">MODEL</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="readout-value">Backbone <span>ViT-B/16</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="readout-value">Input <span>{IMAGE_SIZE}x{IMAGE_SIZE}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="readout-value">Classes <span>{" / ".join(CLASS_NAMES)}</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel-label">REVIEW THRESHOLD</div>', unsafe_allow_html=True)
    confidence_threshold = st.slider(
        "Flag as uncertain below", min_value=50, max_value=99, value=70, label_visibility="collapsed"
    )
    st.caption(f"Predictions under {confidence_threshold}% confidence are flagged for manual review.")

# ===============================
# Load model
# ===============================

try:
    with st.spinner("Initializing model..."):
        model, processor = load_model()
        transform = get_transform(processor)
except FileNotFoundError:
    st.error(f"Checkpoint `{MODEL_PATH}` not found. Place it next to this script.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ===============================
# Header
# ===============================

st.markdown(f"""
<div class="system-header">
    <div class="eyebrow">VISION TRANSFORMER · SURFACE DEFECT ANALYSIS</div>
    <div class="system-title">Microcrack Inspection Dashboard</div>
</div>
""", unsafe_allow_html=True)

tab_single, tab_batch = st.tabs(["Single Scan", "Batch Log"])

# ---------------- Single scan ----------------
with tab_single:
    uploaded_file = st.file_uploader(
        "Upload a surface image", type=["jpg", "jpeg", "png", "bmp"], key="single",
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.markdown(f"""
        <div class="panel" style="text-align:center; padding:60px 20px; color:{TEXT_MUTED};">
            <div class="eyebrow">NO SCAN LOADED</div>
            <div style="margin-top:8px;">Upload a surface image above to begin inspection.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="panel-label">SOURCE IMAGE</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        start = time.time()
        label, probs = predict(image, model, transform)
        elapsed = (time.time() - start) * 1000
        confidence = max(probs) * 100
        is_fail = label == "Positive"

        with col2:
            st.markdown('<div class="panel-label">VERDICT</div>', unsafe_allow_html=True)
            if is_fail:
                st.markdown(f'<div class="verdict verdict-fail">⬤ DEFECT DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict verdict-pass">⬤ NO DEFECT FOUND</div>', unsafe_allow_html=True)

            st.markdown(gauge_html(confidence, is_fail), unsafe_allow_html=True)

            if confidence < confidence_threshold:
                st.warning("Below review threshold — recommend manual inspection.")

            st.markdown(
                f'<div class="readout-value">Inference time <span>{elapsed:.1f} ms</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-label">CLASS PROBABILITIES</div>', unsafe_allow_html=True)
        prob_cols = st.columns(len(CLASS_NAMES))
        for col, name, p in zip(prob_cols, CLASS_NAMES, probs):
            with col:
                st.markdown(f"""
                <div class="panel">
                    <div class="readout-value">{name}</div>
                    <div style="font-family:'JetBrains Mono'; font-size:24px; font-weight:700; color:{TEXT_MAIN}; margin-top:6px;">
                        {p*100:.2f}%
                    </div>
                    <div style="background:{BORDER}; height:4px; border-radius:2px; margin-top:10px;">
                        <div style="background:{ACCENT_FAIL if name=='Positive' else ACCENT_PASS}; width:{p*100}%; height:4px; border-radius:2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ---------------- Batch log ----------------
with tab_batch:
    uploaded_files = st.file_uploader(
        "Upload multiple surface images", type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True, key="batch", label_visibility="collapsed"
    )

    if not uploaded_files:
        st.markdown(f"""
        <div class="panel" style="text-align:center; padding:60px 20px; color:{TEXT_MUTED};">
            <div class="eyebrow">LOG EMPTY</div>
            <div style="margin-top:8px;">Upload a batch of images above to generate an inspection log.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        results = []
        progress = st.progress(0, text="Running batch inspection...")

        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            label, probs = predict(image, model, transform)
            results.append({
                "Filename": file.name,
                "Verdict": "DEFECT" if label == "Positive" else "CLEAR",
                "Confidence (%)": round(max(probs) * 100, 2)
            })
            progress.progress((i + 1) / len(uploaded_files), text=f"Scanning {i+1}/{len(uploaded_files)}")

        progress.empty()
        results_df = pd.DataFrame(results)

        c1, c2, c3 = st.columns(3)
        c1.metric("TOTAL SCANNED", len(results_df))
        c2.metric("DEFECTS FOUND", int((results_df["Verdict"] == "DEFECT").sum()))
        c3.metric("CLEAR", int((results_df["Verdict"] == "CLEAR").sum()))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-label">INSPECTION LOG</div>', unsafe_allow_html=True)

        def highlight_defect(row):
            color = "rgba(255,77,77,0.08)" if row["Verdict"] == "DEFECT" else "rgba(62,207,142,0.05)"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            results_df.style.apply(highlight_defect, axis=1),
            use_container_width=True,
            hide_index=True
        )

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Export log as CSV", data=csv, file_name="inspection_log.csv", mime="text/csv")