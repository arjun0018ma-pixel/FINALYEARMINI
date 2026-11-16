import streamlit as st
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import pandas as pd
import time
import cv2
import numpy as np
import requests
from io import BytesIO
from crack_measure import calculate_crack_length

# ===============================
# Configuration
# ===============================

MODEL_PATH = Path(__file__).resolve().parent / "vit_microcrack.pth"
CLASS_NAMES = ["Negative", "Positive"]  # must match training dataset.classes order
IMAGE_SIZE = 224

ACCENT_FAIL = "#C62828"     # defect detected
ACCENT_PASS = "#1B7F4B"     # no defect
BG_PRIMARY = "#FFFFFF"
BG_PANEL = "#F5F7FA"
BORDER = "#CBD2D9"
TEXT_MUTED = "#52606D"
TEXT_MAIN = "#102A43"

st.set_page_config(
    page_title="Microcrack Inspection System",
    page_icon="◈",
    layout="wide"
)

st.session_state.setdefault("telemetry_log", [])

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
        linear-gradient(rgba(0,0,0,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,0,0,0.04) 1px, transparent 1px);
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
    background-color: rgba(198,40,40,0.08);
    border: 1px solid rgba(198,40,40,0.5);
    color: {ACCENT_FAIL};
}}
.verdict-pass {{
    background-color: rgba(27,127,75,0.08);
    border: 1px solid rgba(27,127,75,0.5);
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

[data-testid="stCameraInput"] {{
    background-color: {BG_PANEL};
    border: 1px dashed {BORDER};
    border-radius: 6px;
    padding: 12px;
}}

.stTabs [data-baseweb="tab"] {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    letter-spacing: 1px;
    text-transform: uppercase;
}}

hr {{ border-color: {BORDER}; }}

.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3 {{
    color: {TEXT_MAIN};
}}
[data-testid="stHeader"] {{ background-color: {BG_PRIMARY}; }}
[data-testid="stHeader"] * {{ color: {TEXT_MAIN}; }}
[data-testid="stSidebar"] {{ background-color: {BG_PANEL}; border-right: 1px solid {BORDER}; }}
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] {{ color: {TEXT_MUTED}; }}
[data-testid="stMetricValue"] {{ color: {TEXT_MAIN}; }}
[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; }}
.stTabs [data-baseweb="tab"] {{ color: {TEXT_MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {ACCENT_FAIL}; }}
[data-baseweb="input"], [data-baseweb="base-input"], .stTextInput input {{
    background-color: {BG_PANEL}; color: {TEXT_MAIN};
}}
.stButton > button:not([kind="primary"]) {{
    background-color: {BG_PRIMARY}; color: {TEXT_MAIN}; border: 1px solid {BORDER};
}}

@media print {{
    * {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
}}
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
    color = ACCENT_FAIL if is_fail else ACCENT_PASS
    angle = confidence_pct * 3.6
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


def render_inspection_results(image, source_label="SOURCE IMAGE"):
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(f'<div class="panel-label">{source_label}</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    start = time.time()
    label, probs = predict(image, model, transform)
    elapsed = time.time() - start
    confidence = max(probs) * 100
    is_fail = label == "Positive"

    with col2:
        st.markdown('<div class="panel-label">VERDICT</div>', unsafe_allow_html=True)
        if is_fail:
            st.markdown('<div class="verdict verdict-fail">⬤ DEFECT DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict verdict-pass">⬤ NO DEFECT FOUND</div>', unsafe_allow_html=True)

        st.markdown(gauge_html(confidence, is_fail), unsafe_allow_html=True)

        if confidence < confidence_threshold:
            st.warning("Below review threshold — recommend manual inspection.")

        st.markdown(
            f'<div class="readout-value">Inference time <span>{elapsed:.2f} sec</span></div>',
            unsafe_allow_html=True
        )

        if is_fail:
            opencv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            length, result = calculate_crack_length(opencv_img)

            st.subheader("Crack Analysis")
            st.metric("Estimated Crack Length", f"{length:.2f} pixels")
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detected Crack")

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

    return label


# ===============================
# Sidebar
# ===============================

with st.sidebar:
    st.markdown('<div class="eyebrow">SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown('<div class="system-title" style="font-size:20px;">Inspection Node</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="panel-label">MODEL</div>', unsafe_allow_html=True)
    st.markdown('<div class="readout-value">Backbone <span>ViT-B/16</span></div>', unsafe_allow_html=True)
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

st.markdown("""
<div class="system-header">
    <div class="eyebrow">VISION TRANSFORMER · SURFACE DEFECT ANALYSIS</div>
    <div class="system-title">Microcrack Inspection Dashboard</div>
</div>
""", unsafe_allow_html=True)

tab_single, tab_esp32, tab_cam, tab_batch = st.tabs([
    "Single Scan", "ESP32 Hardware Node", "Mobile Camera", "Batch Log"
])

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
        render_inspection_results(image, source_label="SOURCE IMAGE")

# ---------------- ESP32 Hardware Node ----------------
# ---------------- ESP32 Hardware Node ----------------

with tab_esp32:
    st.markdown('<div class="panel-label">ESP32-CAM & MPU6050 / DHT11 TELEMETRY NODE</div>', unsafe_allow_html=True)
    c_ip1, c_ip2 = st.columns(2)
    with c_ip1:
        cam_ip = st.text_input("ESP32-CAM IP Address", value="192.168.1.45")
    with c_ip2:
        devkit_ip = st.text_input("ESP32 DevKit V1 IP (MPU6050 + DHT11 + Buzzer)", value="192.168.1.49")

    if devkit_ip.strip():
        t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns([1, 1, 1, 1, 1])
        try:
            s_resp = requests.get(f"http://{devkit_ip.strip()}/sensors", timeout=2)
            s_resp.raise_for_status()
            sensor_data = s_resp.json()
            t_col1.metric("DHT11 TEMP", f"{sensor_data['temp_c']} °C" if sensor_data['temp_c'] is not None else "ERR")
            t_col2.metric("THERMAL DELTA (ΔT)", f"{sensor_data['delta_t']:+.2f} °C")
            t_col3.metric("PEAK VIBRATION", f"{sensor_data['peak_g']:.2f} g")
            t_col4.metric("FATIGUE SHOCKS", f"{sensor_data['vib_events']}")
            with t_col5:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Reset Baseline"):
                    requests.get(f"http://{devkit_ip.strip()}/reset", timeout=2)
                    st.toast("Baseline temperature, peak g & shock counter reset")
        except Exception:
            st.caption("DevKit telemetry offline — enter DevKit IP above to enable MPU6050 + DHT11 correlation.")

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, col_tog = st.columns([1, 2])
    with col_btn:
        capture_clicked = st.button("Capture & Inspect (Telemetry + ViT)", type="primary")
    with col_tog:
        live_preview = st.toggle("Live camera viewfinder (1 fps)", value=False, disabled=capture_clicked)

    if live_preview and not capture_clicked and cam_ip.strip():
        @st.fragment(run_every=1.5)
        def live_viewfinder():
            try:
                r = requests.get(f"http://{cam_ip.strip()}/capture", timeout=5)
                r.raise_for_status()
                frame = Image.open(BytesIO(r.content)).convert("RGB")
                st.image(frame, caption=f"Live Viewfinder ({time.strftime('%H:%M:%S')})", use_container_width=True)
            except Exception as preview_err:
                st.warning(f"Viewfinder waiting on camera: {preview_err}")

        live_viewfinder()

    if capture_clicked:
        latest_sensors = {"temp_c": None, "delta_t": 0.0, "peak_g": 0.0, "vib_events": 0, "vibration": False}
        if devkit_ip.strip():
            try:
                s_res = requests.get(f"http://{devkit_ip.strip()}/sensors", timeout=2).json()
                if s_res.get("vibration"):
                    st.info("MPU6050 detected active vibration — waiting 0.5s for surface to stabilize...")
                    time.sleep(0.5)
                    s_res = requests.get(f"http://{devkit_ip.strip()}/sensors", timeout=2).json()
                latest_sensors = s_res
            except Exception:
                pass

        try:
            with st.spinner("Capturing frame from ESP32-CAM..."):
                resp = requests.get(f"http://{cam_ip.strip()}/capture", timeout=12)
                resp.raise_for_status()
                st.session_state["esp_image"] = Image.open(BytesIO(resp.content)).convert("RGB")
                st.session_state["esp_sensors"] = latest_sensors
                st.session_state["esp_alert_pending"] = True
                st.session_state["esp_error"] = None
        except Exception as cam_err:
            st.session_state["esp_image"] = None
            st.session_state["esp_error"] = f"Could not capture from http://{cam_ip}/capture — {cam_err}"

    if st.session_state.get("esp_error"):
        st.error(st.session_state["esp_error"])

    esp_image = st.session_state.get("esp_image")
    if esp_image is None and not st.session_state.get("esp_error"):
        st.markdown(f"""
        <div class="panel" style="text-align:center; padding:60px 20px; color:{TEXT_MUTED};">
            <div class="eyebrow">NODE IDLE</div>
            <div style="margin-top:8px;">Aim the ESP32-CAM at the specimen and press Capture & Inspect.</div>
        </div>
        """, unsafe_allow_html=True)
    elif esp_image is not None:
        label = render_inspection_results(esp_image, source_label="ESP32-CAM CAPTURED FRAME")

        if st.session_state.get("esp_alert_pending"):
            s_snap = st.session_state.get("esp_sensors", {})
            crack_px = 0.0
            if label == "Positive":
                opencv_img = cv2.cvtColor(np.array(esp_image.convert("RGB")), cv2.COLOR_RGB2BGR)
                crack_px, _ = calculate_crack_length(opencv_img)
                crack_px = round(crack_px, 2)

            delta_t = float(s_snap.get("delta_t") or 0.0)
            peak_g = float(s_snap.get("peak_g") or 0.0)
            shocks = int(s_snap.get("vib_events") or 0)

            if label == "Positive":
                if abs(delta_t) >= 2.0 and (peak_g >= 0.25 or shocks >= 5):
                    driver = "Combined Thermal + Cyclic Fatigue"
                elif abs(delta_t) >= 2.0:
                    driver = "Thermal Expansion Stress"
                elif peak_g >= 0.25 or shocks >= 5:
                    driver = "Mechanical Vibration Fatigue"
                else:
                    driver = "Pre-existing / Static Defect"
            else:
                driver = "Elevated Stress Watch" if (abs(delta_t) >= 2.0 or peak_g >= 0.25) else "Normal Operating State"

            st.session_state["telemetry_log"].append({
                "Timestamp": time.strftime("%H:%M:%S"),
                "Verdict": "DEFECT" if label == "Positive" else "CLEAR",
                "Crack Length (px)": crack_px,
                "Temp (°C)": s_snap.get("temp_c"),
                "ΔT (°C)": delta_t,
                "Peak Accel (g)": round(peak_g, 3),
                "Vibration Shocks": shocks,
                "Stress Correlation": driver
            })

            if devkit_ip.strip():
                endpoint = "defect" if label == "Positive" else "clear"
                try:
                    requests.get(f"http://{devkit_ip.strip()}/{endpoint}", timeout=3)
                    st.toast(f"Buzzer & telemetry updated: {endpoint.upper()}")
                except Exception as hw_err:
                    st.warning(f"Prediction finished, but DevKit unreachable: {hw_err}")

            st.session_state["esp_alert_pending"] = False

    if st.session_state.get("telemetry_log"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-label">THERMAL-STRESS & MPU6050 FATIGUE CORRELATION LOG</div>', unsafe_allow_html=True)
        t_df = pd.DataFrame(st.session_state["telemetry_log"])
        st.dataframe(t_df, use_container_width=True, hide_index=True)

        if len(t_df) >= 2:
            st.line_chart(t_df.set_index("Timestamp")[["Crack Length (px)", "ΔT (°C)", "Peak Accel (g)"]])

        t_csv = t_df.to_csv(index=False).encode("utf-8")
        st.download_button("Export Thermal-Fatigue Correlation CSV", data=t_csv, file_name="thermal_fatigue_log.csv", mime="text/csv")

# ---------------- Mobile Camera ----------------
with tab_cam:
    st.markdown('<div class="panel-label">LIVE CAMERA CAPTURE</div>', unsafe_allow_html=True)
    cam_file = st.camera_input(
        "Point rear camera at surface and capture",
        key="mobile_cam",
        label_visibility="collapsed"
    )

    if cam_file is None:
        st.markdown(f"""
        <div class="panel" style="text-align:center; padding:60px 20px; color:{TEXT_MUTED};">
            <div class="eyebrow">CAMERA READY</div>
            <div style="margin-top:8px;">Allow camera access and capture a surface image above to begin inspection.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(cam_file).convert("RGB")
        image.thumbnail((800, 800))
        render_inspection_results(image, source_label="CAPTURED FRAME")

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
            confidence_pct = round(max(probs) * 100, 2)
            verdict = "DEFECT" if label == "Positive" else "CLEAR"

            crack_length_px = 0.0
            if label == "Positive":
                opencv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
                crack_length_px, _ = calculate_crack_length(opencv_img)
                crack_length_px = round(crack_length_px, 2)

            results.append({
                "Filename": file.name,
                "Verdict": verdict,
                "Confidence (%)": confidence_pct,
                "Crack Length (px)": crack_length_px
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
            color = "rgba(198,40,40,0.10)" if row["Verdict"] == "DEFECT" else "rgba(27,127,75,0.07)"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            results_df.style.apply(highlight_defect, axis=1),
            use_container_width=True,
            hide_index=True
        )

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Export log as CSV", data=csv, file_name="inspection_log.csv", mime="text/csv")