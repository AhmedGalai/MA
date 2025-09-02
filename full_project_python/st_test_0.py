# streamlit_app.py
# Run: streamlit run streamlit_app.py

import os, cv2, time, base64, io, threading
import numpy as np
import streamlit as st
from PIL import Image

# ----- CONFIG -----
MODELS_DIR = "models"
INTRINSICS_URL = "http://localhost:8000/intrinsics"
POSE_API_URL   = "http://localhost:8000/pose"
DEPTHSCALE = 0.001  # meters per count, must match API

# ====== Utility functions ======

def list_ply_models(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(".ply")]

def pil_to_bytes(img_pil, fmt="JPEG"):
    buf = io.BytesIO()
    img_pil.save(buf, format=fmt)
    return buf.getvalue()

def b64_bytes(b: bytes):
    return base64.b64encode(b).decode("ascii")

def np16_png_bytes(arr_u16):
    ok, buf = cv2.imencode(".png", arr_u16)
    return buf.tobytes()

def apply_roi_mask(bgr, center, radius):
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    masked = cv2.bitwise_and(bgr, bgr, mask=mask)
    return masked, mask

# ====== Streamlit UI ======

st.set_page_config("Pose Estimation Demo", layout="wide")
st.title("Camera Pose Estimation with ROI + Mesh Overlay")

# Model selection
models = list_ply_models(MODELS_DIR)
model_choice = st.selectbox("Select a .ply model:", models)

# ROI state
if "roi_center" not in st.session_state:
    st.session_state.roi_center = None
if "roi_radius" not in st.session_state:
    st.session_state.roi_radius = None
if "estimating" not in st.session_state:
    st.session_state.estimating = False

# Buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Reset ROI"):
        st.session_state.roi_center = None
        st.session_state.roi_radius = None
with col2:
    if not st.session_state.estimating:
        if st.button("Start Estimation"):
            st.session_state.estimating = True
    else:
        if st.button("Stop Estimation"):
            st.session_state.estimating = False

# Camera capture (OpenCV)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_placeholder = st.empty()
info_placeholder = st.empty()

click_coords = st.session_state.get("click_coords", [])

# Use streamlit-webrtc or just poll frames
while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera error")
        break
    h, w = frame.shape[:2]

    # Overlay ROI if defined
    if st.session_state.roi_center is not None:
        cv2.circle(frame, st.session_state.roi_center, 5, (0,0,255), -1)
    if st.session_state.roi_center is not None and st.session_state.roi_radius is not None:
        cv2.circle(frame, st.session_state.roi_center, st.session_state.roi_radius, (0,255,0), 2)

    # Display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_placeholder.image(frame_pil, caption="Camera Feed (click inside image to set ROI)", use_column_width=True)

    # Handle clicks
    ev = st.session_state.get("last_clicked", None)
    if ev:
        x, y = ev
        if st.session_state.roi_center is None:
            st.session_state.roi_center = (x, y)
        elif st.session_state.roi_radius is None:
            dx = x - st.session_state.roi_center[0]
            dy = y - st.session_state.roi_center[1]
            r = int(np.hypot(dx, dy))
            st.session_state.roi_radius = r
        st.session_state.last_clicked = None

    # If estimating, simulate API call on ROI-masked snapshot
    if st.session_state.estimating and st.session_state.roi_center and st.session_state.roi_radius:
        masked, mask = apply_roi_mask(frame, st.session_state.roi_center, st.session_state.roi_radius)
        # TODO: run depth estimation + send to API
        info_placeholder.info(f"Would run estimation on ROI radius {st.session_state.roi_radius}")
    else:
        info_placeholder.text("")

    time.sleep(0.05)

