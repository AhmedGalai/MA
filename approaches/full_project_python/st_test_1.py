# streamlit_app_tabs.py
# Run: streamlit run streamlit_app_tabs.py

import os, cv2, io, base64
import numpy as np
import streamlit as st
from PIL import Image
import requests

MODELS_DIR = "models"
INTRINSICS_API = "http://localhost:8000/intrinsics"   # adjust
POSE_API       = "http://localhost:8000/pose"

# ---------- Helpers ----------

def list_ply_models(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(".ply")]

def pil_to_b64(img: Image.Image, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def get_camera_frame():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

# ---------- Streamlit ----------

st.set_page_config("Pose Estimation", layout="wide")
st.title("Camera Pose Estimation with ROI + Mesh Overlay")

tab1, tab2, tab3 = st.tabs(["1️⃣ Select ROI", "2️⃣ Estimate Intrinsics", "3️⃣ Estimate Pose"])

# ------ TAB 1: ROI ------
with tab1:
    st.subheader("Step 1: Select ROI")
    model_choice = st.selectbox("Select a .ply model:", list_ply_models(MODELS_DIR))

    if "roi_center" not in st.session_state:
        st.session_state.roi_center = None
        st.session_state.roi_radius = None

    img = get_camera_frame()
    if img:
        st.image(img, caption="Camera Feed (ROI selection TBD)", use_column_width=True)
    st.button("Reset ROI", on_click=lambda: (st.session_state.update({"roi_center": None, "roi_radius": None})))

    st.info("👉 To implement ROI: use `streamlit-drawable-canvas` to capture clicks/drawings.")

# ------ TAB 2: Intrinsics ------
with tab2:
    st.subheader("Step 2: Estimate Intrinsics")

    if "left_img" not in st.session_state:
        st.session_state.left_img = None
        st.session_state.right_img = None
        st.session_state.camera_matrix = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Capture Left Image"):
            st.session_state.left_img = get_camera_frame()
        if st.session_state.left_img:
            st.image(st.session_state.left_img, caption="Left Image")
    with col2:
        if st.button("Capture Right Image"):
            st.session_state.right_img = get_camera_frame()
        if st.session_state.right_img:
            st.image(st.session_state.right_img, caption="Right Image")

    if st.session_state.left_img and st.session_state.right_img:
        if st.button("Send to API for Intrinsics"):
            payload = {
                "left": pil_to_b64(st.session_state.left_img),
                "right": pil_to_b64(st.session_state.right_img),
            }
            try:
                r = requests.post(INTRINSICS_API, json=payload, timeout=10)
                r.raise_for_status()
                st.session_state.camera_matrix = r.json().get("camera_matrix")
                st.success(f"Got intrinsics: {st.session_state.camera_matrix}")
            except Exception as e:
                st.error(f"API error: {e}")

# ------ TAB 3: Pose ------
with tab3:
    st.subheader("Step 3: Estimate Pose")

    if "estimating" not in st.session_state:
        st.session_state.estimating = False

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.estimating:
            if st.button("Start Estimation"):
                st.session_state.estimating = True
        else:
            if st.button("Stop Estimation"):
                st.session_state.estimating = False

    if st.session_state.estimating:
        frame = get_camera_frame()
        if frame:
            st.image(frame, caption="RGB + Mesh Overlay (to be implemented)", use_column_width=True)
            st.info("👉 Here you would run depth + send payload with ROI mask + mesh to the Pose API, then overlay the mesh.")

