import os, cv2, io, base64, requests, time
import numpy as np
import streamlit as st
from PIL import Image

MODELS_DIR = "models"
API = "http://localhost:8000"

# ---------- Helpers ----------
def list_ply_models(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(".ply")]

def pil_to_b64(img: Image.Image, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def b64_to_pil(b64):
    data = base64.b64decode(b64.encode())
    return Image.open(io.BytesIO(data))

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

def apply_circle_mask(pil_img, center, radius):
    arr = np.array(pil_img)
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    masked = cv2.bitwise_and(arr, arr, mask=mask)
    return Image.fromarray(masked), Image.fromarray(mask)

# ---------- Streamlit ----------
st.set_page_config("Pose Estimation", layout="wide")
st.title("Camera Pose Estimation Demo")

tab1, tab2 = st.tabs(["1️⃣ Intrinsics + ROI", "2️⃣ Pose Estimation"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Step 1: Estimate Intrinsics + Select ROI + Model")

    # Model select
    model_choice = st.selectbox("Select a .ply model:", list_ply_models(MODELS_DIR))

    if "left_img" not in st.session_state:
        st.session_state.left_img = None
        st.session_state.right_img = None
        st.session_state.camera_matrix = None
        st.session_state.rectified_left = None
        st.session_state.roi_center = None
        st.session_state.roi_radius = None
        st.session_state.roi_mask = None
        st.session_state.bounding_mask = None

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

    # Call intrinsics API
    if st.session_state.left_img and st.session_state.right_img:
        if st.button("Send to API for Intrinsics"):
            payload = {
                "left": pil_to_b64(st.session_state.left_img),
                "right": pil_to_b64(st.session_state.right_img),
            }
            try:
                r = requests.post(f"{API}/intrinsics", json=payload, timeout=10)
                r.raise_for_status()
                data = r.json()
                st.session_state.camera_matrix = data.get("camera_matrix")
                st.image(b64_to_pil(data["debug_image"]), caption="ORB Matches")
                # rectify
                rect_req = {"left": payload["left"], "right": payload["right"], "camera_matrix": st.session_state.camera_matrix}
                r2 = requests.post(f"{API}/rectify", json=rect_req).json()
                st.session_state.rectified_left = b64_to_pil(r2["rectified"])
                col1, col2 = st.columns(2)
                col1.image(st.session_state.left_img, caption="Original Left")
                col2.image(st.session_state.rectified_left, caption="Rectified Left")
                st.success(f"Got intrinsics: {st.session_state.camera_matrix}")
            except Exception as e:
                st.error(f"API error: {e}")

    # ROI selection on left
    if st.session_state.left_img:
        cx = st.slider("ROI center X", 0, st.session_state.left_img.width-1, st.session_state.left_img.width//2)
        cy = st.slider("ROI center Y", 0, st.session_state.left_img.height-1, st.session_state.left_img.height//2)
        r  = st.slider("ROI radius", 10, 200, 50)

        if st.button("Set ROI"):
            st.session_state.roi_center = (cx, cy)
            st.session_state.roi_radius = r
            masked, roi_mask = apply_circle_mask(st.session_state.left_img, (cx,cy), r)
            st.session_state.roi_mask = roi_mask
            st.image(masked, caption="ROI Masked Left")

        if st.session_state.roi_center:
            # bounding rect using left as snapshot
            payload = {
                "left": pil_to_b64(st.session_state.left_img),
                "roi_center": list(st.session_state.roi_center),
                "roi_radius": st.session_state.roi_radius,
                "snapshot": pil_to_b64(st.session_state.left_img),
            }
            r = requests.post(f"{API}/bounding_rect", json=payload).json()
            st.session_state.bounding_mask = b64_to_pil(r["mask"])
            st.image(st.session_state.bounding_mask, caption="Bounding Rect Mask")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Step 2: Pose Estimation")

    rate = st.number_input("Sampling rate (Hz)", 0.1, 5.0, 1.0)

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
        snap = get_camera_frame()
        if snap and st.session_state.left_img and st.session_state.roi_center:
            # recompute bounding rect mask for snapshot
            payload = {
                "left": pil_to_b64(st.session_state.left_img),
                "roi_center": list(st.session_state.roi_center),
                "roi_radius": st.session_state.roi_radius,
                "snapshot": pil_to_b64(snap),
            }
            r = requests.post(f"{API}/bounding_rect", json=payload).json()
            bounding_mask = b64_to_pil(r["mask"])

            col1, col2 = st.columns(2)
            col1.image(snap, caption="RGB Snapshot")
            col2.image(bounding_mask, caption="Bounding Rect (Snapshot)")

            # Fake depth map = grayscale of masked region
            arr = np.array(snap.convert("L"))
            depth_pil = Image.fromarray(arr)
            st.image(depth_pil, caption="Estimated Depth (mock)")

            # Pose API call
            pose_req = {
                "camera_matrix": st.session_state.camera_matrix or [[700,0,320],[0,700,240],[0,0,1]],
                "images": [{"filename":"snap","rgb": pil_to_b64(snap),"depth":""}],
                "mesh": "",  # .ply can be read and base64 if needed
                "mask": r["mask"],
                "depthscale": 0.001
            }
            resp = requests.post(f"{API}/pose", json=pose_req, timeout=10).json()
            st.json(resp)

            # Mock overlay: just show RGB + bounding box
            snap_np = np.array(snap)
            mask_np = np.array(bounding_mask.convert("L"))
            ys, xs = np.where(mask_np>0)
            if len(xs)>0 and len(ys)>0:
                x1,y1,x2,y2 = min(xs),min(ys),max(xs),max(ys)
                cv2.rectangle(snap_np, (x1,y1), (x2,y2), (255,0,0), 2)
            st.image(snap_np, caption=f"RGB + Mock Mesh Overlay ({model_choice})")
