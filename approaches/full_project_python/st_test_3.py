import os, cv2, io, base64, requests, time
import numpy as np
import streamlit as st
from PIL import Image
import trimesh

API = "http://localhost:8000"
MODELS_DIR = "models"

# ---------- Helpers ----------
def list_ply_models(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(".ply")]

def pil_to_b64(img: Image.Image, fmt="JPEG"):
    buf = io.BytesIO(); img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def b64_to_pil(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64.encode())))

def get_camera_frame():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read(); cap.release()
    if not ret: return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def apply_circle_mask(pil_img, center, radius):
    arr = np.array(pil_img)
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    masked = cv2.bitwise_and(arr, arr, mask=mask)
    return Image.fromarray(masked), Image.fromarray(mask)

def load_ply_edges(ply_path):
    mesh = trimesh.load(ply_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces, dtype=np.int32)
    edges = set()
    for f in F:
        for a,b in [(f[0],f[1]),(f[1],f[2]),(f[2],f[0])]:
            e=(a,b) if a<b else (b,a); edges.add(e)
    return V, np.array(list(edges))

def overlay_mesh_wireframe(bgr, V, edges, K, T, color=(0,255,0)):
    img = bgr.copy()
    R = np.array(T)[:3,:3]; t = np.array(T)[:3,3].reshape(3,1)
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    pts2d,_ = cv2.projectPoints(V.astype(np.float64), rvec, t.astype(np.float64), np.array(K), None)
    pts2d = pts2d.reshape(-1,2)
    for i,j in edges:
        x1,y1 = pts2d[i]; x2,y2 = pts2d[j]
        if np.isfinite([x1,y1,x2,y2]).all():
            cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),color,1,cv2.LINE_AA)
    return img

# ---------- Streamlit ----------
st.set_page_config("Pose Estimation", layout="wide")
st.title("Camera Pose Estimation Demo")

tab1, tab2 = st.tabs(["1️⃣ Intrinsics + ROI", "2️⃣ Pose Estimation"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Step 1: Estimate Intrinsics + Select ROI + Model")

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
        st.session_state.V, st.session_state.edges = load_ply_edges(os.path.join(MODELS_DIR, model_choice))

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

    # Intrinsics
    if st.session_state.left_img and st.session_state.right_img:
        if st.button("Send to API for Intrinsics"):
            payload = {"left": pil_to_b64(st.session_state.left_img),
                       "right": pil_to_b64(st.session_state.right_img)}
            r = requests.post(f"{API}/intrinsics", json=payload, timeout=10).json()
            st.session_state.camera_matrix = r["camera_matrix"]
            st.image(b64_to_pil(r["debug_image"]), caption="ORB Matches")
            rect_req = {"left": payload["left"], "right": payload["right"], "camera_matrix": st.session_state.camera_matrix}
            r2 = requests.post(f"{API}/rectify", json=rect_req).json()
            st.session_state.rectified_left = b64_to_pil(r2["rectified"])
            col1, col2 = st.columns(2)
            col1.image(st.session_state.left_img, caption="Original Left")
            col2.image(st.session_state.rectified_left, caption="Rectified Left")
            st.success(f"Got intrinsics: {st.session_state.camera_matrix}")

    # ROI selection
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
            payload = {"left": pil_to_b64(st.session_state.left_img),
                       "roi_center": list(st.session_state.roi_center),
                       "roi_radius": st.session_state.roi_radius,
                       "snapshot": pil_to_b64(st.session_state.left_img)}
            r = requests.post(f"{API}/bounding_rect", json=payload).json()
            st.session_state.bounding_mask = b64_to_pil(r["mask"])
            st.image(st.session_state.bounding_mask, caption="Bounding Rect Mask")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Step 2: Pose Estimation (Real-Time)")

    rate = st.number_input("Sampling rate (Hz)", 0.1, 5.0, 1.0)
    delay = 1.0/float(rate)

    if "estimating" not in st.session_state:
        st.session_state.estimating = False

    start_btn = st.button("Start Estimation") if not st.session_state.estimating else None
    stop_btn  = st.button("Stop Estimation") if st.session_state.estimating else None

    if start_btn: st.session_state.estimating = True
    if stop_btn: st.session_state.estimating = False

    frame_ph = st.empty()
    depth_ph = st.empty()
    overlay_ph = st.empty()

    while st.session_state.estimating:
        snap = get_camera_frame()
        if snap and st.session_state.left_img and st.session_state.roi_center:
            # Depth
            r = requests.post(f"{API}/depth", json={"rgb": pil_to_b64(snap)}, timeout=20).json()
            depth_pil = b64_to_pil(r["depth"])

            # Bounding rect for snapshot
            payload = {"left": pil_to_b64(st.session_state.left_img),
                       "roi_center": list(st.session_state.roi_center),
                       "roi_radius": st.session_state.roi_radius,
                       "snapshot": pil_to_b64(snap)}
            r2 = requests.post(f"{API}/bounding_rect", json=payload).json()
            bounding_mask = b64_to_pil(r2["mask"])

            # Pose
            pose_req = {"camera_matrix": st.session_state.camera_matrix or [[700,0,320],[0,700,240],[0,0,1]],
                        "images": [{"filename":"snap","rgb": pil_to_b64(snap),"depth": pil_to_b64(depth_pil, fmt="PNG")}],
                        "mesh": "",
                        "mask": r2["mask"],
                        "depthscale": 0.001}
            resp = requests.post(f"{API}/pose", json=pose_req, timeout=20).json()

            # Overlay both poses
            bgr = cv2.cvtColor(np.array(snap), cv2.COLOR_RGB2BGR)
            for idx,T in enumerate(resp["transformation_matrix"]):
                color = (0,255-100*idx,100*idx)
                bgr = overlay_mesh_wireframe(bgr, st.session_state.V, st.session_state.edges,
                                             st.session_state.camera_matrix, np.array(T), color=color)

            # Update UI
            frame_ph.image(snap, caption="RGB Snapshot")
            depth_ph.image(depth_pil, caption="Estimated Depth")
            overlay_ph.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Mesh Overlay (multiple poses)")

        time.sleep(delay)
        st.experimental_rerun()
