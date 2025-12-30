# pip install streamlit requests opencv-python-headless numpy
import streamlit as st, cv2, base64, io, requests
import numpy as np
from PIL import Image

API = "http://localhost:8000"

def img_to_b64(pil, fmt="JPEG"):
    buf=io.BytesIO(); pil.save(buf,fmt=fmt); return base64.b64encode(buf.getvalue()).decode()

def b64_to_pil(b64):
    img = base64.b64decode(b64.encode()); arr=np.frombuffer(img,np.uint8)
    return Image.open(io.BytesIO(arr))

def capture_frame():
    cap=cv2.VideoCapture(0)
    ret,frame=cap.read()
    cap.release()
    if not ret: return None
    return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

st.set_page_config("Pose Estimation", layout="wide")
st.title("ROI + Intrinsics + Pose Estimation")

# ---- Tab: Combined ----
if "roi_center" not in st.session_state: st.session_state.roi_center=None
if "roi_radius" not in st.session_state: st.session_state.roi_radius=None
if "left_img" not in st.session_state: st.session_state.left_img=None
if "right_img" not in st.session_state: st.session_state.right_img=None
if "K" not in st.session_state: st.session_state.K=None

col1,col2=st.columns(2)
with col1:
    if st.button("Capture Left"):
        st.session_state.left_img=capture_frame()
    if st.session_state.left_img: st.image(st.session_state.left_img, caption="Left Image")

with col2:
    if st.button("Capture Right"):
        st.session_state.right_img=capture_frame()
    if st.session_state.right_img: st.image(st.session_state.right_img, caption="Right Image")

if st.session_state.left_img and st.session_state.right_img:
    if st.button("Estimate Intrinsics via API"):
        payload={"left": img_to_b64(st.session_state.left_img),
                 "right": img_to_b64(st.session_state.right_img)}
        r=requests.post(f"{API}/intrinsics",json=payload).json()
        st.session_state.K=r["camera_matrix"]
        st.image(b64_to_pil(r["debug_image"]), caption="ORB Matches")
        st.success(f"Got K={st.session_state.K}")

# ROI selection: simplified — user clicks manually set coords
if st.session_state.left_img:
    st.write("Click to set ROI center and radius (mock: input boxes)")
    cx=st.number_input("Center X",0,640,320)
    cy=st.number_input("Center Y",0,480,240)
    r =st.slider("Radius",10,200,50)
    if st.button("Set ROI"):
        st.session_state.roi_center=(int(cx),int(cy))
        st.session_state.roi_radius=int(r)

if st.session_state.roi_center:
    st.info(f"ROI center={st.session_state.roi_center}, r={st.session_state.roi_radius}")

# Pose estimation
if st.button("Estimate Pose (uses ROI + left + snapshot)"):
    snap=capture_frame()
    payload={"left": img_to_b64(st.session_state.left_img),
             "roi_center": list(st.session_state.roi_center),
             "roi_radius": st.session_state.roi_radius,
             "snapshot": img_to_b64(snap)}
    r=requests.post(f"{API}/bounding_rect",json=payload).json()
    mask=b64_to_pil(r["mask"])
    st.image(mask, caption="Bounding Rect Mask")
    pose_req={"images":[{"filename":"snap","rgb":img_to_b64(snap),"depth":""}],
              "mesh":"", "mask":r["mask"], "camera_matrix":st.session_state.K,"depthscale":0.001}
    pose=requests.post(f"{API}/pose",json=pose_req).json()
    st.success(f"Pose={pose['transformation_matrix'][0]}")

