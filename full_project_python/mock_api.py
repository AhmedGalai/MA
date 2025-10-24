# mock_api.py
# pip install fastapi uvicorn transformers torch opencv-python-headless pillow requests numpy

import base64, io, requests, os
from typing import List, Optional
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import pipeline, infer_device

POSE_FORWARD_URL = "http://localhost:9000/pose"  # forward to mock_pose_api
MODELS_DIR = "models"  # folder containing .ply files

# ---------------- Model ----------------
device = infer_device()
checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
print(f"[INFO] Loading depth model on {device}")
pipe = pipeline("depth-estimation", model=checkpoint, device=device)

# ---------------- App ----------------
app = FastAPI(title="Main Mock API", version="2.3")

# top of file
import trimesh

# helpers section — add this
def load_ply_edges_from_bytes(ply_bytes: bytes):
    f = io.BytesIO(ply_bytes)
    mesh = trimesh.load(f, file_type='ply', process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces, dtype=np.int32)

    # Unique edges from faces
    edges = set()
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            e = (a, b) if a < b else (b, a)
            edges.add(e)
    E = np.array(list(edges), dtype=np.int32)

    # Center of mass
    try:
        com = np.asarray(mesh.center_mass, dtype=np.float64)
    except Exception:
        com = mesh.centroid.astype(np.float64)

    # Reasonable axis length
    ext = float(np.max(mesh.extents)) if mesh.extents is not None else float(np.linalg.norm(V.max(0) - V.min(0)))
    axis_len = 0.25 * (ext if ext > 0 else 1.0)

    return V, E, com, axis_len



# ---------------- Utils ----------------
def b64_to_img(b64str: str) -> np.ndarray:
    data = base64.b64decode(b64str.encode("ascii"))
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def img_to_b64(img: np.ndarray, fmt=".jpg") -> str:
    ok, buf = cv2.imencode(fmt, img)
    return base64.b64encode(buf).decode("ascii")

def list_ply_models(folder):
    try:
        return [f for f in os.listdir(folder) if f.lower().endswith(".ply")]
    except FileNotFoundError:
        return []

def load_mesh_as_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

# ---------------- Schemas ----------------
class DepthReq(BaseModel):
    rgb: str  # base64 JPEG

class IntrinsicsReq(BaseModel):
    left: str
    right: str

class RectifyReq(BaseModel):
    left: str
    right: str
    camera_matrix: List[List[float]]

class BoundingRectReq(BaseModel):
    left: str
    roi_center: List[int]
    roi_radius: int
    snapshot: str

class PoseReq(BaseModel):
    camera_matrix: List[List[float]]
    images: list
    mesh: Optional[str] = ""      # base64 mesh (will be filled if model provided)
    model: Optional[str] = None   # selected *.ply filename from UI
    mask: Optional[str] = None
    depthscale: float

# ---------------- Endpoints ----------------
@app.get("/models")
def list_models():
    names = list_ply_models(MODELS_DIR)
    return {"models": [{"name": n} for n in names]}

@app.post("/depth")
def depth(req: DepthReq):
    rgb = b64_to_img(req.rgb)
    rgb_pil = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    pred = pipe(rgb_pil)
    depth_pil = pred["depth"]
    buf = io.BytesIO()
    depth_pil.save(buf, format="PNG")
    return {"depth": base64.b64encode(buf.getvalue()).decode("ascii")}

@app.post("/intrinsics")
def intrinsics(req: IntrinsicsReq):
    left = b64_to_img(req.left)
    right = b64_to_img(req.right)
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(left, None)
    k2, d2 = orb.detectAndCompute(right, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(d1, d2) if d1 is not None and d2 is not None else []
    matches = sorted(matches, key=lambda x: x.distance)[:30]
    out = cv2.drawMatches(left, k1, right, k2, matches, None, flags=2)
    h, w = left.shape[:2]
    fx = fy = 0.8 * w
    cx, cy = w/2, h/2
    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    return {"camera_matrix": K, "debug_image": img_to_b64(out)}

@app.post("/rectify")
def rectify(req: RectifyReq):
    left = b64_to_img(req.left)
    out = left.copy()
    cv2.putText(out, "Rectified", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return {"rectified": img_to_b64(out)}

@app.post("/bounding_rect")
def bounding_rect(req: BoundingRectReq):
    left = b64_to_img(req.left)
    snap = b64_to_img(req.snapshot)
    x, y = req.roi_center
    r = req.roi_radius
    template = left[max(y-r,0):y+r, max(x-r,0):x+r]
    mask = np.zeros(snap.shape[:2], dtype=np.uint8)
    if template.size>0 and snap.shape[0]>=template.shape[0] and snap.shape[1]>=template.shape[1]:
        res = cv2.matchTemplate(snap, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        th, tw = template.shape[:2]
        top_left = max_loc
        bottom_right = (top_left[0]+tw, max_loc[1]+th)
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    return {"mask": img_to_b64(mask, fmt=".png")}

@app.post("/pose")
def pose(req: PoseReq):
    payload = req.dict()
    model_name = payload.pop("model", None)

    if (not payload.get("mesh")) and model_name:
        available = set(list_ply_models(MODELS_DIR))
        if model_name not in available:
            raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'. Available: {sorted(available)}")
        mesh_b64 = load_mesh_as_b64(os.path.join(MODELS_DIR, model_name))
        payload["mesh"] = mesh_b64

    # forward EXACTLY per the mock-pose-api spec (no 'model' key)
    resp = requests.post(POSE_FORWARD_URL, json=payload, timeout=20)
    return resp.json()


# endpoint to fetch mesh by name (client uses this to draw overlay)
from fastapi import HTTPException, Query

@app.get("/model")
def get_model(name: str = Query(..., description="PLY filename, e.g. bunny.ply")):
    available = set(list_ply_models(MODELS_DIR))
    if name not in available:
        raise HTTPException(status_code=404, detail=f"Unknown model '{name}'. Available: {sorted(available)}")
    try:
        mesh_b64 = load_mesh_as_b64(os.path.join(MODELS_DIR, name))
        return {"name": name, "mesh": mesh_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read '{name}': {e}")