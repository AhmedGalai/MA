# Requirements:
#   pip install transformers torch pillow opencv-python trimesh requests numpy
# Optional: install torch with CUDA for speed.

import base64, io, json, threading, time, sys
import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import numpy as np
import requests
import trimesh
from PIL import Image, ImageTk
from transformers import pipeline, infer_device


# ----------------------- CONFIG -----------------------
POSE_API_URL     = "http://localhost:8000/pose"     # <-- set your endpoint
INTRINSICS_URL   = "http://localhost:8000/intrinsics"  # <-- returns {"camera_matrix":[[...],[...],[...]]}
MESH_PATH        = r"cube.ply"     # <-- set .ply path
MASK_PATH        = ""                                # optional mask PNG path (can be "")
CAM_INDEX        = 0
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480
DEPTH_SNAP_RATE_HZ_DEFAULT = 0.5

# Depth encoding strategy: 16-bit PNG with depthscale = meters per count (0.001 -> 1mm)
DEPTH_TO_UINT16_SCALE = 1000.0   # counts per meter (depth_uint16 = depth_m * 1000)
DEPTHSCALE_VALUE      = 1.0 / DEPTH_TO_UINT16_SCALE  # meters per count

# ------------------- HELPERS: IO / ENCODE -------------------

def pil_to_jpeg_bytes(img_pil, quality=90):
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def np16_png_bytes(arr_u16):
    # arr_u16: HxW np.uint16
    ok, buf = cv2.imencode(".png", arr_u16)
    if not ok:
        raise RuntimeError("PNG encode failed for uint16 depth.")
    return buf.tobytes()

def b64_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def load_intrinsics_from_api(url: str) -> np.ndarray:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    K = np.array(data["camera_matrix"], dtype=float).reshape(3,3)
    return K

def load_mesh_vertices_faces(ply_path: str):
    mesh = trimesh.load(ply_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        # If it's a Scene, merge
        mesh = trimesh.util.concatenate(tuple(m for m in mesh.geometry.values()))
    V = np.asarray(mesh.vertices, dtype=np.float32)  # (N,3)
    F = np.asarray(mesh.faces, dtype=np.int32)       # (M,3)
    return V, F

def unique_edges_from_faces(F: np.ndarray):
    # Return sorted unique edges (Nx2)
    E = set()
    for f in F:
        i, j, k = int(f[0]), int(f[1]), int(f[2])
        for a,b in [(i,j),(j,k),(k,i)]:
            e = (a,b) if a<b else (b,a)
            E.add(e)
    return np.array(sorted(list(E)), dtype=np.int32)

# ------------------- MODEL: DEPTH PIPELINE -------------------

def build_depth_pipeline():
    device = infer_device()
    ckpt = "depth-anything/Depth-Anything-V2-base-hf"
    print(f"[INFO] Loading depth model '{ckpt}' on device: {device}", flush=True)
    return pipeline("depth-estimation", model=ckpt, device=device)

def infer_depth_float(pipeline_obj, bgr_img: np.ndarray) -> np.ndarray:
    # BGR -> PIL RGB
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pred = pipeline_obj(pil_img)
    depth_pil = pred["depth"]  # PIL L, but values are float-scaled internally by the model
    # Convert to float32; many HF depth heads output ~linear but arbitrary scale
    d = np.array(depth_pil, dtype=np.float32)
    return d

def colorize_depth_for_display(depth: np.ndarray) -> Image.Image:
    d = depth.copy()
    if np.isnan(d).any():
        d = np.nan_to_num(d, nan=0.0)
    dmin, dmax = float(np.min(d)), float(np.max(d))
    if dmax - dmin < 1e-6:
        d8 = np.zeros_like(d, dtype=np.uint8)
    else:
        d8 = ((d - dmin) * (255.0 / (dmax - dmin))).astype(np.uint8)
    d_color = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    d_rgb = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)
    return Image.fromarray(d_rgb)

# ------------------- PROJECTION / OVERLAY -------------------

def overlay_mesh_wireframe(bgr_img: np.ndarray, V: np.ndarray, edges: np.ndarray,
                           K: np.ndarray, T_cam_obj: np.ndarray,
                           color=(0,255,0), thickness=1) -> np.ndarray:
    """
    Project mesh vertices with camera intrinsics K and pose T (4x4, object->camera),
    draw wireframe edges on a copy of bgr_img.
    """
    img = bgr_img.copy()
    R = T_cam_obj[:3,:3].astype(np.float64)
    t = T_cam_obj[:3, 3].reshape(3,1).astype(np.float64)
    dist = np.zeros((5,1), dtype=np.float64)  # assume no distortion (adjust if you have it)

    # OpenCV wants rvec/tvec
    rvec, _ = cv2.Rodrigues(R)
    pts2d, _ = cv2.projectPoints(V.astype(np.float64), rvec, t, K.astype(np.float64), dist)
    pts2d = pts2d.reshape(-1,2)

    h, w = img.shape[:2]
    # Draw edges
    for (i,j) in edges:
        x1,y1 = pts2d[i]
        x2,y2 = pts2d[j]
        if np.isfinite([x1,y1,x2,y2]).all():
            # Optional: clip if both behind camera (Z<=0). projectPoints already clips, but we can skip.
            cv2.line(img,
                     (int(round(x1)), int(round(y1))),
                     (int(round(x2)), int(round(y2))),
                     color, thickness, lineType=cv2.LINE_AA)
    return img

# ------------------- API CALL -------------------

def build_request_payload(K: np.ndarray, rgb_pil: Image.Image, depth_float: np.ndarray,
                          mesh_bytes: bytes, mask_bytes: bytes|None, depthscale: float):
    # Encode RGB as JPEG
    rgb_jpg_b = pil_to_jpeg_bytes(rgb_pil, quality=90)

    # Encode depth as 16-bit PNG with fixed scale (counts per meter)
    depth_u16 = np.clip(depth_float * DEPTH_TO_UINT16_SCALE, 0, 65535).astype(np.uint16)
    depth_png_b = np16_png_bytes(depth_u16)

    payload = {
        "camera_matrix": [[str(K[0,0]), str(K[0,1]), str(K[0,2])],
                          [str(K[1,0]), str(K[1,1]), str(K[1,2])],
                          [str(K[2,0]), str(K[2,1]), str(K[2,2])]],
        "images": [
            {
                "filename": "snapshot",
                "rgb":   b64_bytes(rgb_jpg_b),
                "depth": b64_bytes(depth_png_b),
            }
        ],
        "mesh": b64_bytes(mesh_bytes),
        "mask": b64_bytes(mask_bytes) if mask_bytes else "",
        "depthscale": str(depthscale),
    }
    return payload

def post_pose_request(url: str, payload: dict) -> np.ndarray|None:
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "transformation_matrix" not in data or not data["transformation_matrix"]:
        return None
    # Take the first solution
    T = np.array(data["transformation_matrix"][0], dtype=float)
    if T.shape != (4,4):
        # Some APIs wrap twice; handle [["r.."]]
        T = np.array(data["transformation_matrix"][0][0], dtype=float)
    return T

# ------------------- TK APP -------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RGB + Depth + Mesh Overlay (Pose API)")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Controls
        ctrl = ttk.Frame(self); ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Label(ctrl, text="Camera:").pack(side=tk.LEFT)
        self.cam_var = tk.IntVar(value=CAM_INDEX)
        ttk.Entry(ctrl, textvariable=self.cam_var, width=4).pack(side=tk.LEFT, padx=(2,8))

        ttk.Label(ctrl, text="Rate (Hz):").pack(side=tk.LEFT)
        self.rate_var = tk.DoubleVar(value=DEPTH_SNAP_RATE_HZ_DEFAULT)
        ttk.Entry(ctrl, textvariable=self.rate_var, width=6).pack(side=tk.LEFT, padx=(2,8))

        self.start_btn = ttk.Button(ctrl, text="Start", command=self.start); self.start_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn  = ttk.Button(ctrl, text="Stop",  command=self.stop,  state=tk.DISABLED); self.stop_btn.pack(side=tk.LEFT, padx=4)

        self.status = tk.StringVar(value="idle")
        ttk.Label(ctrl, textvariable=self.status).pack(side=tk.LEFT, padx=10)

        # Views
        views = ttk.Frame(self); views.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        ttk.Label(views, text="RGB (live)").grid(row=0, column=0)
        ttk.Label(views, text="Depth (snapshot)").grid(row=0, column=1)
        ttk.Label(views, text="Overlay (pose)").grid(row=0, column=2)

        self.rgb_panel    = ttk.Label(views); self.rgb_panel.grid(row=1, column=0, padx=4)
        self.depth_panel  = ttk.Label(views); self.depth_panel.grid(row=1, column=1, padx=4)
        self.overlay_panel= ttk.Label(views); self.overlay_panel.grid(row=1, column=2, padx=4)

        # State
        self.cap = None
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_bgr = None
        self.rgb_photo = None
        self.depth_photo = None
        self.overlay_photo = None

        # Load heavy stuff up front
        self.status.set("loading model + intrinsics + mesh...")
        self.pipe = build_depth_pipeline()
        self.K = load_intrinsics_from_api(INTRINSICS_URL)
        self.V, self.F = load_mesh_vertices_faces(MESH_PATH)
        self.edges = unique_edges_from_faces(self.F)
        with open(MESH_PATH, "rb") as f:
            self.mesh_bytes = f.read()
        self.mask_bytes = None
        if MASK_PATH:
            with open(MASK_PATH, "rb") as f:
                self.mask_bytes = f.read()
        self.status.set("ready")

        # UI tick
        self.after(33, self.ui_tick)

    def start(self):
        if self.cap is not None:
            return
        idx = int(self.cam_var.get())
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            self.status.set(f"cannot open camera {idx}")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.stop_event.clear()

        self.preview_thread = threading.Thread(target=self.preview_loop, daemon=True)
        self.snapshot_thread= threading.Thread(target=self.snapshot_loop, daemon=True)
        self.preview_thread.start(); self.snapshot_thread.start()

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status.set("running")

    def stop(self):
        self.stop_event.set()
        for t in [getattr(self, 'preview_thread', None), getattr(self, 'snapshot_thread', None)]:
            if t and t.is_alive():
                t.join(timeout=1.0)
        if self.cap is not None:
            try: self.cap.release()
            except: pass
            self.cap = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status.set("stopped")

    def on_close(self):
        self.stop()
        self.destroy()

    def preview_loop(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.status.set("camera read failed")
                break
            with self.frame_lock:
                self.latest_bgr = frame
            time.sleep(0.01)  # ~100 Hz

    def snapshot_loop(self):
        while not self.stop_event.is_set():
            # clamp rate to [0.1, 5] Hz
            try:
                rate = float(self.rate_var.get())
            except Exception:
                rate = DEPTH_SNAP_RATE_HZ_DEFAULT
            rate = max(0.1, min(rate, 5.0))
            time.sleep(1.0 / rate)

            with self.frame_lock:
                snap = None if self.latest_bgr is None else self.latest_bgr.copy()
            if snap is None:
                continue

            try:
                # Optional downscale to max side ~512 for faster depth
                h, w = snap.shape[:2]
                scale = 512.0 / max(h, w)
                proc = cv2.resize(snap, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else snap

                # Depth inference
                depth = infer_depth_float(self.pipe, proc)
                depth_disp = colorize_depth_for_display(depth)

                # Build API payload
                rgb_for_api = Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
                payload = build_request_payload(self.K, rgb_for_api, depth, self.mesh_bytes, self.mask_bytes, DEPTHSCALE_VALUE)

                # POST to pose API
                T = post_pose_request(POSE_API_URL, payload)
                if T is None:
                    self.status.set("pose API returned no transform")
                    # still show depth
                    self.depth_photo = ImageTk.PhotoImage(depth_disp.resize((640, int(640*depth_disp.height/max(1, depth_disp.width))), Image.NEAREST))
                    continue

                # Overlay mesh on the ORIGINAL resolution frame (snap)
                overlay_bgr = overlay_mesh_wireframe(snap, self.V, self.edges, self.K, T, color=(0,255,0), thickness=1)

                # Update UI images
                self.depth_photo   = ImageTk.PhotoImage(depth_disp.resize((640, int(640*depth_disp.height/max(1, depth_disp.width))), Image.NEAREST))
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                overlay_pil = Image.fromarray(overlay_rgb)
                self.overlay_photo = ImageTk.PhotoImage(overlay_pil.resize((640, int(640*overlay_pil.height/max(1, overlay_pil.width))), Image.BILINEAR))

            except Exception as e:
                self.status.set(f"snapshot error: {e}")

    def ui_tick(self):
        # RGB live
        with self.frame_lock:
            frame = None if self.latest_bgr is None else self.latest_bgr.copy()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            disp = Image.fromarray(rgb)
            w = 640; h = int(w * disp.height / max(1, disp.width))
            self.rgb_photo = ImageTk.PhotoImage(disp.resize((w, h), Image.BILINEAR))
            self.rgb_panel.configure(image=self.rgb_photo)

        # Depth snapshot
        if self.depth_photo is not None:
            self.depth_panel.configure(image=self.depth_photo)

        # Overlay view
        if self.overlay_photo is not None:
            self.overlay_panel.configure(image=self.overlay_photo)

        self.after(33, self.ui_tick)


# ----------------------- MAIN -----------------------

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()

