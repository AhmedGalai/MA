import tkinter as tk
from tkinter import ttk
import cv2, requests, base64, io, time, threading, os, queue
import numpy as np
from PIL import Image, ImageTk
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

def tk_image(pil, size=(320,240)):
    return ImageTk.PhotoImage(pil.resize(size))

# ---------- GUI ----------
class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tk Pose Estimation Client")

        # State
        self.left_img = None
        self.right_img = None
        self.K = [[700,0,320],[0,700,240],[0,0,1]]
        self.roi_center = None
        self.roi_radius = None
        self.roi_preview = None
        self.model_choice = tk.StringVar(value=list_ply_models(MODELS_DIR)[0])
        self.V, self.edges = load_ply_edges(os.path.join(MODELS_DIR, self.model_choice.get()))
        self.running = False
        self.preview_running = False
        self.rate = 1.0
        self.tk_left = None

        # Queues
        self.snap_queue = queue.Queue(maxsize=5)
        self.mask_queue = queue.Queue(maxsize=5)
        self.pose_queue = queue.Queue(maxsize=5)

        # Camera preview at top
        preview_frame = ttk.Frame(root)
        preview_frame.pack(side="top", pady=5, fill="x")
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(side="left")
        ttk.Button(preview_frame, text="Start Preview", command=self.start_preview).pack(side="left", padx=5)
        ttk.Button(preview_frame, text="Stop Preview", command=self.stop_preview).pack(side="left", padx=5)

        # Notebook
        self.nb = ttk.Notebook(root)
        self.tab1 = ttk.Frame(self.nb)
        self.tab2 = ttk.Frame(self.nb)
        self.nb.add(self.tab1,text="Intrinsics + ROI")
        self.nb.add(self.tab2,text="Pose Estimation")
        self.nb.pack(fill="both",expand=True)

        # --- Tab1 UI ---
        ttk.Label(self.tab1, text="Select Model:").pack(side="top", anchor="w")
        self.model_dropdown = ttk.Combobox(self.tab1, textvariable=self.model_choice,
                                           values=list_ply_models(MODELS_DIR), state="readonly")
        self.model_dropdown.pack(side="top", fill="x")
        self.model_dropdown.bind("<<ComboboxSelected>>", self.update_model)

        btn_frame = ttk.Frame(self.tab1); btn_frame.pack(side="top", pady=5)
        ttk.Button(btn_frame,text="Capture Left",command=self.capture_left).pack(side="left", padx=5)
        ttk.Button(btn_frame,text="Capture Right",command=self.capture_right).pack(side="left", padx=5)
        ttk.Button(btn_frame,text="Send to Intrinsics API",command=self.intrinsics).pack(side="left", padx=5)

        img_frame = ttk.Frame(self.tab1); img_frame.pack(side="top", pady=5)
        self.left_label = ttk.Label(img_frame); self.left_label.pack(side="left", padx=5)
        self.right_label = ttk.Label(img_frame); self.right_label.pack(side="left", padx=5)

        roi_frame = ttk.Frame(self.tab1); roi_frame.pack(side="top", pady=5)
        self.canvas = tk.Canvas(roi_frame, width=640, height=480, bg="black")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)

        ttk.Button(self.tab1,text="Reset ROI",command=self.reset_roi).pack(side="top", pady=5)
        self.roi_label = ttk.Label(self.tab1); self.roi_label.pack(side="left", padx=5)
        self.bounding_label = ttk.Label(self.tab1); self.bounding_label.pack(side="left", padx=5)

        # --- Tab2 UI ---
        ctrl = ttk.Frame(self.tab2); ctrl.pack(side="top", fill="x")
        ttk.Label(ctrl,text="Rate (Hz):").pack(side="left")
        self.rate_var = tk.DoubleVar(value=1.0)
        ttk.Entry(ctrl,textvariable=self.rate_var,width=5).pack(side="left")
        ttk.Button(ctrl,text="Start Estimation",command=self.start).pack(side="left", padx=5)
        ttk.Button(ctrl,text="Stop Estimation",command=self.stop).pack(side="left", padx=5)

        view = ttk.Frame(self.tab2); view.pack(side="top", fill="both", expand=True)
        self.live_label = ttk.Label(view); self.live_label.pack(side="left", padx=5)
        self.depth_label = ttk.Label(view); self.depth_label.pack(side="left", padx=5)
        self.overlay_label = ttk.Label(view); self.overlay_label.pack(side="left", padx=5)
        self.masked_label = ttk.Label(view); self.masked_label.pack(side="left", padx=5)

        self.log_text = tk.Text(self.tab2, height=8, state="disabled", bg="black", fg="lime")
        self.log_text.pack(side="bottom", fill="both", expand=True)

    # --- Logging ---
    def log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # --- Preview control ---
    def start_preview(self):
        if not self.preview_running:
            self.preview_running = True
            self.update_preview()
            self.log("Preview started")

    def stop_preview(self):
        self.preview_running = False
        self.log("Preview stopped")

    def update_preview(self):
        if not self.preview_running: return
        frame = get_camera_frame()
        if frame:
            self.tk_preview = tk_image(frame, size=(320,240))
            self.preview_label.configure(image=self.tk_preview)
        self.root.after(100, self.update_preview)

    # --- Model dropdown ---
    def update_model(self, evt=None):
        path = os.path.join(MODELS_DIR, self.model_choice.get())
        self.V, self.edges = load_ply_edges(path)
        self.log(f"Loaded model {self.model_choice.get()}")

    # --- ROI ---
    def on_click(self, event):
        if self.roi_center is None:
            self.roi_center = (event.x, event.y)
        else:
            dx = event.x - self.roi_center[0]
            dy = event.y - self.roi_center[1]
            self.roi_radius = int((dx**2 + dy**2)**0.5)
            if self.roi_preview:
                self.canvas.delete(self.roi_preview)
            self.canvas.create_oval(
                self.roi_center[0]-self.roi_radius,
                self.roi_center[1]-self.roi_radius,
                self.roi_center[0]+self.roi_radius,
                self.roi_center[1]+self.roi_radius,
                outline="green", width=2
            )

    def on_motion(self, event):
        if self.roi_center and self.roi_radius is None:
            dx = event.x - self.roi_center[0]
            dy = event.y - self.roi_center[1]
            r = int((dx**2 + dy**2)**0.5)
            if self.roi_preview:
                self.canvas.delete(self.roi_preview)
            self.roi_preview = self.canvas.create_oval(
                self.roi_center[0]-r,
                self.roi_center[1]-r,
                self.roi_center[0]+r,
                self.roi_center[1]+r,
                outline="red", dash=(2,2)
            )

    def reset_roi(self):
        self.roi_center=None; self.roi_radius=None
        if self.roi_preview: self.canvas.delete(self.roi_preview)
        self.roi_preview=None
        self.log("ROI reset")

    # --- Capture / Intrinsics ---
    def capture_left(self):
        self.left_img = get_camera_frame()
        if self.left_img:
            self.left_label.img = tk_image(self.left_img)
            self.left_label.configure(image=self.left_label.img)

    def capture_right(self):
        self.right_img = get_camera_frame()
        if self.right_img:
            self.right_label.img = tk_image(self.right_img)
            self.right_label.configure(image=self.right_label.img)

    def intrinsics(self):
        if not (self.left_img and self.right_img): return
        payload={"left":pil_to_b64(self.left_img),"right":pil_to_b64(self.right_img)}
        r = requests.post(f"{API}/intrinsics",json=payload).json()
        self.K = r["camera_matrix"]
        self.log("Got intrinsics")

    # --- Workers ---
    def mask_worker(self):
        while self.running:
            try:
                snap = self.snap_queue.get(timeout=1)
                if not (self.left_img and self.roi_center and self.roi_radius):
                    continue
                payload={"left":pil_to_b64(self.left_img),
                         "roi_center":list(self.roi_center),
                         "roi_radius":self.roi_radius,
                         "snapshot":pil_to_b64(snap)}
                r2=requests.post(f"{API}/bounding_rect",json=payload).json()
                mask_img = b64_to_pil(r2["mask"])
                self.mask_queue.put((snap, mask_img))
            except queue.Empty:
                continue
            except Exception as e:
                self.log(f"Mask worker error: {e}")

    def pose_worker(self):
        while self.running:
            try:
                snap, mask_img = self.mask_queue.get(timeout=1)
                # Depth
                r=requests.post(f"{API}/depth",json={"rgb":pil_to_b64(snap)},timeout=20).json()
                depth=b64_to_pil(r["depth"])
                # Pose
                pose_req={"camera_matrix":self.K,
                          "images":[{"filename":"snap","rgb":pil_to_b64(snap),"depth":pil_to_b64(depth,fmt="PNG")}],
                          "mesh":"","mask":pil_to_b64(mask_img,fmt="PNG"),"depthscale":0.001}
                resp=requests.post(f"{API}/pose",json=pose_req,timeout=20).json()
                self.pose_queue.put((snap, mask_img, depth, resp))
            except queue.Empty:
                continue
            except Exception as e:
                self.log(f"Pose worker error: {e}")

    # --- Start/Stop estimation ---
    def start(self):
        if self.running: return
        self.running=True
        self.rate=self.rate_var.get()
        threading.Thread(target=self.mask_worker, daemon=True).start()
        threading.Thread(target=self.pose_worker, daemon=True).start()
        threading.Thread(target=self.loop,daemon=True).start()
        self.log("Estimation started")

    def stop(self):
        self.running=False
        self.log("Estimation stopped")

    def loop(self):
        while self.running:
            snap=get_camera_frame()
            if snap:
                try:
                    self.snap_queue.put_nowait(snap)
                except queue.Full:
                    pass
            # consume pose results if ready
            try:
                snap, mask_img, depth, resp = self.pose_queue.get_nowait()
                self.update_pose_ui(snap, mask_img, depth, resp)
            except queue.Empty:
                pass
            time.sleep(1.0/self.rate)

    def update_pose_ui(self, snap, mask_img, depth, resp):
        # masked snapshot
        snap_arr = np.array(snap)
        mask_arr = np.array(mask_img.convert("L"))
        snap_masked = cv2.bitwise_and(snap_arr, snap_arr, mask=mask_arr)
        snap_masked_pil = Image.fromarray(snap_masked)

        # overlay
        bgr=cv2.cvtColor(np.array(snap),cv2.COLOR_RGB2BGR)
        for idx,T in enumerate(resp["transformation_matrix"]):
            color=(0,255-100*idx,100*idx)
            bgr=overlay_mesh_wireframe(bgr,self.V,self.edges,self.K,np.array(T),color)
        rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))

        # update GUI
        self.live_label.img=tk_image(snap); self.live_label.configure(image=self.live_label.img)
        self.depth_label.img=tk_image(depth); self.depth_label.configure(image=self.depth_label.img)
        self.overlay_label.img=tk_image(rgb); self.overlay_label.configure(image=self.overlay_label.img)
        self.masked_label.img=tk_image(snap_masked_pil); self.masked_label.configure(image=self.masked_label.img)
        self.log("Updated pose view")

# ---------- Main ----------
if __name__=="__main__":
    root=tk.Tk()
    app=PoseApp(root)
    root.mainloop()

