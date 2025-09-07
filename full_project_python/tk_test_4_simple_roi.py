import tkinter as tk
from tkinter import ttk
import cv2, requests, base64, io, time, threading, os, queue
import numpy as np
from PIL import Image, ImageTk
import trimesh

from tkinter import colorchooser  # add this


API = "http://localhost:8000"
# API = "http://192.168.178.134:8000"
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

    # Unique edges from faces
    edges = set()
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            e = (a, b) if a < b else (b, a)
            edges.add(e)
    E = np.array(list(edges), dtype=np.int32)

    # Center of mass (uniform density). Trimesh handles watertight vs. non-watertight.
    try:
        com = np.asarray(mesh.center_mass, dtype=np.float64)
    except Exception:
        # Fallback: area centroid
        com = mesh.centroid.astype(np.float64)

    # Pick a reasonable 3D axis length (fraction of model size)
    ext = float(np.max(mesh.extents)) if mesh.extents is not None else float(np.linalg.norm(V.max(0) - V.min(0)))
    axis_len = 0.25 * (ext if ext > 0 else 1.0)

    return V, E, com, axis_len


def overlay_wireframe_and_com(bgr, V, edges, K, T, com_obj, axis_len,
                              edge_color=(0,255,0), edge_thick=2):
    img = bgr.copy()

    K = np.array(K, dtype=np.float64)
    T = np.array(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)

    rvec, _ = cv2.Rodrigues(R)

    # Project mesh vertices
    pts2d, _ = cv2.projectPoints(V.astype(np.float64), rvec, t, K, None)
    pts2d = pts2d.reshape(-1, 2)

    # Draw wireframe
    for i, j in edges:
        x1, y1 = pts2d[i]; x2, y2 = pts2d[j]
        if np.isfinite([x1, y1, x2, y2]).all():
            cv2.line(img,
                     (int(round(x1)), int(round(y1))),
                     (int(round(x2)), int(round(y2))),
                     edge_color, int(edge_thick), cv2.LINE_AA)

    # COM in world: R*com + t
    com_w = (R @ com_obj.reshape(3, 1) + t).reshape(3)
    # Axis endpoints in world
    ax_pts_w = [
        com_w + axis_len * R[:, 0],  # +X
        com_w + axis_len * R[:, 1],  # +Y
        com_w + axis_len * R[:, 2],  # +Z
    ]

    # Project COM and endpoints
    com2d, _ = cv2.projectPoints(com_w.reshape(1, 3), rvec, t, K, None)
    com2d = com2d.reshape(2)
    cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=red, Y=green, Z=blue (BGR)

    for p_w, col in zip(ax_pts_w, cols):
        p2d, _ = cv2.projectPoints(p_w.reshape(1, 3), rvec, t, K, None)
        p2d = p2d.reshape(2)
        if np.isfinite(np.r_[com2d, p2d]).all():
            cv2.line(img,
                     (int(round(com2d[0])), int(round(com2d[1]))),
                     (int(round(p2d[0])),   int(round(p2d[1]))),
                     col, max(1, int(edge_thick)), cv2.LINE_AA)
            # small dot at COM
            cv2.circle(img, (int(round(com2d[0])), int(round(com2d[1]))),
                       radius=max(2, int(edge_thick)), color=col, thickness=-1)

    return img


def tk_image(pil, size=(320,240)):
    return ImageTk.PhotoImage(pil.resize(size))

# ---------- GUI ----------
class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tk Pose Estimation Client (ROI only)")

        # State
        self.left_img = None
        self.right_img = None
        self.K = [[700,0,320],[0,700,240],[0,0,1]]
        self.roi_center = None
        self.roi_radius = None
        self.roi_preview = None
        self.model_choice = tk.StringVar(value=list_ply_models(MODELS_DIR)[0])
        # self.V, self.edges = load_ply_edges(os.path.join(MODELS_DIR, self.model_choice.get()))
        self.V, self.edges, self.com, self.axis_len = load_ply_edges(
            os.path.join(MODELS_DIR, self.model_choice.get())
        )

        self.running = False
        self.preview_running = False
        self.rate = 1.0

        # Queues
        self.snap_queue = queue.Queue(maxsize=5)
        self.pose_queue = queue.Queue(maxsize=5)

        # --- Camera preview at top ---
        preview_frame = ttk.Frame(root)
        preview_frame.pack(side="top", pady=5, fill="x")
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.grid(row=0, column=0, rowspan=2, padx=5)
        ttk.Button(preview_frame, text="Start Preview", command=self.start_preview).grid(row=0, column=1, padx=5)
        ttk.Button(preview_frame, text="Stop Preview", command=self.stop_preview).grid(row=1, column=1, padx=5)

        # --- Notebook ---
        self.nb = ttk.Notebook(root)
        self.tab1 = ttk.Frame(self.nb)
        self.tab2 = ttk.Frame(self.nb)
        self.nb.add(self.tab1,text="Intrinsics + ROI")
        self.nb.add(self.tab2,text="Pose Estimation")
        self.nb.pack(fill="both",expand=True)
        self.edge_hex = "#00ff00"            # default edge color (green)
        self.edge_bgr = (0, 255, 0)          # BGR for OpenCV
        self.boldness_var = tk.IntVar(value=2)

        # load model with COM + axis length
        self.V, self.edges, self.com, self.axis_len = load_ply_edges(
            os.path.join(MODELS_DIR, self.model_choice.get())
        )


        # --- Tab1 UI (scrollable) ---
        # Create canvas + scrollbar for scrollable frame
                # --- Tab1 UI (scrollable) ---
        # Canvas + Scrollbar
        tab1_container = ttk.Frame(self.tab1)
        tab1_container.pack(fill="both", expand=True)

        tab1_canvas = tk.Canvas(tab1_container)
        scrollbar = ttk.Scrollbar(tab1_container, orient="vertical", command=tab1_canvas.yview)
        scroll_frame = ttk.Frame(tab1_canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: tab1_canvas.configure(
                scrollregion=tab1_canvas.bbox("all")
            )
        )

        tab1_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        tab1_canvas.configure(yscrollcommand=scrollbar.set)

        tab1_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Now inside scroll_frame, we use grid exclusively
        ttk.Label(scroll_frame, text="Select Model:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_dropdown = ttk.Combobox(scroll_frame, textvariable=self.model_choice,
                                           values=list_ply_models(MODELS_DIR), state="readonly")
        self.model_dropdown.grid(row=0, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.update_model)

        ttk.Button(scroll_frame,text="Capture Left",command=self.capture_left).grid(row=1,column=0,padx=5,pady=5)
        ttk.Button(scroll_frame,text="Capture Right",command=self.capture_right).grid(row=1,column=1,padx=5,pady=5)
        ttk.Button(scroll_frame,text="Send Intrinsics",command=self.intrinsics).grid(row=1,column=2,padx=5,pady=5)

        self.left_label = ttk.Label(scroll_frame)
        self.left_label.grid(row=2, column=0, padx=5, pady=5)
        self.right_label = ttk.Label(scroll_frame)
        self.right_label.grid(row=2, column=1, padx=5, pady=5)

        self.canvas = tk.Canvas(scroll_frame, width=640, height=480, bg="black")
        self.canvas.grid(row=3, column=0, columnspan=3, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas_img_id = None

        ttk.Button(scroll_frame,text="Reset ROI",command=self.reset_roi).grid(row=4, column=0, padx=5, pady=5)
        self.roi_label = ttk.Label(scroll_frame)
        self.roi_label.grid(row=4, column=1, padx=5, pady=5)


        # --- Tab2 UI ---
        ctrl = ttk.Frame(self.tab2)
        ctrl.grid(row=0, column=0, columnspan=4, sticky="ew", pady=5)
        ttk.Label(ctrl,text="Rate (Hz):").pack(side="left")
        self.rate_var = tk.DoubleVar(value=1.0)
        ttk.Entry(ctrl,textvariable=self.rate_var,width=5).pack(side="left", padx=5)
        ttk.Button(ctrl,text="Start Estimation",command=self.start).pack(side="left", padx=5)
        ttk.Button(ctrl,text="Stop Estimation",command=self.stop).pack(side="left", padx=5)

        self.live_label = ttk.Label(self.tab2)
        self.live_label.grid(row=1, column=0, padx=5, pady=5)
        self.depth_label = ttk.Label(self.tab2)
        self.depth_label.grid(row=1, column=1, padx=5, pady=5)
        self.overlay_label = ttk.Label(self.tab2)
        self.overlay_label.grid(row=1, column=2, padx=5, pady=5)
        self.masked_label = ttk.Label(self.tab2)
        self.masked_label.grid(row=1, column=3, padx=5, pady=5)

        self.log_text = tk.Text(self.tab2, height=8, state="disabled", bg="black", fg="lime")
        self.log_text.grid(row=2, column=0, columnspan=4, sticky="nsew", padx=5, pady=5)
        # --- color picker ---
        def _pick_color():
            rgb, hexcol = colorchooser.askcolor(initialcolor=self.edge_hex, parent=self.tab2)
            if hexcol:
                self.edge_hex = hexcol
                r, g, b = map(int, rgb)
                self.edge_bgr = (b, g, r)  # to BGR for OpenCV
                # update swatch
                self.color_swatch.configure(background=self.edge_hex)

        ttk.Button(ctrl, text="Edge Color…", command=_pick_color).pack(side="left", padx=8)

        # Swatch
        self.color_swatch = tk.Label(ctrl, text="   ", background=self.edge_hex, relief="solid", width=3)
        self.color_swatch.pack(side="left", padx=4)

        # --- boldness slider ---
        ttk.Label(ctrl, text="Boldness:").pack(side="left", padx=(12, 2))
        ttk.Scale(ctrl, from_=1, to=8, orient="horizontal",
                  variable=self.boldness_var, length=120).pack(side="left")


    # --- Logging ---
    def log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # --- Preview ---
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
            # Show masked left
            if self.left_img:
                arr=np.array(self.left_img)
                mask=np.zeros(arr.shape[:2],dtype=np.uint8)
                cv2.circle(mask,self.roi_center,self.roi_radius,255,-1)
                masked=cv2.bitwise_and(arr,arr,mask=mask)
                self.roi_label.img=tk_image(Image.fromarray(masked))
                self.roi_label.configure(image=self.roi_label.img)

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
        self.roi_label.configure(image="")
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
        if not (self.left_img and self.right_img): 
            self.log("Capture both left and right first")
            return
        payload={"left":pil_to_b64(self.left_img),"right":pil_to_b64(self.right_img)}
        r = requests.post(f"{API}/intrinsics",json=payload).json()
        self.K = r["camera_matrix"]
        self.log("Got intrinsics")

        # Show left image in ROI canvas
        if self.left_img:
            resized = self.left_img.resize((640,480))
            self.tk_left_canvas = ImageTk.PhotoImage(resized)
            if self.canvas_img_id:
                self.canvas.delete(self.canvas_img_id)
            self.canvas_img_id = self.canvas.create_image(0,0,anchor="nw",image=self.tk_left_canvas)

    # --- Model dropdown ---
    def update_model(self, evt=None):
        path = os.path.join(MODELS_DIR, self.model_choice.get())
        # self.V, self.edges = load_ply_edges(path)
        self.V, self.edges, self.com, self.axis_len = load_ply_edges(path)

        self.log(f"Loaded model {self.model_choice.get()}")


    # --- Workers ---
    def pose_worker(self):
        while self.running:
            try:
                snap = self.snap_queue.get(timeout=1)
                if not (self.left_img and self.roi_center and self.roi_radius):
                    continue
                # ROI mask
                arr=np.array(snap)
                mask=np.zeros(arr.shape[:2],dtype=np.uint8)
                cv2.circle(mask,self.roi_center,self.roi_radius,255,-1)
                snap_masked=cv2.bitwise_and(arr,arr,mask=mask)
                snap_masked_pil=Image.fromarray(snap_masked)

                # Depth
                # r=requests.post(f"{API}/depth",json={"rgb":pil_to_b64(snap)},timeout=20).json()
                r=requests.post(f"{API}/depth",json={"rgb":pil_to_b64(snap)},timeout=120).json()
                depth=b64_to_pil(r["depth"])

                # Pose
                pose_req={"camera_matrix":self.K,
                          "images":[{"filename":"snap","rgb":pil_to_b64(snap),"depth":pil_to_b64(depth,fmt="PNG")}],
                          "mesh":"","mask":pil_to_b64(snap_masked_pil,fmt="PNG"),"depthscale":0.001}
                # resp=requests.post(f"{API}/pose",json=pose_req,timeout=20).json()
                resp=requests.post(f"{API}/pose",json=pose_req,timeout=120).json()

                self.pose_queue.put((snap, depth, snap_masked_pil, resp))
            except queue.Empty:
                continue
            except Exception as e:
                self.log(f"Pose worker error: {e}")

    # --- Start/Stop estimation ---
    def start(self):
        if self.running: return
        self.running=True
        self.rate=self.rate_var.get()
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
                try: self.snap_queue.put_nowait(snap)
                except queue.Full: pass
            try:
                snap, depth, masked, resp = self.pose_queue.get_nowait()
                self.update_pose_ui(snap, depth, masked, resp)
            except queue.Empty: pass
            time.sleep(1.0/self.rate)

    def update_pose_ui(self, snap, depth, masked, resp):
        bgr = cv2.cvtColor(np.array(snap), cv2.COLOR_RGB2BGR)
        thickness = int(self.boldness_var.get())

        for idx, T in enumerate(resp["transformation_matrix"]):
            # bgr = overlay_mesh_wireframe(bgr, self.V, self.edges, self.K, np.array(T), color)
            bgr = overlay_wireframe_and_com(
                bgr, self.V, self.edges, self.K, np.array(T),
                self.com, self.axis_len,
                edge_color=self.edge_bgr, edge_thick=int(self.boldness_var.get())
            )


        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        self.live_label.img = tk_image(snap);   self.live_label.configure(image=self.live_label.img)
        self.depth_label.img = tk_image(depth); self.depth_label.configure(image=self.depth_label.img)
        self.overlay_label.img = tk_image(rgb); self.overlay_label.configure(image=self.overlay_label.img)
        self.masked_label.img = tk_image(masked); self.masked_label.configure(image=self.masked_label.img)
        self.log("Updated pose view")


# ---------- Main ----------
if __name__=="__main__":
    root=tk.Tk()
    app=PoseApp(root)
    root.mainloop()

