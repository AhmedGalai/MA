import tkinter as tk
from tkinter import ttk
import cv2, requests, base64, io, time, threading, os
import numpy as np
from PIL import Image, ImageTk
import trimesh

API = "http://localhost:8000"
MODELS_DIR = "models"

# ---------- Helpers ----------
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
# class PoseApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Tk Pose Estimation Client")

#         # State
#         self.left_img = None
#         self.right_img = None
#         self.K = [[700,0,320],[0,700,240],[0,0,1]]
#         self.roi_center = None
#         self.roi_radius = None
#         self.roi_preview = None
#         self.V, self.edges = load_ply_edges(os.path.join(MODELS_DIR, os.listdir(MODELS_DIR)[0]))
#         self.model_path = None
#         self.running = False
#         self.rate = 1.0
#         self.tk_left = None

#         # Notebook
#         self.nb = ttk.Notebook(root)
#         self.tab1 = ttk.Frame(self.nb)
#         self.tab2 = ttk.Frame(self.nb)
#         self.nb.add(self.tab1,text="Intrinsics + ROI")
#         self.nb.add(self.tab2,text="Pose Estimation")
#         self.nb.pack(fill="both",expand=True)

#         # --- Tab1 UI
#         ttk.Button(self.tab1,text="Capture Left",command=self.capture_left).grid(row=0,column=0)
#         ttk.Button(self.tab1,text="Capture Right",command=self.capture_right).grid(row=0,column=1)
#         ttk.Button(self.tab1,text="Send to Intrinsics API",command=self.intrinsics).grid(row=0,column=2)

#         self.canvas = tk.Canvas(self.tab1, width=640, height=480)
#         self.canvas.grid(row=1, column=0, columnspan=3)
#         self.canvas.bind("<Button-1>", self.on_click)
#         self.canvas.bind("<Motion>", self.on_motion)

#         self.right_label = ttk.Label(self.tab1); self.right_label.grid(row=2,column=0)
#         self.debug_label = ttk.Label(self.tab1); self.debug_label.grid(row=2,column=1)

#         ttk.Button(self.tab1,text="Set ROI",command=self.set_roi).grid(row=3,column=0)
#         self.roi_label = ttk.Label(self.tab1); self.roi_label.grid(row=3,column=1)
#         self.bounding_label = ttk.Label(self.tab1); self.bounding_label.grid(row=3,column=2)

#         # --- Tab2 UI
#         ttk.Label(self.tab2,text="Rate (Hz):").grid(row=0,column=0)
#         self.rate_var = tk.DoubleVar(value=1.0)
#         ttk.Entry(self.tab2,textvariable=self.rate_var,width=5).grid(row=0,column=1)
#         ttk.Button(self.tab2,text="Start",command=self.start).grid(row=0,column=2)
#         ttk.Button(self.tab2,text="Stop",command=self.stop).grid(row=0,column=3)

#         self.rgb_label = ttk.Label(self.tab2); self.rgb_label.grid(row=1,column=0)
#         self.depth_label = ttk.Label(self.tab2); self.depth_label.grid(row=1,column=1)
#         self.overlay_label = ttk.Label(self.tab2); self.overlay_label.grid(row=1,column=2)

#     # --- ROI mouse handlers ---
#     def on_click(self, event):
#         if self.roi_center is None:
#             self.roi_center = (event.x, event.y)
#         else:
#             dx = event.x - self.roi_center[0]
#             dy = event.y - self.roi_center[1]
#             self.roi_radius = int((dx**2 + dy**2)**0.5)
#             if self.roi_preview:
#                 self.canvas.delete(self.roi_preview)
#             self.canvas.create_oval(
#                 self.roi_center[0]-self.roi_radius,
#                 self.roi_center[1]-self.roi_radius,
#                 self.roi_center[0]+self.roi_radius,
#                 self.roi_center[1]+self.roi_radius,
#                 outline="green", width=2
#             )

#     def on_motion(self, event):
#         if self.roi_center and self.roi_radius is None:
#             dx = event.x - self.roi_center[0]
#             dy = event.y - self.roi_center[1]
#             r = int((dx**2 + dy**2)**0.5)
#             if self.roi_preview:
#                 self.canvas.delete(self.roi_preview)
#             self.roi_preview = self.canvas.create_oval(
#                 self.roi_center[0]-r,
#                 self.roi_center[1]-r,
#                 self.roi_center[0]+r,
#                 self.roi_center[1]+r,
#                 outline="red", dash=(2,2)
#             )

#     # --- Actions ---
#     def capture_left(self):
#         self.left_img = get_camera_frame()
#         if self.left_img:
#             self.tk_left = ImageTk.PhotoImage(self.left_img.resize((640,480)))
#             self.canvas.create_image(0,0, anchor="nw", image=self.tk_left)

#     def capture_right(self):
#         self.right_img = get_camera_frame()
#         if self.right_img:
#             self.right_label.img = tk_image(self.right_img)
#             self.right_label.configure(image=self.right_label.img)

#     def intrinsics(self):
#         if not (self.left_img and self.right_img): return
#         payload={"left":pil_to_b64(self.left_img),"right":pil_to_b64(self.right_img)}
#         r = requests.post(f"{API}/intrinsics",json=payload).json()
#         self.K = r["camera_matrix"]
#         dbg = b64_to_pil(r["debug_image"])
#         self.debug_label.img = tk_image(dbg)
#         self.debug_label.configure(image=self.debug_label.img)
#         # rectify
#         r2=requests.post(f"{API}/rectify",json={"left":payload["left"],"right":payload["right"],"camera_matrix":self.K}).json()
#         rect = b64_to_pil(r2["rectified"])
#         self.right_label.img = tk_image(rect)
#         self.right_label.configure(image=self.right_label.img)

#     def set_roi(self):
#         if not (self.left_img and self.roi_center and self.roi_radius):
#             return
#         arr=np.array(self.left_img)
#         mask=np.zeros(arr.shape[:2],dtype=np.uint8)
#         cv2.circle(mask,self.roi_center,self.roi_radius,255,-1)
#         masked=cv2.bitwise_and(arr,arr,mask=mask)
#         masked_pil=Image.fromarray(masked)
#         self.roi_label.img=tk_image(masked_pil)
#         self.roi_label.configure(image=self.roi_label.img)
#         # bounding rect
#         payload={"left":pil_to_b64(self.left_img),
#                  "roi_center":list(self.roi_center),
#                  "roi_radius":self.roi_radius,
#                  "snapshot":pil_to_b64(self.left_img)}
#         r=requests.post(f"{API}/bounding_rect",json=payload).json()
#         bounding=b64_to_pil(r["mask"])
#         self.bounding_label.img=tk_image(bounding)
#         self.bounding_label.configure(image=self.bounding_label.img)

#     def start(self):
#         self.running=True
#         self.rate=self.rate_var.get()
#         threading.Thread(target=self.loop,daemon=True).start()

#     def stop(self):
#         self.running=False

#     def loop(self):
#         while self.running:
#             snap=get_camera_frame()
#             if snap and self.left_img and self.roi_center and self.roi_radius:
#                 # Depth
#                 r=requests.post(f"{API}/depth",json={"rgb":pil_to_b64(snap)},timeout=20).json()
#                 depth=b64_to_pil(r["depth"])
#                 # Bounding rect
#                 payload={"left":pil_to_b64(self.left_img),
#                          "roi_center":list(self.roi_center),
#                          "roi_radius":self.roi_radius,
#                          "snapshot":pil_to_b64(snap)}
#                 r2=requests.post(f"{API}/bounding_rect",json=payload).json()
#                 # Pose
#                 pose_req={"camera_matrix":self.K,
#                           "images":[{"filename":"snap","rgb":pil_to_b64(snap),"depth":pil_to_b64(depth,fmt="PNG")}],
#                           "mesh":"","mask":r2["mask"],"depthscale":0.001}
#                 resp=requests.post(f"{API}/pose",json=pose_req,timeout=20).json()
#                 # Overlay
#                 bgr=cv2.cvtColor(np.array(snap),cv2.COLOR_RGB2BGR)
#                 for idx,T in enumerate(resp["transformation_matrix"]):
#                     color=(0,255-100*idx,100*idx)
#                     bgr=overlay_mesh_wireframe(bgr,self.V,self.edges,self.K,np.array(T),color)
#                 rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
#                 # Update GUI
#                 self.rgb_label.img=tk_image(snap)
#                 self.rgb_label.configure(image=self.rgb_label.img)
#                 self.depth_label.img=tk_image(depth)
#                 self.depth_label.configure(image=self.depth_label.img)
#                 self.overlay_label.img=tk_image(rgb)
#                 self.overlay_label.configure(image=self.overlay_label.img)
#             time.sleep(1.0/self.rate)

# ... (imports and helpers unchanged)

class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tk Pose Estimation Client")

        # state vars ...
        self.left_img = None
        self.right_img = None
        self.K = [[700,0,320],[0,700,240],[0,0,1]]
        self.roi_center = None
        self.roi_radius = None
        self.roi_preview = None
        self.V, self.edges = load_ply_edges(os.path.join(MODELS_DIR, os.listdir(MODELS_DIR)[0]))
        self.running = False
        self.rate = 1.0
        self.tk_left = None

        # Notebook
        self.nb = ttk.Notebook(root)
        self.tab1 = ttk.Frame(self.nb)
        self.tab2 = ttk.Frame(self.nb)
        self.nb.add(self.tab1,text="Intrinsics + ROI")
        self.nb.add(self.tab2,text="Pose Estimation")
        self.nb.pack(fill="both",expand=True)

        # ----- Tab 1 -----
        top = ttk.Frame(self.tab1); top.pack(side="top", fill="x")
        ttk.Button(top,text="Capture Left",command=self.capture_left).pack(side="left", padx=5)
        ttk.Button(top,text="Capture Right",command=self.capture_right).pack(side="left", padx=5)
        ttk.Button(top,text="Send to Intrinsics API",command=self.intrinsics).pack(side="left", padx=5)

        mid = ttk.Frame(self.tab1); mid.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(mid, width=640, height=480, bg="black")
        self.canvas.pack(side="left", padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)

        self.right_label = ttk.Label(mid); self.right_label.pack(side="left", padx=5)
        self.debug_label = ttk.Label(mid); self.debug_label.pack(side="left", padx=5)

        bottom = ttk.Frame(self.tab1); bottom.pack(side="bottom", fill="x")
        ttk.Button(bottom,text="Set ROI",command=self.set_roi).pack(side="left", padx=5)
        self.roi_label = ttk.Label(bottom); self.roi_label.pack(side="left", padx=5)
        self.bounding_label = ttk.Label(bottom); self.bounding_label.pack(side="left", padx=5)

        # ----- Tab 2 -----
        ctrl = ttk.Frame(self.tab2); ctrl.pack(side="top", fill="x")
        ttk.Label(ctrl,text="Rate (Hz):").pack(side="left")
        self.rate_var = tk.DoubleVar(value=1.0)
        ttk.Entry(ctrl,textvariable=self.rate_var,width=5).pack(side="left")
        ttk.Button(ctrl,text="Start",command=self.start).pack(side="left", padx=5)
        ttk.Button(ctrl,text="Stop",command=self.stop).pack(side="left", padx=5)

        mid2 = ttk.Frame(self.tab2); mid2.pack(side="top", fill="both", expand=True)
        self.live_label = ttk.Label(mid2); self.live_label.pack(side="left", padx=5)
        self.depth_label = ttk.Label(mid2); self.depth_label.pack(side="left", padx=5)
        self.overlay_label = ttk.Label(mid2); self.overlay_label.pack(side="left", padx=5)
        self.masked_label = ttk.Label(mid2); self.masked_label.pack(side="left", padx=5)  # NEW


        # Logging panel
        log_frame = ttk.Frame(self.tab2); log_frame.pack(side="bottom", fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=8, state="disabled", bg="black", fg="lime")
        self.log_text.pack(fill="both", expand=True)

    def log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # --- rest of your ROI handlers, capture, intrinsics, set_roi same as before ---

    # --- ROI mouse handlers ---
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


        # --- Actions ---
    def capture_left(self):
        self.left_img = get_camera_frame()
        if self.left_img:
            self.tk_left = ImageTk.PhotoImage(self.left_img.resize((640,480)))
            self.canvas.create_image(0,0, anchor="nw", image=self.tk_left)

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
        dbg = b64_to_pil(r["debug_image"])
        self.debug_label.img = tk_image(dbg)
        self.debug_label.configure(image=self.debug_label.img)
        # rectify
        r2=requests.post(f"{API}/rectify",json={"left":payload["left"],"right":payload["right"],"camera_matrix":self.K}).json()
        rect = b64_to_pil(r2["rectified"])
        self.right_label.img = tk_image(rect)
        self.right_label.configure(image=self.right_label.img)

    def set_roi(self):
        if not (self.left_img and self.roi_center and self.roi_radius):
            return
        arr=np.array(self.left_img)
        mask=np.zeros(arr.shape[:2],dtype=np.uint8)
        cv2.circle(mask,self.roi_center,self.roi_radius,255,-1)
        masked=cv2.bitwise_and(arr,arr,mask=mask)
        masked_pil=Image.fromarray(masked)
        self.roi_label.img=tk_image(masked_pil)
        self.roi_label.configure(image=self.roi_label.img)
        # bounding rect
        payload={"left":pil_to_b64(self.left_img),
                 "roi_center":list(self.roi_center),
                 "roi_radius":self.roi_radius,
                 "snapshot":pil_to_b64(self.left_img)}
        r=requests.post(f"{API}/bounding_rect",json=payload).json()
        bounding=b64_to_pil(r["mask"])
        self.bounding_label.img=tk_image(bounding)
        self.bounding_label.configure(image=self.bounding_label.img)

    def start(self):
        self.running=True
        self.rate=self.rate_var.get()
        threading.Thread(target=self.loop,daemon=True).start()

    def stop(self):
        self.running=False

    def loop(self):
        while self.running:
            try:
                snap=get_camera_frame()
                if snap and self.left_img and self.roi_center and self.roi_radius:
                    self.log("Captured frame")

                    # Depth
                    r=requests.post(f"{API}/depth",json={"rgb":pil_to_b64(snap)},timeout=20).json()
                    depth=b64_to_pil(r["depth"])
                    self.log("Depth API ok")

                    # Bounding rect
                    # Bounding rect
                    payload={"left":pil_to_b64(self.left_img),
                             "roi_center":list(self.roi_center),
                             "roi_radius":self.roi_radius,
                             "snapshot":pil_to_b64(snap)}
                    r2=requests.post(f"{API}/bounding_rect",json=payload).json()
                    mask_img = b64_to_pil(r2["mask"])
                    self.log("Bounding rect ok")

                    # Create masked snapshot
                    snap_arr = np.array(snap)
                    mask_arr = np.array(mask_img.convert("L"))
                    snap_masked = cv2.bitwise_and(snap_arr, snap_arr, mask=mask_arr)
                    snap_masked_pil = Image.fromarray(snap_masked)


                    # Pose
                    pose_req={"camera_matrix":self.K,
                              "images":[{"filename":"snap","rgb":pil_to_b64(snap),"depth":pil_to_b64(depth,fmt="PNG")}],
                              "mesh":"","mask":r2["mask"],"depthscale":0.001}
                    resp=requests.post(f"{API}/pose",json=pose_req,timeout=20).json()
                    self.log("Pose API ok")

                    # Overlay
                    bgr=cv2.cvtColor(np.array(snap),cv2.COLOR_RGB2BGR)
                    for idx,T in enumerate(resp["transformation_matrix"]):
                        color=(0,255-100*idx,100*idx)
                        bgr=overlay_mesh_wireframe(bgr,self.V,self.edges,self.K,np.array(T),color)
                    rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))

                    # Update GUI
                    self.live_label.img=tk_image(snap)
                    self.live_label.configure(image=self.live_label.img)

                    self.depth_label.img=tk_image(depth)
                    self.depth_label.configure(image=self.depth_label.img)

                    self.overlay_label.img=tk_image(rgb)
                    self.overlay_label.configure(image=self.overlay_label.img)

                    self.masked_label.img=tk_image(snap_masked_pil)   # NEW
                    self.masked_label.configure(image=self.masked_label.img)


            except Exception as e:
                self.log(f"ERROR: {e}")
            time.sleep(1.0/self.rate)




# ---------- Main ----------
if __name__=="__main__":
    root=tk.Tk()
    app=PoseApp(root)
    root.mainloop()
