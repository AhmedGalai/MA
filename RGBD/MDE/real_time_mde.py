# Requirements:
#   pip install transformers torch pillow opencv-python
#   (Optional for better performance with CUDA: install the right torch build)

import sys, threading, time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from transformers import pipeline, infer_device


# ---------------------- Model/Pipeline ----------------------

def build_depth_pipeline():
    """
    Create the Hugging Face depth-estimation pipeline using Depth-Anything V2.
    Uses CPU or GPU automatically via infer_device().
    """
    device = infer_device()  # 'cpu' or 'cuda' / device index
    checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
    print(f"[INFO] Loading model '{checkpoint}' on device: {device}", flush=True)
    return pipeline("depth-estimation", model=checkpoint, device=device)


# ---------------------- Tkinter App ------------------------

class DepthApp(tk.Tk):
    def __init__(self, pipe):
        super().__init__()
        self.title("Live RGB + Depth (snapshots)")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- Controls ---
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Label(ctrl, text="Camera index:").pack(side=tk.LEFT)
        self.cam_var = tk.IntVar(value=0)
        ttk.Entry(ctrl, textvariable=self.cam_var, width=5).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(ctrl, text="Sampling rate (Hz):").pack(side=tk.LEFT)
        self.rate_var = tk.DoubleVar(value=0.5)  # snapshots per second
        ttk.Entry(ctrl, textvariable=self.rate_var, width=6).pack(side=tk.LEFT, padx=(2, 10))

        self.start_btn = ttk.Button(ctrl, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = ttk.Button(ctrl, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        self.status = tk.StringVar(value="idle")
        ttk.Label(ctrl, textvariable=self.status).pack(side=tk.LEFT, padx=12)

        # --- Views ---
        img_frame = ttk.Frame(self)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        ttk.Label(img_frame, text="RGB (live)").grid(row=0, column=0)
        ttk.Label(img_frame, text="Depth (snapshots)").grid(row=0, column=1)

        self.rgb_panel = ttk.Label(img_frame)
        self.rgb_panel.grid(row=1, column=0, padx=5)

        self.depth_panel = ttk.Label(img_frame)
        self.depth_panel.grid(row=1, column=1, padx=5)

        # --- State ---
        self.pipe = pipe
        self.cap = None
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_bgr = None           # last live frame (BGR)
        self.latest_depth_image = None   # last depth color snapshot (PIL.Image)
        self.rgb_photo = None            # Tk photo refs to avoid GC
        self.depth_photo = None

        # UI refresher (runs on main thread)
        self.after(33, self.ui_tick)  # ~30 FPS

    # ---------- Control flow ----------
    def start(self):
        if self.cap is not None:
            return
        idx = int(self.cam_var.get())

        # Try DirectShow (good on Windows). Fallback to default if needed.
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            self.status.set(f"cannot open camera {idx}")
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            return

        # Lower resolution for speed (adjust if you want)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.stop_event.clear()

        self.preview_thread = threading.Thread(target=self.preview_loop, daemon=True)
        self.depth_thread = threading.Thread(target=self.depth_loop, daemon=True)
        self.preview_thread.start()
        self.depth_thread.start()

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status.set("running")

    def stop(self):
        self.stop_event.set()
        for t in [getattr(self, 'preview_thread', None), getattr(self, 'depth_thread', None)]:
            if t and t.is_alive():
                t.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status.set("stopped")

    def on_close(self):
        self.stop()
        self.destroy()

    # ---------- Worker threads ----------
    def preview_loop(self):
        """Continuously grab frames and keep the latest one for display."""
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.status.set("camera read failed")
                break
            with self.frame_lock:
                self.latest_bgr = frame
            time.sleep(0.01)  # ~100 Hz capture loop

    def depth_loop(self):
        """Sample at user-defined Hz; run depth on the latest frame."""
        while not self.stop_event.is_set():
            # Clamp rate to [0.1, 5] Hz
            try:
                rate = float(self.rate_var.get())
            except Exception:
                rate = 0.5
            rate = max(0.1, min(rate, 5.0))
            time.sleep(1.0 / rate)

            with self.frame_lock:
                snap = None if self.latest_bgr is None else self.latest_bgr.copy()
            if snap is None:
                continue

            try:
                # Optional downscale for speed (keep max side ~512 px)
                h, w = snap.shape[:2]
                scale = 512.0 / max(h, w)
                if scale < 1.0:
                    snap = cv2.resize(snap, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                # BGR -> PIL RGB
                rgb = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                # Run model
                pred = self.pipe(pil_img)
                depth_pil = pred['depth']  # PIL 'L' image

                # Normalize to 0..255 and colorize for readability
                d = np.array(depth_pil, dtype=np.float32)
                if np.isnan(d).any():
                    d = np.nan_to_num(d, nan=0.0)
                dmin, dmax = float(d.min()), float(d.max())
                if dmax - dmin < 1e-6:
                    d8 = np.zeros_like(d, dtype=np.uint8)
                else:
                    d8 = ((d - dmin) * (255.0 / (dmax - dmin))).astype(np.uint8)

                dcolor_bgr = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
                dcolor_rgb = cv2.cvtColor(dcolor_bgr, cv2.COLOR_BGR2RGB)
                self.latest_depth_image = Image.fromarray(dcolor_rgb)

            except Exception as e:
                self.status.set(f"depth error: {e}")

    # ---------- UI refresh (main thread) ----------
    def ui_tick(self):
        # Update RGB view
        with self.frame_lock:
            frame = None if self.latest_bgr is None else self.latest_bgr.copy()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            disp = Image.fromarray(rgb)
            # Fit width ~640 while preserving aspect
            w_target = 640
            h_target = int(w_target * disp.height / max(1, disp.width))
            disp = disp.resize((w_target, h_target), Image.BILINEAR)
            self.rgb_photo = ImageTk.PhotoImage(disp)
            self.rgb_panel.configure(image=self.rgb_photo)

        # Update Depth view
        if self.latest_depth_image is not None:
            w_target = 640
            h_target = int(w_target * self.latest_depth_image.height / max(1, self.latest_depth_image.width))
            ddisp = self.latest_depth_image.resize((w_target, h_target), Image.NEAREST)
            self.depth_photo = ImageTk.PhotoImage(ddisp)
            self.depth_panel.configure(image=self.depth_photo)

        # schedule next tick
        self.after(33, self.ui_tick)  # ~30 FPS


# ---------------------- Entry Point ------------------------

def main():
    pipe = build_depth_pipeline()
    app = DepthApp(pipe)
    app.mainloop()  # In Jupyter: run `%gui tk` and comment this out.

if __name__ == "__main__":
    main()

