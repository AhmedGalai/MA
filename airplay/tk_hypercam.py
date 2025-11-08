#!/usr/bin/env python3
import time, threading, colorsys
import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
import numpy as np
import cv2 as cv
from dataclasses import dataclass
from typing import Optional, Tuple
from mss import mss
from PIL import Image, ImageTk

# ------------------ ArUco helpers ------------------
def get_aruco_handles():
    if not hasattr(cv, "aruco"):
        print("[info] cv2.aruco not available; detection disabled")
        return None, None, None
    aruco = cv.aruco
    try:
        dct = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    except Exception:
        dct = aruco.Dictionary_get(aruco.DICT_4X4_50)
    try:
        params = aruco.DetectorParameters()
    except Exception:
        params = aruco.DetectorParameters_create()
    if hasattr(aruco, "ArucoDetector"):
        det = aruco.ArucoDetector(dct, params)
        api = "new"
    else:
        det = params
        api = "old"
    return dct, det, api

ROWS, COLS = 3, 4
MARKER_SIZE_M = 0.030
SEPARATION_M  = 0.010

def board_id_to_corners_m(marker_id: int):
    if marker_id < 0 or marker_id >= ROWS*COLS: return None
    row, col = divmod(marker_id, COLS)
    x0 = col * (MARKER_SIZE_M + SEPARATION_M)
    y0 = row * (MARKER_SIZE_M + SEPARATION_M)
    return np.array([[x0,               y0,               0],
                     [x0+MARKER_SIZE_M, y0,               0],
                     [x0+MARKER_SIZE_M, y0+MARKER_SIZE_M, 0],
                     [x0,               y0+MARKER_SIZE_M, 0]], dtype=np.float32)

def solve_board_pose_fallback(corners, ids, K, dist):
    if ids is None or len(ids)==0: return None
    obj_pts, img_pts = [], []
    for i, c in zip(ids.flatten().tolist(), corners):
        obj = board_id_to_corners_m(i)
        if obj is None: continue
        pts = np.asarray(c, dtype=np.float32).reshape(-1,2)
        obj_pts.append(obj); img_pts.append(pts)
    if not obj_pts: return None
    obj_pts = np.concatenate(obj_pts, axis=0)
    img_pts = np.concatenate(img_pts, axis=0)
    ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K, dist, flags=cv.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K, dist)
        if not ok: return None
    return rvec.reshape(3), tvec.reshape(3)

def draw_axis(bgr, K, dist, rvec, tvec, s=0.05):
    axis = np.float32([[0,0,0],[s,0,0],[0,s,0],[0,0,s]])
    proj, _ = cv.projectPoints(axis, rvec, tvec, K, dist)
    p = proj.reshape(-1,2).astype(int)
    cv.line(bgr, tuple(p[0]), tuple(p[1]), (0,0,255), 2)
    cv.line(bgr, tuple(p[0]), tuple(p[2]), (0,255,0), 2)
    cv.line(bgr, tuple(p[0]), tuple(p[3]), (255,0,0), 2)

def draw_detected(bgr, corners, ids):
    out = bgr.copy()
    if ids is not None and len(ids)>0:
        try:
            cv.aruco.drawDetectedMarkers(out, corners, ids)
        except Exception:
            for c in corners:
                pts = np.asarray(c, dtype=np.int32).reshape(-1,2)
                for i in range(4):
                    cv.line(out, tuple(pts[i]), tuple(pts[(i+1)%4]), (0,255,255), 2)
    return out

def default_K_for_size(w, h):
    f = 0.8 * max(w, h)
    cx, cy = w/2.0, h/2.0
    K = np.array([[f, 0, cx],[0, f, cy],[0, 0, 1]], dtype=np.float32)
    dist = np.zeros((5,1), np.float32)
    return K, dist

# ------------------ Screen capture ------------------
@dataclass
class Crop:
    left: int
    top: int
    width: int
    height: int
# after
class ScreenGrabber:
    def __init__(self, crop: Crop):
        self.mon = {"left": crop.left, "top": crop.top, "width": crop.width, "height": crop.height}

    def update_crop(self, crop: Crop):
        self.mon = {"left": crop.left, "top": crop.top, "width": crop.width, "height": crop.height}

# ------------------ Tkinter app ------------------
class HyperCamApp:
    def __init__(self, root):
        self.root = root
        root.title("HyperCam (Tk)")

        # ArUco
        self._ARUCO_DICT, self._DET_OR_PARAMS, self._API = get_aruco_handles()

        # State
        self.running = False
        self.last_proc_t = 0.0
        self.frames_read = 0
        self.last_ts = 0.0

        # Defaults
        self.crop = Crop(100, 100, 960, 540)
        self.proc_fps = tk.IntVar(value=10)
        self.ui_hz = tk.IntVar(value=12)

        self.show_detection = tk.BooleanVar(value=True)
        self.show_intrinsics = tk.BooleanVar(value=False)
        self.show_disparity = tk.BooleanVar(value=False)
        self.show_roi = tk.BooleanVar(value=False)
        self.show_contours = tk.BooleanVar(value=False)

        self.h_tol = tk.IntVar(value=12)   # degrees
        self.s_tol = tk.IntVar(value=50)   # %
        self.v_tol = tk.IntVar(value=50)   # %
        self.hex_color = "#22AAFF"

        # Build UI
        self._build_ui()

        # Capture + processing thread
        self.lock = threading.Lock()
        self.latest = None
        self.reader_alive = False
        self.grabber = ScreenGrabber(self.crop)

        # Start UI refresh loop
        self._schedule_ui_refresh()

    # ---------- UI building ----------
    def _build_ui(self):
        self.root.geometry("1400x820")
        main = ttk.Frame(self.root); main.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(main); left.pack(side="left", fill="y", padx=(0,8))
        right = ttk.Frame(main); right.pack(side="left", fill="both", expand=True)

        # Controls
        ttk.Label(left, text="Capture region").pack(anchor="w")
        grid = ttk.Frame(left); grid.pack(fill="x", pady=4)
        self.x_var = tk.IntVar(value=self.crop.left)
        self.y_var = tk.IntVar(value=self.crop.top)
        self.w_var = tk.IntVar(value=self.crop.width)
        self.h_var = tk.IntVar(value=self.crop.height)
        for i, (lab, var) in enumerate([("Left", self.x_var), ("Top", self.y_var),
                                        ("Width", self.w_var), ("Height", self.h_var)]):
            ttk.Label(grid, text=lab).grid(row=i, column=0, sticky="w")
            ttk.Entry(grid, textvariable=var, width=8).grid(row=i, column=1, sticky="w")

        ttk.Separator(left).pack(fill="x", pady=6)

        ttk.Label(left, text="Timing").pack(anchor="w")
        ttk.Label(left, text="Process FPS").pack(anchor="w")
        ttk.Scale(left, from_=1, to=30, orient="horizontal", variable=self.proc_fps).pack(fill="x")
        ttk.Label(left, text="UI Hz").pack(anchor="w")
        ttk.Scale(left, from_=1, to=30, orient="horizontal", variable=self.ui_hz).pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=6)

        ttk.Label(left, text="Views").pack(anchor="w")
        ttk.Checkbutton(left, text="Detection", variable=self.show_detection).pack(anchor="w")
        ttk.Checkbutton(left, text="Intrinsics", variable=self.show_intrinsics).pack(anchor="w")
        ttk.Checkbutton(left, text="Disparity (placeholder)", variable=self.show_disparity).pack(anchor="w")
        ttk.Checkbutton(left, text="ROI (HSV near picked color)", variable=self.show_roi).pack(anchor="w")
        ttk.Checkbutton(left, text="Contours (needs ROI)", variable=self.show_contours).pack(anchor="w")

        ttk.Separator(left).pack(fill="x", pady=6)

        ttk.Label(left, text="ROI color").pack(anchor="w")
        btn = ttk.Button(left, text="Pick color", command=self._pick_color); btn.pack(anchor="w", pady=2)
        ttk.Label(left, text="Hue tol (deg)").pack(anchor="w")
        ttk.Scale(left, from_=5, to=30, orient="horizontal", variable=self.h_tol).pack(fill="x")
        ttk.Label(left, text="Sat tol (%)").pack(anchor="w")
        ttk.Scale(left, from_=10, to=60, orient="horizontal", variable=self.s_tol).pack(fill="x")
        ttk.Label(left, text="Val tol (%)").pack(anchor="w")
        ttk.Scale(left, from_=10, to=60, orient="horizontal", variable=self.v_tol).pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=6)

        ctrl = ttk.Frame(left); ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Start", command=self.start).pack(side="left", expand=True, fill="x")
        ttk.Button(ctrl, text="Stop", command=self.stop).pack(side="left", expand=True, fill="x")
        ttk.Button(left, text="Apply Crop", command=self._apply_crop).pack(fill="x", pady=4)

        self.status_var = tk.StringVar(value="Stopped.")
        ttk.Label(left, textvariable=self.status_var, wraplength=220).pack(anchor="w", pady=6)

        # Image panes
        panes = ttk.Frame(right); panes.pack(fill="both", expand=True)
        self.rgb_label = ttk.Label(panes); self.rgb_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.det_label = ttk.Label(panes); self.det_label.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self.misc_label = ttk.Label(panes); self.misc_label.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)
        panes.columnconfigure(0, weight=1); panes.columnconfigure(1, weight=1); panes.columnconfigure(2, weight=1)
        panes.rowconfigure(0, weight=1)

        # Keep references to PhotoImage to avoid GC
        self._img_rgb = None
        self._img_det = None
        self._img_misc = None

    def _pick_color(self):
        rgb, hx = colorchooser.askcolor(color=self.hex_color, title="Pick ROI color")
        if hx:
            self.hex_color = hx

    def _apply_crop(self):
        left = max(0, int(self.x_var.get()))
        top  = max(0, int(self.y_var.get()))
        width  = max(16, int(self.w_var.get()))
        height = max(16, int(self.h_var.get()))
        self.crop = Crop(left, top, width, height)
        self.grabber.update_crop(self.crop)

    # ---------- Run/stop ----------
    def start(self):
        if self.running:
            return
        self.running = True
        self.reader_alive = True
        self.frames_read = 0
        self.last_ts = 0.0
        threading.Thread(target=self._reader_loop, daemon=True).start()
        self.status_var.set("Running… mirror something into the cropped region.")

    def stop(self):
        self.reader_alive = False
        self.running = False
        self.status_var.set("Stopped.")

    # ---------- Reader ----------
    
    def _reader_loop(self):
    # mss object lives and dies in THIS thread
        with mss() as sct:
            while self.reader_alive:
                try:
                    frame = np.asarray(sct.grab(self.grabber.mon))  # BGRA
                    bgr = frame[..., :3].copy()
                    with self.lock:
                        self.latest = (bgr, time.time())
                    time.sleep(0.001)
                except Exception as e:
                    with self.lock:
                        self.latest = None
                    print("reader error:", e)
                    time.sleep(0.25)
    # ---------- Processing ----------
    @staticmethod
    def _hex_to_hsv_bounds(hex_str, h_deg, s_pct, v_pct):
        hex_str = hex_str.lstrip("#")
        r = int(hex_str[0:2], 16); g = int(hex_str[2:4], 16); b = int(hex_str[4:6], 16)
        bgr = np.uint8([[[b,g,r]]])
        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)[0,0,:].astype(int)
        h0,s0,v0 = hsv.tolist()
        dh = int(round((h_deg/360.0)*179.0))
        ds = int(round((s_pct/100.0)*255.0))
        dv = int(round((v_pct/100.0)*255.0))
        lo = np.array([max(0, h0-dh), max(0, s0-ds), max(0, v0-dv)], dtype=np.uint8)
        hi = np.array([min(179, h0+dh), min(255, s0+ds), min(255, v0+dv)], dtype=np.uint8)
        return lo, hi

    def _make_info_panel(self, K, dist, size):
        panel = np.zeros((160, 380, 3), np.uint8)
        y=24
        def put(s):
            nonlocal y
            cv.putText(panel, s, (8,y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv.LINE_AA)
            y += 22
        if K is None:
            put("Intrinsics: default")
        else:
            put("Intrinsics:")
            put(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
            put(f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
        if size: put(f"img={size[0]}x{size[1]}")
        return panel

    def _process_once(self, frame):
        h, w = frame.shape[:2]
        K_use, D_use = default_K_for_size(w, h)

        det_img = None
        if self.show_detection.get() and self._ARUCO_DICT is not None:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if self._API == "new":
                corners, ids, _rej = self._DET_OR_PARAMS.detectMarkers(gray)
            else:
                corners, ids, _rej = cv.aruco.detectMarkers(gray, self._ARUCO_DICT, parameters=self._DET_OR_PARAMS)
            det_img = draw_detected(frame, corners, ids)
            if ids is not None and len(ids)>0 and self.show_intrinsics.get():
                sol = solve_board_pose_fallback(corners, ids, K_use, D_use)
                if sol is not None:
                    rvec, tvec = sol
                    try: draw_axis(det_img, K_use, D_use, rvec, tvec)
                    except Exception: pass
        # misc stack
        misc_panels = []
        if self.show_intrinsics.get():
            misc_panels.append(self._make_info_panel(K_use, D_use, (w,h)))
        if self.show_disparity.get():
            disp = np.zeros((h, w, 3), np.uint8)
            cv.putText(disp, "Disparity placeholder (no depth)", (10,30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv.LINE_AA)
            misc_panels.append(disp)
        if self.show_roi.get():
            lo, hi = self._hex_to_hsv_bounds(self.hex_color, self.h_tol.get(), self.s_tol.get(), self.v_tol.get())
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            m = cv.inRange(hsv, lo, hi)
            roi = cv.bitwise_and(frame, frame, mask=m)
            misc_panels.append(roi)
        if self.show_contours.get() and self.show_roi.get():
            lo, hi = self._hex_to_hsv_bounds(self.hex_color, self.h_tol.get(), self.s_tol.get(), self.v_tol.get())
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            m = cv.inRange(hsv, lo, hi)
            cnts, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            ac = frame.copy()
            if cnts:
                c = max(cnts, key=cv.contourArea)
                cv.drawContours(ac, [c], -1, (0,0,255), 2)
            misc_panels.append(ac)

        # prepare outputs
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        if det_img is None:
            det_img = np.zeros_like(frame)
        else:
            det_img = cv.cvtColor(det_img, cv.COLOR_BGR2RGB)

        if misc_panels:
            target_w = w
            res = []
            for p in misc_panels:
                phh, pww = p.shape[:2]
                if pww != target_w:
                    scale = target_w / float(pww)
                    p = cv.resize(p, (target_w, int(phh*scale)), interpolation=cv.INTER_AREA)
                if p.ndim == 2:
                    p = cv.cvtColor(p, cv.COLOR_GRAY2RGB)
                else:
                    p = cv.cvtColor(p, cv.COLOR_BGR2RGB)
                res.append(p)
            misc_img = np.vstack(res)
        else:
            misc_img = np.zeros_like(rgb_img)

        return rgb_img, det_img, misc_img

    # ---------- UI refresh ----------
    def _schedule_ui_refresh(self):
        period_ms = max(5, int(1000 / max(1, self.ui_hz.get())))
        self.root.after(period_ms, self._ui_tick)

    def _ui_tick(self):
        # fetch latest
        with self.lock:
            sample = self.latest
        if sample is None:
            self._show_blank()
            self.status_var.set("Frames: 0 | Latest age: -- | Proc FPS=%d | UI=%d Hz | Color %s" %
                                (self.proc_fps.get(), self.ui_hz.get(), self.hex_color))
        else:
            frame_bgr, ts = sample
            now = time.time()
            age = now - ts
            # throttle processing
            min_dt = 1.0 / float(max(1, self.proc_fps.get()))
            do_process = (now - self.last_proc_t) >= min_dt
            if do_process:
                rgb_img, det_img, misc_img = self._process_once(frame_bgr)
                self._update_panel(self.rgb_label, rgb_img, which="rgb")
                self._update_panel(self.det_label, det_img, which="det")
                self._update_panel(self.misc_label, misc_img, which="misc")
                self.frames_read += 1
                self.last_proc_t = now
            self.status_var.set("Frames: %d | Latest age: %.2fs | Proc FPS=%d | UI=%d Hz | Crop %dx%d@(%d,%d) | Color %s" %
                                (self.frames_read, age, self.proc_fps.get(), self.ui_hz.get(),
                                 self.crop.width, self.crop.height, self.crop.left, self.crop.top, self.hex_color))
        self._schedule_ui_refresh()

    def _show_blank(self):
        # create simple black placeholders sized to last crop
        h, w = self.crop.height, self.crop.width
        blank = np.zeros((h, w, 3), np.uint8)
        self._update_panel(self.rgb_label, blank, which="rgb")
        self._update_panel(self.det_label, blank, which="det")
        self._update_panel(self.misc_label, blank, which="misc")

    def _update_panel(self, label, img_rgb, which="rgb"):
        # Resize to fit label area width while preserving aspect; for simplicity, clamp width
        max_w = 420
        h, w = img_rgb.shape[:2]
        if w > max_w:
            scale = max_w / float(w)
            img_rgb = cv.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv.INTER_AREA)
        im = Image.fromarray(img_rgb)
        tk_im = ImageTk.PhotoImage(image=im)
        label.configure(image=tk_im)
        if which == "rgb":
            self._img_rgb = tk_im
        elif which == "det":
            self._img_det = tk_im
        else:
            self._img_misc = tk_im

# ------------------ Main ------------------
if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = HyperCamApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    root.mainloop()

