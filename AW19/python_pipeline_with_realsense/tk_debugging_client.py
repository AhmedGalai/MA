#!/usr/bin/env python3
"""
AVP API Debug Viewer
Simple debug client that displays API pipeline results.
NO screen capture - capture happens in screen_capture.py
NO control - control happens in screen_capture.py

ARCHITECTURE:
- screen_capture.py: Captures screen region and forwards to API
- main_api.py: Processes frames (ArUco detection, pose estimation, masking)
- tk_debugging_client.py: Debug viewer for pipeline results
"""

import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
import requests
import base64
import io

# ------------------ Configuration ------------------
try:
    from app_config import APP_CONFIG
    API_BASE_URL = APP_CONFIG.get("main_api", {}).get("base_url", "http://localhost:5000")
    DEFAULT_UI_HZ = APP_CONFIG.get("defaults", {}).get("ui_refresh_hz", 30)
except Exception:
    API_BASE_URL = "http://localhost:5000"
    DEFAULT_UI_HZ = 30

# ------------------ API Client ------------------
class AVPAPIClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self):
        """Check if API is available"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=1)
            return response.status_code == 200
        except Exception:
            return False

    def get_config(self):
        """Get current configuration from API"""
        try:
            response = self.session.get(f"{self.base_url}/config", timeout=1)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def set_config(self, config):
        """Update configuration on API"""
        try:
            response = self.session.post(f"{self.base_url}/config", json=config, timeout=1)
            return response.status_code == 200
        except Exception:
            return False

    def get_data_batch(self):
        """Fetch all display data in one batch"""
        try:
            results = {}

            # Get RGB frame
            try:
                r = self.session.get(f"{self.base_url}/rgb_frame", timeout=0.5)
                if r.status_code == 200:
                    data = r.json()
                    frame_str = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
                    img_data = base64.b64decode(frame_str)
                    img = Image.open(io.BytesIO(img_data))
                    results['rgb_frame'] = np.array(img)
                    # keep base64 for forwarding to /avp_pose
                    results['rgb_frame_b64'] = data['frame']
            except Exception:
                pass

            # Get intrinsics
            try:
                r = self.session.get(f"{self.base_url}/intrinsics", timeout=0.5)
                if r.status_code == 200:
                    results['intrinsics'] = r.json()
            except Exception:
                pass

            # Get pose
            try:
                r = self.session.get(f"{self.base_url}/pose", timeout=0.5)
                if r.status_code == 200:
                    results['pose'] = r.json()
            except Exception:
                pass

            # Get mask
            try:
                r = self.session.get(f"{self.base_url}/mask", timeout=0.5)
                if r.status_code == 200:
                    data = r.json()
                    mask_str = data['mask'].split(',')[1] if ',' in data['mask'] else data['mask']
                    img_data = base64.b64decode(mask_str)
                    img = Image.open(io.BytesIO(img_data))
                    results['mask'] = np.array(img)
            except Exception:
                pass

            # Get detected frame
            try:
                r = self.session.get(f"{self.base_url}/detected_frame", timeout=0.5)
                if r.status_code == 200:
                    data = r.json()
                    frame_str = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
                    img_data = base64.b64decode(frame_str)
                    img = Image.open(io.BytesIO(img_data))
                    results['detected_frame'] = np.array(img)
            except Exception:
                pass

            # Get disparity
            try:
                r = self.session.get(f"{self.base_url}/disparity", timeout=0.5)
                if r.status_code == 200:
                    data = r.json()
                    disp_str = data['disparity'].split(',')[1] if ',' in data['disparity'] else data['disparity']
                    img_data = base64.b64decode(disp_str)
                    img = Image.open(io.BytesIO(img_data))
                    results['disparity'] = np.array(img)
            except Exception:
                pass

            # Get stats
            try:
                r = self.session.get(f"{self.base_url}/stats", timeout=0.5)
                if r.status_code == 200:
                    results['stats'] = r.json()
            except Exception:
                pass

            # Get head pose
            try:
                r = self.session.get(f"{self.base_url}/head_pose", timeout=0.5)
                if r.status_code == 200:
                    results['head_pose'] = r.json()
            except Exception:
                pass

            return results
        except Exception as e:
            print(f"[ERROR] get_data_batch: {e}")
            return {}

    def request_final_pose(self, rgb_b64, camera_matrix):
        """Ask main_api to forward to pose_api and return final pose"""
        try:
            payload = {
                "rgb_frame": rgb_b64,
                "camera_matrix": camera_matrix,
            }
            r = self.session.post(f"{self.base_url}/avp_pose", json=payload, timeout=2.0)
            if r.status_code == 200:
                return r.json()
            else:
                return {"error": f"HTTP {r.status_code}", "detail": r.text}
        except Exception as e:
            return {"error": str(e)}

# ------------------ Tkinter App ------------------
class AVPDebugViewer:
    def __init__(self, root):
        self.root = root
        root.title("AVP Debugging Client")
        root.geometry("1200x900")

        # API client
        self.api = AVPAPIClient(API_BASE_URL)

        # State
        self.running = False
        self.fetch_thread_alive = False

        # Queues for thread communication
        self.display_queue = queue.Queue(maxsize=1)

        # HSV color for ROI (only for config updates)
        self.h_var = tk.IntVar(value=90)
        self.s_var = tk.IntVar(value=128)
        self.v_var = tk.IntVar(value=128)
        self._roi_rgb = (0, 255, 0)  # default green

        # Tolerances
        self.h_tol_var = tk.IntVar(value=12)
        self.s_tol_var = tk.IntVar(value=50)
        self.v_tol_var = tk.IntVar(value=50)

        # Display options
        self.show_intrinsics = tk.BooleanVar(value=True)
        self.show_pose = tk.BooleanVar(value=True)
        self.show_mask = tk.BooleanVar(value=True)
        self.show_disparity = tk.BooleanVar(value=True)
        self.show_detected = tk.BooleanVar(value=True)
        self.show_head_pose = tk.BooleanVar(value=True)
        self.show_final_pose = tk.BooleanVar(value=False)

        self.ui_hz = tk.IntVar(value=DEFAULT_UI_HZ)
        self.status_var = tk.StringVar(value="Connecting...")

        # Build UI
        self._build_ui()

        # Start UI refresh loop
        self._schedule_ui_refresh()

        # Sync with API and check health
        self.root.after(500, self._initial_sync)

    # ---------- UI Building ----------
    def _build_ui(self):
        self.root.geometry("1600x900")
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        # Left panel - controls
        left = ttk.Frame(main, width=300)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        # Right panel - display
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        # --- Controls ---
        ttk.Label(left, text="AVP Viewer", font=("", 12, "bold")).pack(anchor="w", pady=(0, 8))

        # Info label
        info_text = "⚠️ Note: Start screen_capture.py\n   to begin sending frames"
        ttk.Label(left, text=info_text, font=("", 8), foreground="blue").pack(anchor="w", pady=(0, 8))

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        # UI Refresh Rate
        ttk.Label(left, text="Display Settings", font=("", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self._create_slider(left, "UI Refresh Hz", self.ui_hz, 1, 60)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)

        # ROI Color Picker (replaces HSV sliders)
        ttk.Label(left, text="ROI Color", font=("", 10, "bold")).pack(anchor="w", pady=(0, 4))
        color_frame = ttk.Frame(left)
        color_frame.pack(fill="x", pady=2)
        self._color_swatch = tk.Canvas(color_frame, width=24, height=16, highlightthickness=1, highlightbackground="#888")
        self._color_swatch.pack(side="left", padx=(0, 8))
        self._update_color_swatch()
        ttk.Button(color_frame, text="Pick Color", command=self._on_pick_color).pack(side="left")
        ttk.Label(color_frame, text="Applies to HSV center").pack(side="left", padx=6)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)

        # Tolerances
        ttk.Label(left, text="ROI Tolerances", font=("", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self._create_slider(left, "Hue Tol (deg)", self.h_tol_var, 5, 30)
        self._create_slider(left, "Sat Tol (%)", self.s_tol_var, 10, 60)
        self._create_slider(left, "Val Tol (%)", self.v_tol_var, 10, 60)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)

        # Apply button
        ttk.Button(left, text="Apply Settings to API", command=self._apply_settings).pack(fill="x", pady=4)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)

        # Display options
        ttk.Label(left, text="Display Options", font=("", 10, "bold")).pack(anchor="w", pady=(0, 4))
        ttk.Checkbutton(left, text="Show Intrinsics", variable=self.show_intrinsics).pack(anchor="w")
        ttk.Checkbutton(left, text="Show Pose", variable=self.show_pose).pack(anchor="w")
        ttk.Checkbutton(left, text="Show Mask", variable=self.show_mask).pack(anchor="w")
        ttk.Checkbutton(left, text="Show Detected Frame", variable=self.show_detected).pack(anchor="w")
        ttk.Checkbutton(left, text="Show Disparity", variable=self.show_disparity).pack(anchor="w")
        ttk.Checkbutton(left, text="Show Head Pose", variable=self.show_head_pose).pack(anchor="w")
        ttk.Checkbutton(left, text="Show Final Pose Overlay", variable=self.show_final_pose).pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)

        # Status
        self.status_var = tk.StringVar(value="Connecting to API...")
        ttk.Label(left, textvariable=self.status_var, wraplength=280, font=("", 8)).pack(anchor="w", pady=6)

        # --- Display Panels ---
        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)

        # RGB Feed tab
        rgb_frame = ttk.Frame(notebook)
        notebook.add(rgb_frame, text="RGB Feed")
        self.rgb_label = ttk.Label(rgb_frame, text="Waiting for data...")
        self.rgb_label.pack(fill="both", expand=True, padx=4, pady=4)

        # Intrinsics tab
        intrinsics_frame = ttk.Frame(notebook)
        notebook.add(intrinsics_frame, text="Intrinsics")
        self.intrinsics_text = tk.Text(intrinsics_frame, wrap="word", font=("Courier", 10))
        self.intrinsics_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Pose tab
        pose_frame = ttk.Frame(notebook)
        notebook.add(pose_frame, text="Pose")
        self.pose_text = tk.Text(pose_frame, wrap="word", font=("Courier", 10))
        self.pose_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Mask tab
        mask_frame = ttk.Frame(notebook)
        notebook.add(mask_frame, text="Mask")
        self.mask_label = ttk.Label(mask_frame, text="Waiting for data...")
        self.mask_label.pack(fill="both", expand=True, padx=4, pady=4)

        # Disparity tab
        disparity_frame = ttk.Frame(notebook)
        notebook.add(disparity_frame, text="Disparity")
        self.disparity_label = ttk.Label(disparity_frame, text="Waiting for data...")
        self.disparity_label.pack(fill="both", expand=True, padx=4, pady=4)

        # Detected Frame tab
        detected_frame = ttk.Frame(notebook)
        notebook.add(detected_frame, text="Detected Markers")
        self.detected_label = ttk.Label(detected_frame, text="Waiting for data...")
        self.detected_label.pack(fill="both", expand=True, padx=4, pady=4)

        # Head Pose tab
        head_pose_frame = ttk.Frame(notebook)
        notebook.add(head_pose_frame, text="Head Pose")
        self.head_pose_text = tk.Text(head_pose_frame, wrap="word", font=("Courier", 10))
        self.head_pose_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Final Pose tab (overlay on RGB with arrow)
        final_pose_frame = ttk.Frame(notebook)
        notebook.add(final_pose_frame, text="Final Pose")
        self.final_pose_label = ttk.Label(final_pose_frame, text="Waiting for data...")
        self.final_pose_label.pack(fill="both", expand=True, padx=4, pady=4)

        # Keep references to prevent GC
        self._img_rgb = None
        self._img_mask = None
        self._img_detected = None
        self._img_disparity = None
        self._img_final_pose = None

    def _update_color_swatch(self):
        r, g, b = self._roi_rgb
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self._color_swatch.delete("all")
        self._color_swatch.create_rectangle(0, 0, 24, 16, fill=hex_color, outline="")

    @staticmethod
    def _rgb_to_hsv_opencv(rgb_tuple):
        arr = np.uint8([[list(rgb_tuple)]])  # 1x1 RGB
        bgr = arr[:, :, ::-1]  # convert RGB->BGR for OpenCV
        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        h, s, v = [int(x) for x in hsv[0, 0]]
        return h, s, v

    def _on_pick_color(self):
        color = colorchooser.askcolor(color=f"#{self._roi_rgb[0]:02x}{self._roi_rgb[1]:02x}{self._roi_rgb[2]:02x}")
        if color and color[0]:
            r, g, b = [int(c) for c in color[0]]
            self._roi_rgb = (r, g, b)
            self._update_color_swatch()
            # Update HSV IntVars based on chosen RGB
            h, s, v = self._rgb_to_hsv_opencv((r, g, b))
            self.h_var.set(h)
            self.s_var.set(s)
            self.v_var.set(v)

    def _create_slider(self, parent, label, variable, from_, to):
        """Helper to create labeled slider"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label, width=15).pack(side="left")
        ttk.Scale(frame, from_=from_, to=to, orient="horizontal", variable=variable).pack(side="left", fill="x", expand=True)
        ttk.Label(frame, textvariable=variable, width=5).pack(side="left")

    def _initial_sync(self):
        """Initial sync with API (non-blocking with retries)"""
        def sync():
            max_retries = 10
            retry_delay = 1.0  # seconds

            for attempt in range(1, max_retries + 1):
                # Update status
                self.root.after(0, lambda a=attempt: self.status_var.set(
                    f"Connecting to API... (attempt {a}/{max_retries})"
                ))

                if self.api.health_check():
                    # Get current config from API
                    config = self.api.get_config()
                    if config:
                        # Update UI with API config
                        self.root.after(0, lambda: self._update_from_config(config))
                        self.root.after(0, lambda: self.status_var.set("Connected to API - Waiting for frames..."))
                        # Start fetch thread automatically
                        self.root.after(0, lambda: self._start_display())
                        return  # Success!
                    else:
                        self.root.after(0, lambda: self.status_var.set("API connected but config failed"))
                        return

                # Wait before retry (except on last attempt)
                if attempt < max_retries:
                    time.sleep(retry_delay)

            # All retries failed
            self.root.after(0, lambda: messagebox.showwarning(
                "API Connection",
                f"Cannot connect to API at {API_BASE_URL}\n\n"
                f"Tried {max_retries} times.\n\n"
                "Please ensure:\n"
                "1. Main API is running: python main_api.py\n"
                "2. API is on port 5000\n\n"
                "The viewer will keep trying in the background."
            ))
            self.root.after(0, lambda: self.status_var.set("API not available - retrying..."))

            # Schedule another retry in 5 seconds
            self.root.after(5000, self._initial_sync)

        threading.Thread(target=sync, daemon=True).start()

    def _update_from_config(self, config):
        """Update UI from API config"""
        # Only update HSV settings (screen capture settings removed)
        hsv = config.get('hsv_center', [90, 128, 128])
        self.h_var.set(hsv[0])
        self.s_var.set(hsv[1])
        self.v_var.set(hsv[2])
        # Keep swatch in sync with HSV
        # Convert HSV->RGB via OpenCV for visual preview
        hsv_img = np.uint8([[[self.h_var.get(), self.s_var.get(), self.v_var.get()]]])
        bgr = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)[0, 0]
        self._roi_rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # BGR->RGB
        self._update_color_swatch()

        # Handle both old and new config format
        if 'tolerances' in config:
            tol = config.get('tolerances', {'h': 12, 's': 50, 'v': 50})
            self.h_tol_var.set(tol.get('h', 12))
            self.s_tol_var.set(tol.get('s', 50))
            self.v_tol_var.set(tol.get('v', 50))
        else:
            # New format with direct keys
            self.h_tol_var.set(config.get('h_tol', 12))
            self.s_tol_var.set(config.get('s_tol', 50))
            self.v_tol_var.set(config.get('v_tol', 50))

    def _apply_settings(self):
        """Apply current settings to API"""
        def apply():
            config = {
                'hsv_center': [self.h_var.get(), self.s_var.get(), self.v_var.get()],
                'h_tol': self.h_tol_var.get(),
                's_tol': self.s_tol_var.get(),
                'v_tol': self.v_tol_var.get()
            }
            success = self.api.set_config(config)
            if success:
                self.root.after(0, lambda: self.status_var.set("Settings applied successfully"))
            else:
                self.root.after(0, lambda: self.status_var.set("Failed to apply settings"))

        threading.Thread(target=apply, daemon=True).start()

    # ---------- Display Control ----------
    def _start_display(self):
        """Start display fetch thread"""
        if self.running:
            return

        self.running = True
        self.fetch_thread_alive = True
        threading.Thread(target=self._fetch_loop, daemon=True).start()

    # ---------- Fetch Loop (Background Thread) ----------
    def _fetch_loop(self):
        """Fetch processed data from API"""
        while self.fetch_thread_alive:
            try:
                # Fetch all data at once
                data = self.api.get_data_batch()

                if data:
                    # Optionally request final pose and overlay
                    try:
                        if self.show_final_pose.get() and 'rgb_frame_b64' in data and 'intrinsics' in data:
                            K = data['intrinsics'].get('K')
                            pose_result = self.api.request_final_pose(data['rgb_frame_b64'], K)
                            if pose_result and 'pose' in pose_result:
                                # Prefer detected frame for overlay; fall back to raw RGB
                                base_img = None
                                if 'detected_frame' in data:
                                    base_img = data['detected_frame']
                                elif 'rgb_frame' in data:
                                    base_img = data['rgb_frame']
                                overlay = self._make_final_pose_overlay(base_img, K, pose_result['pose'])
                                if overlay is not None:
                                    data['final_pose_overlay'] = overlay
                                    data['final_pose'] = pose_result['pose']
                    except Exception as e:
                        print(f"[WARN] final pose overlay failed: {e}")

                    # Put in display queue (replace old data)
                    try:
                        self.display_queue.get_nowait()  # Remove old
                    except queue.Empty:
                        pass
                    self.display_queue.put_nowait(data)

                # Throttle fetch rate
                time.sleep(1.0 / max(1, self.ui_hz.get()))
            except Exception as e:
                print(f"[ERROR] Fetch: {e}")
                time.sleep(0.25)

    # ---------- UI Refresh (Main Thread) ----------
    def _schedule_ui_refresh(self):
        """Schedule next UI update"""
        period_ms = max(16, int(1000 / max(1, self.ui_hz.get())))
        self.root.after(period_ms, self._ui_tick)

    def _ui_tick(self):
        """Update UI with latest data (non-blocking)"""
        try:
            # Get latest data from queue (non-blocking)
            data = self.display_queue.get_nowait()

            # Update RGB feed
            if 'rgb_frame' in data:
                self._update_image_label(self.rgb_label, data['rgb_frame'], "rgb")

            # Update intrinsics
            if self.show_intrinsics.get() and 'intrinsics' in data:
                self._update_intrinsics_display(data['intrinsics'])

            # Update pose
            if self.show_pose.get() and 'pose' in data:
                self._update_pose_display(data['pose'])

            # Update mask
            if self.show_mask.get() and 'mask' in data:
                self._update_image_label(self.mask_label, data['mask'], "mask")

            # Update disparity
            if self.show_disparity.get() and 'disparity' in data:
                self._update_image_label(self.disparity_label, data['disparity'], "disparity")

            # Update detected frame
            if self.show_detected.get() and 'detected_frame' in data:
                self._update_image_label(self.detected_label, data['detected_frame'], "detected")

            # Update head pose
            if self.show_head_pose.get() and 'head_pose' in data:
                self._update_head_pose_display(data['head_pose'])

            # Update final pose overlay
            if self.show_final_pose.get() and 'final_pose_overlay' in data:
                self._update_image_label(self.final_pose_label, data['final_pose_overlay'], "final")

            # Update status
            if 'stats' in data:
                stats = data['stats']
                self.status_var.set(
                    f"Frames processed: {stats.get('frames_processed', 0)} | "
                    f"Pose: {'✓' if stats.get('has_pose') else '✗'} | "
                    f"Head Pose: {'✓' if stats.get('has_head_pose') else '✗'}"
                )

        except queue.Empty:
            pass  # No new data

        self._schedule_ui_refresh()

    def _update_image_label(self, label, img_arr, which):
        """Update image label with numpy array"""
        if img_arr is None or img_arr.size == 0:
            return

        # Ensure RGB format
        if img_arr.ndim == 2:
            img_rgb = cv.cvtColor(img_arr, cv.COLOR_GRAY2RGB)
        elif img_arr.shape[2] == 4:
            img_rgb = cv.cvtColor(img_arr, cv.COLOR_BGRA2RGB)
        elif img_arr.shape[2] == 3:
            img_rgb = img_arr
        else:
            return

        # Resize to fit display
        h, w = img_rgb.shape[:2]
        max_w = 900
        if w > max_w:
            scale = max_w / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_rgb = cv.resize(img_rgb, (new_w, new_h), interpolation=cv.INTER_AREA)

        # Convert to PhotoImage
        pil_img = Image.fromarray(img_rgb.astype('uint8'))
        tk_img = ImageTk.PhotoImage(image=pil_img)
        label.configure(image=tk_img)
        try:
            label.configure(text="")
        except Exception:
            pass

        # Keep reference
        if which == "rgb":
            self._img_rgb = tk_img
        elif which == "mask":
            self._img_mask = tk_img
        elif which == "detected":
            self._img_detected = tk_img
        elif which == "disparity":
            self._img_disparity = tk_img
        elif which == "final":
            self._img_final_pose = tk_img

    @staticmethod
    def _project_point(K, pt_cam):
        x, y, z = float(pt_cam[0]), float(pt_cam[1]), float(pt_cam[2])
        if z <= 1e-6:
            return None
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        return int(round(u)), int(round(v))

    def _make_final_pose_overlay(self, img_rgb, K, pose_payload):
        try:
            if img_rgb is None:
                return None
            K = np.array(K, dtype=float)
            # Extract 4x4 transform (first item if list)
            T_list = None
            if isinstance(pose_payload, dict):
                if 'transformation_matrix' in pose_payload:
                    T_list = pose_payload['transformation_matrix']
                elif 'T' in pose_payload:
                    T_list = pose_payload['T']
            if isinstance(T_list, list) and len(T_list) > 0:
                T = np.array(T_list[0], dtype=float)
            else:
                return None

            R = T[:3, :3]
            t = T[:3, 3]

            # Choose arrow length proportional to distance
            dist = float(np.linalg.norm(t))
            L = max(0.2 * dist, 1.0)
            origin_cam = t
            dir_cam = R @ np.array([0.0, 0.0, L], dtype=float)
            tip_cam = origin_cam + dir_cam

            p0 = self._project_point(K, origin_cam)
            p1 = self._project_point(K, tip_cam)
            if p0 is None or p1 is None:
                return None

            # Draw on a copy, handle color spaces explicitly
            overlay_bgr = img_rgb[:, :, ::-1].copy()  # RGB->BGR
            cv.arrowedLine(overlay_bgr, p0, p1, color=(0, 0, 255), thickness=4, tipLength=0.12)
            # Small circle at origin
            cv.circle(overlay_bgr, p0, 6, (0, 255, 255), -1)
            return overlay_bgr[:, :, ::-1]  # back to RGB
        except Exception as e:
            print(f"[ERROR] overlay: {e}")
            return None

    def _update_intrinsics_display(self, intrinsics):
        """Update intrinsics text display"""
        self.intrinsics_text.delete(1.0, tk.END)
        self.intrinsics_text.insert(tk.END, "Camera Intrinsics Matrix (K):\n\n")

        K = np.array(intrinsics['K'])
        for row in K:
            self.intrinsics_text.insert(tk.END, f"  {row[0]:10.2f}  {row[1]:10.2f}  {row[2]:10.2f}\n")

        self.intrinsics_text.insert(tk.END, f"\n\nFocal Length:\n")
        self.intrinsics_text.insert(tk.END, f"  fx = {K[0, 0]:.2f}\n")
        self.intrinsics_text.insert(tk.END, f"  fy = {K[1, 1]:.2f}\n")
        self.intrinsics_text.insert(tk.END, f"\nPrincipal Point:\n")
        self.intrinsics_text.insert(tk.END, f"  cx = {K[0, 2]:.2f}\n")
        self.intrinsics_text.insert(tk.END, f"  cy = {K[1, 2]:.2f}\n")

        dist = intrinsics['dist']
        self.intrinsics_text.insert(tk.END, f"\n\nDistortion Coefficients:\n")
        self.intrinsics_text.insert(tk.END, f"  {dist}\n")

    def _update_pose_display(self, pose):
        """Update pose text display"""
        self.pose_text.delete(1.0, tk.END)
        self.pose_text.insert(tk.END, "Board Pose:\n\n")

        if 'error' in pose:
            self.pose_text.insert(tk.END, f"Error: {pose['error']}\n")
            return

        rvec = pose.get('rvec', [0, 0, 0])
        tvec = pose.get('tvec', [0, 0, 0])
        markers = pose.get('markers_detected', 0)

        self.pose_text.insert(tk.END, f"Markers Detected: {markers}\n\n")
        self.pose_text.insert(tk.END, f"Rotation Vector (rvec):\n")
        self.pose_text.insert(tk.END, f"  x: {rvec[0]:10.6f}\n")
        self.pose_text.insert(tk.END, f"  y: {rvec[1]:10.6f}\n")
        self.pose_text.insert(tk.END, f"  z: {rvec[2]:10.6f}\n")
        self.pose_text.insert(tk.END, f"\nTranslation Vector (tvec):\n")
        self.pose_text.insert(tk.END, f"  x: {tvec[0]:10.6f} m\n")
        self.pose_text.insert(tk.END, f"  y: {tvec[1]:10.6f} m\n")
        self.pose_text.insert(tk.END, f"  z: {tvec[2]:10.6f} m\n")

        # Calculate distance
        distance = np.sqrt(sum([t**2 for t in tvec]))
        self.pose_text.insert(tk.END, f"\nDistance: {distance:.6f} m\n")

    def _update_head_pose_display(self, head_pose_data):
        """Update head pose text display"""
        self.head_pose_text.delete(1.0, tk.END)
        self.head_pose_text.insert(tk.END, "Head Pose Data (from AVP):\n\n")

        if 'error' in head_pose_data:
            self.head_pose_text.insert(tk.END, f"Error: {head_pose_data['error']}\n")
            return

        # Extract head pose information
        head_pose = head_pose_data.get('head_pose', {})
        age = head_pose_data.get('age_seconds', 0)
        received_at = head_pose_data.get('received_at', 0)

        # Position
        position = head_pose.get('position', [0, 0, 0])
        self.head_pose_text.insert(tk.END, f"Position:\n")
        self.head_pose_text.insert(tk.END, f"  x: {position[0]:10.6f} m\n")
        self.head_pose_text.insert(tk.END, f"  y: {position[1]:10.6f} m\n")
        self.head_pose_text.insert(tk.END, f"  z: {position[2]:10.6f} m\n")

        # Rotation (Euler angles)
        rotation = head_pose.get('rotation', [0, 0, 0])
        self.head_pose_text.insert(tk.END, f"\nRotation (Euler):\n")
        self.head_pose_text.insert(tk.END, f"  pitch: {rotation[0]:10.6f} rad\n")
        self.head_pose_text.insert(tk.END, f"  yaw:   {rotation[1]:10.6f} rad\n")
        self.head_pose_text.insert(tk.END, f"  roll:  {rotation[2]:10.6f} rad\n")

        # Quaternion
        quaternion = head_pose.get('quaternion', [0, 0, 0, 1])
        self.head_pose_text.insert(tk.END, f"\nQuaternion:\n")
        self.head_pose_text.insert(tk.END, f"  x: {quaternion[0]:10.6f}\n")
        self.head_pose_text.insert(tk.END, f"  y: {quaternion[1]:10.6f}\n")
        self.head_pose_text.insert(tk.END, f"  z: {quaternion[2]:10.6f}\n")
        self.head_pose_text.insert(tk.END, f"  w: {quaternion[3]:10.6f}\n")

        # Metadata
        confidence = head_pose.get('confidence', 1.0)
        timestamp = head_pose.get('timestamp', 0)
        metadata = head_pose.get('metadata', {})

        self.head_pose_text.insert(tk.END, f"\nMetadata:\n")
        self.head_pose_text.insert(tk.END, f"  Confidence: {confidence:.4f}\n")
        self.head_pose_text.insert(tk.END, f"  Data age: {age:.3f} seconds\n")
        self.head_pose_text.insert(tk.END, f"  Timestamp: {timestamp:.3f}\n")

        if metadata:
            self.head_pose_text.insert(tk.END, f"\nAdditional Info:\n")
            for key, value in metadata.items():
                self.head_pose_text.insert(tk.END, f"  {key}: {value}\n")

# ------------------ Main ------------------
if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    app = AVPDebugViewer(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'fetch_thread_alive', False), root.destroy()))
    root.mainloop()
