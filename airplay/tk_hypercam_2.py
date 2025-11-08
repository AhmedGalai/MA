#!/usr/bin/env python3
"""
AVP API Debug Viewer
Simple debug client that displays API pipeline results.
NO screen capture - capture happens in screen_capture.py
NO control - control happens in screen_capture.py

ARCHITECTURE:
- screen_capture.py: Captures screen region and forwards to API
- avp_api.py: Processes frames (ArUco detection, pose estimation, masking)
- tk_hypercam_2.py: Debug viewer for pipeline results
"""

import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
import requests
import base64
import io

# ------------------ Configuration ------------------
API_BASE_URL = "http://localhost:5000"

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

# ------------------ Tkinter App ------------------
class AVPDebugViewer:
    def __init__(self, root):
        self.root = root
        root.title("AVP API Debug Viewer")
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

        # Tolerances
        self.h_tol_var = tk.IntVar(value=12)
        self.s_tol_var = tk.IntVar(value=50)
        self.v_tol_var = tk.IntVar(value=50)

        # Display options
        self.show_intrinsics = tk.BooleanVar(value=True)
        self.show_pose = tk.BooleanVar(value=True)
        self.show_mask = tk.BooleanVar(value=True)
        self.show_detected = tk.BooleanVar(value=True)
        self.show_head_pose = tk.BooleanVar(value=True)

        self.ui_hz = tk.IntVar(value=30)
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

        # HSV Color Selection
        ttk.Label(left, text="ROI Color (HSV)", font=("", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self._create_slider(left, "Hue", self.h_var, 0, 179)
        self._create_slider(left, "Saturation", self.s_var, 0, 255)
        self._create_slider(left, "Value", self.v_var, 0, 255)

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
        ttk.Checkbutton(left, text="Show Head Pose", variable=self.show_head_pose).pack(anchor="w")

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

        # Keep references to prevent GC
        self._img_rgb = None
        self._img_mask = None
        self._img_detected = None

    def _create_slider(self, parent, label, variable, from_, to):
        """Helper to create labeled slider"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label, width=15).pack(side="left")
        ttk.Scale(frame, from_=from_, to=to, orient="horizontal", variable=variable).pack(side="left", fill="x", expand=True)
        ttk.Label(frame, textvariable=variable, width=5).pack(side="left")

    def _initial_sync(self):
        """Initial sync with API (non-blocking)"""
        def sync():
            if self.api.health_check():
                # Get current config from API
                config = self.api.get_config()
                if config:
                    # Update UI with API config
                    self.root.after(0, lambda: self._update_from_config(config))
                    self.root.after(0, lambda: self.status_var.set("Connected to API - Waiting for frames..."))
                    # Start fetch thread automatically
                    self.root.after(0, lambda: self._start_display())
                else:
                    self.root.after(0, lambda: self.status_var.set("API connected but config failed"))
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "API Error",
                    f"Cannot connect to API at {API_BASE_URL}\n\nPlease start the API server:\npython avp_api.py"
                ))
                self.root.after(0, lambda: self.status_var.set("API not available"))

        threading.Thread(target=sync, daemon=True).start()

    def _update_from_config(self, config):
        """Update UI from API config"""
        self.left_var.set(config.get('left', 934))
        self.top_var.set(config.get('top', 100))
        self.width_var.set(config.get('width', 812))
        self.height_var.set(config.get('height', 1080))
        self.fps_var.set(config.get('fps', 30))

        hsv = config.get('hsv_center', [90, 128, 128])
        self.h_var.set(hsv[0])
        self.s_var.set(hsv[1])
        self.v_var.set(hsv[2])

        tol = config.get('tolerances', {'h': 12, 's': 50, 'v': 50})
        self.h_tol_var.set(tol.get('h', 12))
        self.s_tol_var.set(tol.get('s', 50))
        self.v_tol_var.set(tol.get('v', 50))

    def _apply_settings(self):
        """Apply current settings to API"""
        def apply():
            config = {
                'left': self.left_var.get(),
                'top': self.top_var.get(),
                'width': self.width_var.get(),
                'height': self.height_var.get(),
                'fps': self.fps_var.get(),
                'hsv_center': [self.h_var.get(), self.s_var.get(), self.v_var.get()],
                'tolerances': {
                    'h': self.h_tol_var.get(),
                    's': self.s_tol_var.get(),
                    'v': self.v_tol_var.get()
                }
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

            # Update detected frame
            if self.show_detected.get() and 'detected_frame' in data:
                self._update_image_label(self.detected_label, data['detected_frame'], "detected")

            # Update head pose
            if self.show_head_pose.get() and 'head_pose' in data:
                self._update_head_pose_display(data['head_pose'])

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

        # Keep reference
        if which == "rgb":
            self._img_rgb = tk_img
        elif which == "mask":
            self._img_mask = tk_img
        elif which == "detected":
            self._img_detected = tk_img

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
