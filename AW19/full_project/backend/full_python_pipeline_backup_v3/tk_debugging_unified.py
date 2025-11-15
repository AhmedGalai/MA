#!/usr/bin/env python3
"""
Enhanced AVP API Debug Viewer
Complete debugging client with all necessary feeds and controls:
- RGB feed
- Disparity feed (RealSense or Transformers)
- ArUco Detection feed
- Binary mask feed
- RGB feed with 6D pose overlay
- Final clean binary mask (ROI circle only)
- ROI threshold controls (color picker with tolerance sliders)
- Hough circle transformation for clean ROI
- API connection controls
- UI refresh rate control
- Data display from main API
"""

import time
import threading
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
    DEFAULT_HOST = APP_CONFIG.get("main_api", {}).get("host", "localhost")
    DEFAULT_PORT = APP_CONFIG.get("main_api", {}).get("port", 8000)
    DEFAULT_UI_HZ = APP_CONFIG.get("defaults", {}).get("ui_refresh_hz", 15)
    DEFAULT_HSV = APP_CONFIG.get("defaults", {}).get("roi_hsv_center", [90, 128, 128])
    DEFAULT_TOL = APP_CONFIG.get("defaults", {}).get("tolerances", {"h": 10, "s": 50, "v": 50})
except Exception:
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 8000
    DEFAULT_UI_HZ = 15
    DEFAULT_HSV = [90, 128, 128]
    DEFAULT_TOL = {"h": 10, "s": 50, "v": 50}

# ------------------ Helper Functions ------------------
def decode_base64_image(base64_str):
    """Decode base64 string to numpy array"""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        return np.array(img)
    except Exception as e:
        print(f"[ERROR] decode_base64_image: {e}")
        return None

def normalize_for_display(img_array):
    """Normalize depth/disparity array to 0-255 range for proper display."""
    if img_array is None or img_array.size == 0:
        return None

    if img_array.dtype != np.float32:
        img_array = img_array.astype(np.float32)

    img_min = img_array.min()
    img_max = img_array.max()

    if img_max - img_min < 1e-6:
        return np.zeros_like(img_array, dtype=np.uint8)

    normalized = ((img_array - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    return normalized

def apply_colormap_for_depth(img_array, colormap=cv.COLORMAP_TURBO):
    """Apply colormap to depth/disparity for better visualization."""
    normalized = normalize_for_display(img_array)
    if normalized is None:
        return None
    colored = cv.applyColorMap(normalized, colormap)
    return colored

def detect_roi_circle_hough(binary_mask):
    """
    Use Hough Circle Transform to detect the ROI circle.
    Returns: (x, y, radius) or None
    Falls back to None if detection fails.
    """
    if binary_mask is None or binary_mask.size == 0:
        return None

    try:
        # Ensure single channel uint8
        if binary_mask.ndim == 3:
            binary_mask = cv.cvtColor(binary_mask, cv.COLOR_BGR2GRAY)

        if binary_mask.dtype != np.uint8:
            binary_mask = binary_mask.astype(np.uint8)

        # Apply some morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, kernel)
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, kernel)

        # Detect circles using Hough transform
        circles = cv.HoughCircles(
            cleaned,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=300
        )

        if circles is not None and len(circles) > 0:
            circles = np.uint16(np.around(circles))
            # Take the first (strongest) circle
            x, y, r = circles[0, 0]
            return (int(x), int(y), int(r))

        return None
    except Exception as e:
        print(f"[WARNING] Hough circle detection failed: {e}")
        return None

def create_clean_roi_mask(binary_mask, circle_params):
    """
    Create a clean binary mask with only the ROI circle, rest is black.

    Args:
        binary_mask: Original binary mask
        circle_params: (x, y, radius) from Hough circle detection

    Returns:
        Clean mask with only the circle region
    """
    if circle_params is None:
        return binary_mask

    try:
        x, y, r = circle_params
        clean_mask = np.zeros_like(binary_mask)
        cv.circle(clean_mask, (x, y), r, 255, -1)  # Fill circle with white
        return clean_mask
    except Exception as e:
        print(f"[ERROR] create_clean_roi_mask: {e}")
        return binary_mask

def draw_6d_pose_overlay(rgb_frame, pose_data, intrinsics_data):
    """
    Draw 6D pose overlay on RGB frame.

    Args:
        rgb_frame: RGB image as numpy array
        pose_data: Pose data from API (rvec, tvec)
        intrinsics_data: Camera intrinsics (K, dist)

    Returns:
        Frame with pose overlay drawn
    """
    if pose_data is None or intrinsics_data is None:
        return rgb_frame

    try:
        frame = rgb_frame.copy()

        # Extract pose
        rvec = np.array(pose_data.get('rvec', [[0], [0], [0]]), dtype=np.float32).reshape(3, 1)
        tvec = np.array(pose_data.get('tvec', [[0], [0], [0]]), dtype=np.float32).reshape(3, 1)

        # Extract intrinsics
        K = np.array(intrinsics_data.get('K', [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), dtype=np.float32)
        dist = np.array(intrinsics_data.get('dist', [0, 0, 0, 0, 0]), dtype=np.float32)

        # Define axis points in 3D (length = 0.05m = 50mm)
        axis_length = 0.05
        axis_points_3d = np.float32([
            [0, 0, 0],  # Origin
            [axis_length, 0, 0],  # X-axis (red)
            [0, axis_length, 0],  # Y-axis (green)
            [0, 0, axis_length]   # Z-axis (blue)
        ])

        # Project 3D points to 2D
        img_points, _ = cv.projectPoints(axis_points_3d, rvec, tvec, K, dist)
        img_points = img_points.reshape(-1, 2).astype(int)

        # Draw axes
        origin = tuple(img_points[0])
        x_end = tuple(img_points[1])
        y_end = tuple(img_points[2])
        z_end = tuple(img_points[3])

        # Draw lines with different colors for each axis
        cv.line(frame, origin, x_end, (0, 0, 255), 3)  # X-axis: Red
        cv.line(frame, origin, y_end, (0, 255, 0), 3)  # Y-axis: Green
        cv.line(frame, origin, z_end, (255, 0, 0), 3)  # Z-axis: Blue

        # Draw origin point
        cv.circle(frame, origin, 5, (255, 255, 255), -1)

        return frame
    except Exception as e:
        print(f"[ERROR] draw_6d_pose_overlay: {e}")
        return rgb_frame

def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB for color display"""
    # Normalize HSV values to 0-1 range
    h_norm = h / 179.0
    s_norm = s / 255.0
    v_norm = v / 255.0

    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
    return (int(r * 255), int(g * 255), int(b * 255))

# ------------------ API Client ------------------
class UnifiedAPIClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()

    def update_base_url(self, host, port):
        """Update the base URL dynamically"""
        self.base_url = f"http://{host}:{port}"

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

    def update_config(self, config_data):
        """Update API configuration"""
        try:
            response = self.session.post(f"{self.base_url}/config", json=config_data, timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def get_stats(self):
        """Get pipeline statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=1)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def get_data_batch(self):
        """Fetch all display data in one batch"""
        try:
            results = {}
            fetch_time = time.time()

            # Get RGB frame
            try:
                r = self.session.get(f"{self.base_url}/rgb_frame", timeout=0.5)
                if r.status_code == 200:
                    data = r.json()
                    img_array = decode_base64_image(data['frame'])
                    if img_array is not None:
                        results['rgb_frame'] = img_array
            except Exception as e:
                print(f"[DEBUG] rgb_frame error: {e}")

            # Get intrinsics
            try:
                r = self.session.get(f"{self.base_url}/intrinsics", timeout=0.5)
                if r.status_code == 200:
                    results['intrinsics'] = r.json()
            except Exception as e:
                print(f"[DEBUG] intrinsics error: {e}")

            # Get pose
            try:
                r = self.session.get(f"{self.base_url}/pose", timeout=0.5)
                if r.status_code == 200:
                    results['pose'] = r.json()
            except Exception as e:
                print(f"[DEBUG] pose error: {e}")

            # Get mask
            try:
                r = self.session.get(f"{self.base_url}/mask", timeout=0.5)
                if r.status_code == 200:
                    data = r.json()
                    img_array = decode_base64_image(data['mask'])
                    if img_array is not None:
                        results['mask'] = img_array
            except Exception as e:
                print(f"[DEBUG] mask error: {e}")

            # Get disparity
            try:
                r = self.session.get(f"{self.base_url}/disparity", timeout=0.5)
                if r.status_code == 200:
                    data = r.json()
                    img_array = decode_base64_image(data['disparity'])
                    if img_array is not None:
                        results['disparity'] = img_array
                        results['disparity_timestamp'] = data.get('timestamp')
            except Exception as e:
                print(f"[DEBUG] disparity error: {e}")

            # Get detected frame (with ArUco markers)
            try:
                r = self.session.get(f"{self.base_url}/detected_frame", timeout=0.5)
                if r.status_code == 200:
                    data = r.json()
                    img_array = decode_base64_image(data['frame'])
                    if img_array is not None:
                        results['detected_frame'] = img_array
            except Exception as e:
                print(f"[DEBUG] detected_frame error: {e}")

            # Get head pose
            try:
                r = self.session.get(f"{self.base_url}/head_pose", timeout=0.5)
                if r.status_code == 200:
                    results['head_pose'] = r.json()
            except Exception as e:
                print(f"[DEBUG] head_pose error: {e}")

            results['fetch_time'] = fetch_time
            return results

        except Exception as e:
            print(f"[ERROR] get_data_batch: {e}")
            return {}

# ------------------ Main UI ------------------
class EnhancedDebugViewer:
    def __init__(self, root):
        self.root = root
        root.title("Enhanced AVP Pipeline Debug Viewer")
        root.geometry("1800x1000")

        # API client
        self.api_host = DEFAULT_HOST
        self.api_port = DEFAULT_PORT
        self.client = UnifiedAPIClient(f"http://{self.api_host}:{self.api_port}")

        # State
        self.running = False
        self.update_thread = None
        self.photo_refs = {}
        self.ui_refresh_hz = DEFAULT_UI_HZ
        self.use_realsense = False
        self.use_reverse_rs = False  # Reverse RealSense approach

        # ROI parameters
        self.hsv_center = DEFAULT_HSV.copy()
        self.h_tol = DEFAULT_TOL.get("h", 10)
        self.s_tol = DEFAULT_TOL.get("s", 50)
        self.v_tol = DEFAULT_TOL.get("v", 50)

        # Build UI
        self._build_ui()

        # Check API connection
        if not self.client.health_check():
            messagebox.showwarning(
                "API Connection",
                f"Cannot connect to API at http://{self.api_host}:{self.api_port}\n"
                "Make sure main_api.py is running."
            )
        else:
            self.start_updates()

        # Handle window close
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        """Build the complete UI layout"""
        # Main container with scrollbar
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top: Controls panel
        controls_frame = ttk.LabelFrame(main_container, text="Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        self._build_controls(controls_frame)

        # Middle: Image grid (2 rows x 3 columns)
        images_frame = ttk.Frame(main_container)
        images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self._build_image_grid(images_frame)

        # Bottom: Info panel
        info_frame = ttk.LabelFrame(main_container, text="API Data & Statistics", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        self._build_info_panel(info_frame)

    def _build_controls(self, parent):
        """Build the controls panel"""
        # Row 1: API Connection
        conn_frame = ttk.Frame(parent)
        conn_frame.pack(fill=tk.X, pady=5)

        ttk.Label(conn_frame, text="API Host:").pack(side=tk.LEFT, padx=5)
        self.host_entry = ttk.Entry(conn_frame, width=15)
        self.host_entry.insert(0, self.api_host)
        self.host_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(conn_frame, text="Port:").pack(side=tk.LEFT, padx=5)
        self.port_entry = ttk.Entry(conn_frame, width=8)
        self.port_entry.insert(0, str(self.api_port))
        self.port_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(conn_frame, text="Connect", command=self.update_connection).pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(conn_frame, text="Status: Disconnected", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Row 2: ROI Color Controls
        roi_frame = ttk.LabelFrame(parent, text="ROI Color Thresholds", padding=5)
        roi_frame.pack(fill=tk.X, pady=5)

        # Color picker button
        color_btn_frame = ttk.Frame(roi_frame)
        color_btn_frame.pack(side=tk.LEFT, padx=10)

        self.color_display = tk.Canvas(color_btn_frame, width=40, height=40, bg="cyan", relief=tk.SUNKEN, bd=2)
        self.color_display.pack(side=tk.LEFT, padx=5)
        self._update_color_display()

        ttk.Button(color_btn_frame, text="Pick Color", command=self.pick_color).pack(side=tk.LEFT, padx=5)

        # Tolerance sliders
        sliders_frame = ttk.Frame(roi_frame)
        sliders_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # H tolerance
        h_frame = ttk.Frame(sliders_frame)
        h_frame.pack(fill=tk.X, pady=2)
        ttk.Label(h_frame, text="H Tol:", width=8).pack(side=tk.LEFT)
        self.h_tol_label = ttk.Label(h_frame, text=f"{self.h_tol}", width=4)
        self.h_tol_scale = ttk.Scale(h_frame, from_=0, to=90, orient=tk.HORIZONTAL,
                                      command=self.on_h_tol_change)
        self.h_tol_scale.set(self.h_tol)
        self.h_tol_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.h_tol_label.pack(side=tk.LEFT)

        # S tolerance
        s_frame = ttk.Frame(sliders_frame)
        s_frame.pack(fill=tk.X, pady=2)
        ttk.Label(s_frame, text="S Tol:", width=8).pack(side=tk.LEFT)
        self.s_tol_label = ttk.Label(s_frame, text=f"{self.s_tol}", width=4)
        self.s_tol_scale = ttk.Scale(s_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                      command=self.on_s_tol_change)
        self.s_tol_scale.set(self.s_tol)
        self.s_tol_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.s_tol_label.pack(side=tk.LEFT)

        # V tolerance
        v_frame = ttk.Frame(sliders_frame)
        v_frame.pack(fill=tk.X, pady=2)
        ttk.Label(v_frame, text="V Tol:", width=8).pack(side=tk.LEFT)
        self.v_tol_label = ttk.Label(v_frame, text=f"{self.v_tol}", width=4)
        self.v_tol_scale = ttk.Scale(v_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                      command=self.on_v_tol_change)
        self.v_tol_scale.set(self.v_tol)
        self.v_tol_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.v_tol_label.pack(side=tk.LEFT)

        ttk.Button(roi_frame, text="Apply ROI Settings", command=self.apply_roi_settings).pack(side=tk.LEFT, padx=10)

        # Row 3: UI Controls
        ui_frame = ttk.Frame(parent)
        ui_frame.pack(fill=tk.X, pady=5)

        # Refresh rate control
        ttk.Label(ui_frame, text="UI Refresh (Hz):").pack(side=tk.LEFT, padx=5)
        self.refresh_label = ttk.Label(ui_frame, text=f"{self.ui_refresh_hz} Hz", width=8)
        self.refresh_scale = ttk.Scale(ui_frame, from_=1, to=60, orient=tk.HORIZONTAL,
                                        command=self.on_refresh_change)
        self.refresh_scale.set(self.ui_refresh_hz)
        self.refresh_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.refresh_label.pack(side=tk.LEFT, padx=5)

        # RealSense toggle
        self.rs_var = tk.BooleanVar(value=self.use_realsense)
        self.rs_toggle = ttk.Checkbutton(ui_frame, text="Use RealSense",
                                          variable=self.rs_var,
                                          command=self.toggle_realsense)
        self.rs_toggle.pack(side=tk.LEFT, padx=20)

        # Reverse RealSense toggle
        self.reverse_rs_var = tk.BooleanVar(value=self.use_reverse_rs)
        self.reverse_rs_toggle = ttk.Checkbutton(ui_frame, text="Reverse RS Mode",
                                                   variable=self.reverse_rs_var,
                                                   command=self.toggle_reverse_rs)
        self.reverse_rs_toggle.pack(side=tk.LEFT, padx=5)

        # Random pose toggle
        self.random_pose_var = tk.BooleanVar(value=False)
        self.random_pose_toggle = ttk.Checkbutton(ui_frame, text="Use Random Pose",
                                                   variable=self.random_pose_var,
                                                   command=self.toggle_random_pose)
        self.random_pose_toggle.pack(side=tk.LEFT, padx=5)

        # Pause/Resume button
        self.toggle_btn = ttk.Button(ui_frame, text="Pause Updates", command=self.toggle_updates)
        self.toggle_btn.pack(side=tk.LEFT, padx=5)

        # Refresh now button
        ttk.Button(ui_frame, text="Refresh Now", command=self.refresh_now).pack(side=tk.LEFT, padx=5)

        # RealSense panel button
        ttk.Button(ui_frame, text="Open RealSense Panel", command=self.open_rs_panel).pack(side=tk.LEFT, padx=5)

    def _build_image_grid(self, parent):
        """Build the image grid (2 rows x 3 columns)"""
        self.image_labels = {}
        image_titles = [
            ("RGB Feed", "rgb"),
            ("Disparity Feed", "disparity"),
            ("ArUco Detection", "aruco"),
            ("Binary Mask", "mask"),
            ("RGB with 6D Pose", "pose_overlay"),
            ("Final Clean ROI Mask", "clean_mask")
        ]

        for idx, (title, key) in enumerate(image_titles):
            row = idx // 3
            col = idx % 3

            frame = ttk.LabelFrame(parent, text=title, padding=5)
            frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

            label = ttk.Label(frame, text="Waiting for data...", anchor="center")
            label.pack(fill=tk.BOTH, expand=True)

            self.image_labels[key] = label

            parent.rowconfigure(row, weight=1)
            parent.columnconfigure(col, weight=1)

    def _build_info_panel(self, parent):
        """Build the info panel"""
        self.info_text = tk.Text(parent, height=10, wrap=tk.WORD, font=("Courier", 9))
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)

        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # ------------------ Control Handlers ------------------
    def update_connection(self):
        """Update API connection with new host/port"""
        try:
            new_host = self.host_entry.get().strip()
            new_port = int(self.port_entry.get().strip())

            self.api_host = new_host
            self.api_port = new_port
            self.client.update_base_url(new_host, new_port)

            if self.client.health_check():
                messagebox.showinfo("Connection", f"Successfully connected to {new_host}:{new_port}")
                if not self.running:
                    self.start_updates()
            else:
                messagebox.showerror("Connection", f"Failed to connect to {new_host}:{new_port}")
        except ValueError:
            messagebox.showerror("Error", "Invalid port number")

    def pick_color(self):
        """Open color picker dialog"""
        # Convert HSV to RGB for display
        h, s, v = self.hsv_center
        r, g, b = hsv_to_rgb(h, s, v)

        color = colorchooser.askcolor(
            title="Pick ROI Color",
            initialcolor=(r, g, b)
        )

        if color[0]:  # color[0] is RGB tuple
            r, g, b = color[0]
            # Convert RGB back to HSV
            rgb_array = np.uint8([[[r, g, b]]])
            hsv_array = cv.cvtColor(rgb_array, cv.COLOR_RGB2HSV)
            h, s, v = hsv_array[0, 0]

            self.hsv_center = [int(h), int(s), int(v)]
            self._update_color_display()

    def _update_color_display(self):
        """Update the color display canvas"""
        r, g, b = hsv_to_rgb(*self.hsv_center)
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        self.color_display.configure(bg=hex_color)

    def on_h_tol_change(self, value):
        """Handle H tolerance slider change"""
        self.h_tol = int(float(value))
        self.h_tol_label.config(text=f"{self.h_tol}")

    def on_s_tol_change(self, value):
        """Handle S tolerance slider change"""
        self.s_tol = int(float(value))
        self.s_tol_label.config(text=f"{self.s_tol}")

    def on_v_tol_change(self, value):
        """Handle V tolerance slider change"""
        self.v_tol = int(float(value))
        self.v_tol_label.config(text=f"{self.v_tol}")

    def on_refresh_change(self, value):
        """Handle refresh rate slider change"""
        self.ui_refresh_hz = int(float(value))
        self.refresh_label.config(text=f"{self.ui_refresh_hz} Hz")

    def apply_roi_settings(self):
        """Apply ROI settings to the API"""
        config_data = {
            'hsv_center': self.hsv_center,
            'h_tol': self.h_tol,
            's_tol': self.s_tol,
            'v_tol': self.v_tol
        }

        if self.client.update_config(config_data):
            messagebox.showinfo("Success", "ROI settings applied successfully")
        else:
            messagebox.showerror("Error", "Failed to apply ROI settings")

    def toggle_realsense(self):
        """Toggle RealSense mode and refresh pipeline"""
        self.use_realsense = self.rs_var.get()

        config_data = {
            'use_realsense': self.use_realsense
        }

        if self.client.update_config(config_data):
            mode = "RealSense" if self.use_realsense else "Transformers"
            messagebox.showinfo("RealSense", f"Switched to {mode} depth mode")
            # Pipeline will automatically refresh on next frame
        else:
            messagebox.showerror("Error", "Failed to toggle RealSense mode")
            # Revert checkbox
            self.rs_var.set(not self.use_realsense)

    def toggle_reverse_rs(self):
        """Toggle reverse RealSense mode"""
        self.use_reverse_rs = self.reverse_rs_var.get()

        if self.use_reverse_rs:
            # Check if RealSense is available
            try:
                response = self.client.session.get(f"{self.client.base_url}/stats", timeout=1)
                if response.status_code == 200:
                    stats = response.json()
                    rs_available = stats.get('cv_pipeline', {}).get('realsense_available', False)

                    if not rs_available:
                        messagebox.showwarning(
                            "RealSense Not Available",
                            "RealSense camera is not available. Reverse RS mode requires RealSense hardware."
                        )
                        self.reverse_rs_var.set(False)
                        self.use_reverse_rs = False
                        return

            except Exception as e:
                messagebox.showerror("Error", f"Failed to check RealSense availability: {e}")
                self.reverse_rs_var.set(False)
                self.use_reverse_rs = False
                return

            messagebox.showinfo(
                "Reverse RealSense Mode",
                "Reverse RS Mode enabled.\n\n"
                "In this mode:\n"
                "• AVP mask is transformed to RealSense view\n"
                "• RealSense RGB, intrinsics, and depth are used for pose estimation\n"
                "• Returned 6D pose is transformed back to AVP view\n\n"
                "Use this mode for pose estimation in RealSense native space."
            )
        else:
            messagebox.showinfo("Reverse RealSense Mode", "Reverse RS Mode disabled. Using standard AVP view.")

    def toggle_random_pose(self):
        """Toggle random pose mode"""
        use_random = self.random_pose_var.get()

        config_data = {
            'use_random_pose': use_random
        }

        if self.client.update_config(config_data):
            mode = "Random/Mock" if use_random else "Real API"
            messagebox.showinfo("Pose Mode", f"Switched to {mode} pose mode")
        else:
            messagebox.showerror("Error", "Failed to toggle pose mode")
            # Revert checkbox
            self.random_pose_var.set(not use_random)

    def open_rs_panel(self):
        """Open RealSense panel in a new window"""
        try:
            # Create new top-level window
            rs_window = tk.Toplevel(self.root)
            rs_window.title("RealSense Panel")
            rs_window.geometry("1200x800")

            # Create RealSense panel
            RSPanel(rs_window, self.client)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open RealSense panel: {e}")

    # ------------------ Update Management ------------------
    def start_updates(self):
        """Start background update thread"""
        if not self.running:
            self.running = True
            self.toggle_btn.config(text="Pause Updates")
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

    def stop_updates(self):
        """Stop background updates"""
        self.running = False
        self.toggle_btn.config(text="Resume Updates")

    def toggle_updates(self):
        """Toggle updates on/off"""
        if self.running:
            self.stop_updates()
        else:
            self.start_updates()

    def refresh_now(self):
        """Force immediate refresh"""
        self._fetch_and_display()

    def _update_loop(self):
        """Background update loop with dynamic refresh rate"""
        while self.running:
            try:
                update_interval = 1.0 / max(1, self.ui_refresh_hz)
                self._fetch_and_display()
                time.sleep(update_interval)
            except Exception as e:
                print(f"[ERROR] Update loop: {e}")
                time.sleep(1.0)

    def _fetch_and_display(self):
        """Fetch data from API and update display"""
        try:
            data = self.client.get_data_batch()
            config = self.client.get_config()
            stats = self.client.get_stats()

            # Update images
            self.root.after(0, self._update_images, data)

            # Update info
            self.root.after(0, self._update_info, data, config, stats)

            # Update status
            self.root.after(0, self._update_status, True)

        except Exception as e:
            print(f"[ERROR] _fetch_and_display: {e}")
            self.root.after(0, self._update_status, False)

    def _update_images(self, data):
        """Update all image displays"""
        try:
            # RGB frame
            if 'rgb_frame' in data:
                self._set_image('rgb', data['rgb_frame'])

                # Create RGB with 6D pose overlay
                if 'pose' in data and 'intrinsics' in data:
                    pose_overlay = draw_6d_pose_overlay(
                        data['rgb_frame'],
                        data['pose'],
                        data['intrinsics']
                    )
                    self._set_image('pose_overlay', pose_overlay)

            # Disparity with colormap
            if 'disparity' in data:
                disparity_colored = apply_colormap_for_depth(data['disparity'])
                if disparity_colored is not None:
                    self._set_image('disparity', disparity_colored)

            # ArUco detection frame
            if 'detected_frame' in data:
                self._set_image('aruco', data['detected_frame'])

            # Binary mask
            if 'mask' in data:
                self._set_image('mask', data['mask'])

                # Create clean ROI mask with Hough circle detection
                circle_params = detect_roi_circle_hough(data['mask'])
                if circle_params:
                    clean_mask = create_clean_roi_mask(data['mask'], circle_params)
                    self._set_image('clean_mask', clean_mask)
                else:
                    # If no circle detected, show original mask
                    self._set_image('clean_mask', data['mask'])

        except Exception as e:
            print(f"[ERROR] _update_images: {e}")
            import traceback
            traceback.print_exc()

    def _set_image(self, key, img_array):
        """Set image in label"""
        if img_array is None:
            return

        try:
            # Convert to PIL Image
            if img_array.ndim == 2:
                # Grayscale
                pil_img = Image.fromarray(img_array)
            elif img_array.shape[2] == 3:
                # RGB (OpenCV BGR to RGB)
                img_rgb = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
            else:
                return

            # Resize to fit display (larger for better visibility)
            pil_img.thumbnail((500, 350), Image.LANCZOS)

            # Create PhotoImage and display
            photo = ImageTk.PhotoImage(pil_img)
            self.image_labels[key].configure(image=photo, text="")
            self.photo_refs[key] = photo

        except Exception as e:
            print(f"[ERROR] _set_image({key}): {e}")

    def _update_info(self, data, config, stats):
        """Update info text panel with API data"""
        try:
            self.info_text.delete("1.0", tk.END)

            # API Connection
            self.info_text.insert(tk.END, "=== API CONNECTION ===\n")
            self.info_text.insert(tk.END, f"Host: {self.api_host}:{self.api_port}\n")
            self.info_text.insert(tk.END, f"UI Refresh: {self.ui_refresh_hz} Hz\n\n")

            # Configuration
            if config:
                use_realsense = config.get('use_realsense', False)

                self.info_text.insert(tk.END, "=== CONFIGURATION ===\n")
                self.info_text.insert(tk.END, f"Depth Mode: {'RealSense Hardware' if use_realsense else 'Transformers AI'}\n")
                self.info_text.insert(tk.END, f"HSV Center: {config.get('hsv_center', 'N/A')}\n")
                self.info_text.insert(tk.END, f"H Tolerance: {config.get('h_tol', 'N/A')}\n")
                self.info_text.insert(tk.END, f"S Tolerance: {config.get('s_tol', 'N/A')}\n")
                self.info_text.insert(tk.END, f"V Tolerance: {config.get('v_tol', 'N/A')}\n\n")

            # Statistics
            if stats:
                self.info_text.insert(tk.END, "=== STATISTICS ===\n")
                cv_stats = stats.get('cv_pipeline', {})
                self.info_text.insert(tk.END, f"Frames Processed: {cv_stats.get('frames_processed', 0)}\n")
                self.info_text.insert(tk.END, f"ArUco Detections: {cv_stats.get('aruco_detections', 0)}\n")
                self.info_text.insert(tk.END, f"Pose Successes: {cv_stats.get('pose_successes', 0)}\n")
                self.info_text.insert(tk.END, f"Device: {cv_stats.get('device', 'N/A')}\n")
                self.info_text.insert(tk.END, f"RealSense Available: {cv_stats.get('realsense_available', False)}\n")
                self.info_text.insert(tk.END, f"Selected Model: {stats.get('selected_model', 'N/A')}\n\n")

            # Camera Intrinsics
            if 'intrinsics' in data and data['intrinsics']:
                K = data['intrinsics'].get('K', [])
                if K:
                    self.info_text.insert(tk.END, "=== CAMERA INTRINSICS ===\n")
                    self.info_text.insert(tk.END, f"fx: {K[0][0]:.1f}, fy: {K[1][1]:.1f}\n")
                    self.info_text.insert(tk.END, f"cx: {K[0][2]:.1f}, cy: {K[1][2]:.1f}\n\n")

            # ArUco Pose
            if 'pose' in data and data['pose']:
                pose = data['pose']
                self.info_text.insert(tk.END, "=== ARUCO POSE (from AVP view) ===\n")
                self.info_text.insert(tk.END, f"Markers Detected: {pose.get('markers_detected', 0)}\n")
                if 'rvec' in pose:
                    rvec = pose['rvec']
                    self.info_text.insert(tk.END, f"Rotation: [{rvec[0]:.3f}, {rvec[1]:.3f}, {rvec[2]:.3f}]\n")
                if 'tvec' in pose:
                    tvec = pose['tvec']
                    self.info_text.insert(tk.END, f"Translation: [{tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}]\n")
                if 'timestamp' in pose:
                    fetch_time = data.get('fetch_time', 0)
                    age_ms = (fetch_time - pose['timestamp']) * 1000
                    self.info_text.insert(tk.END, f"Age: {age_ms:.1f} ms\n")
                self.info_text.insert(tk.END, "\n")

            # Head Pose
            if 'head_pose' in data and data['head_pose']:
                head_pose_data = data['head_pose'].get('head_pose', {})
                self.info_text.insert(tk.END, "=== HEAD POSE ===\n")
                if 'position' in head_pose_data:
                    pos = head_pose_data['position']
                    self.info_text.insert(tk.END, f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n")
                if 'rotation' in head_pose_data:
                    rot = head_pose_data['rotation']
                    self.info_text.insert(tk.END, f"Rotation: [{rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}]\n")
                if 'age_seconds' in data['head_pose']:
                    age = data['head_pose']['age_seconds']
                    self.info_text.insert(tk.END, f"Age: {age:.3f} s\n")

        except Exception as e:
            print(f"[ERROR] _update_info: {e}")
            import traceback
            traceback.print_exc()

    def _update_status(self, connected):
        """Update connection status"""
        if connected:
            self.status_label.config(text="Status: Connected ✓", foreground="green")
        else:
            self.status_label.config(text="Status: Disconnected ✗", foreground="red")

    def on_close(self):
        """Handle window close"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        self.root.destroy()

# ------------------ RealSense Panel ------------------
class RSPanel:
    """
    RealSense panel showing:
    - RGB feed
    - ArUco pattern detection feed
    - Disparity feed
    - Pattern pose information
    - Coordinate transformation
    """
    def __init__(self, window, api_client):
        self.window = window
        self.client = api_client
        self.running = False
        self.update_thread = None
        self.photo_refs = {}

        # Try to import RealSense adapter
        try:
            from realsense_adapter_adjusted import RealSenseToAVPAligner
            self.rs_adapter = RealSenseToAVPAligner()
            self.rs_available = self.rs_adapter.available
        except Exception as e:
            print(f"[WARNING] RealSense adapter not available: {e}")
            self.rs_adapter = None
            self.rs_available = False

        self._build_ui()

        if not self.rs_available:
            messagebox.showwarning("RealSense", "RealSense camera not available")
        else:
            self.start_updates()

        window.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        """Build RealSense panel UI"""
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Image grid (2x2)
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(fill=tk.BOTH, expand=True)

        self.image_labels = {}
        image_titles = [
            ("RGB Feed", "rgb"),
            ("ArUco Detection", "aruco"),
            ("Disparity Feed", "disparity"),
            ("Info", "info")
        ]

        for idx, (title, key) in enumerate(image_titles):
            row = idx // 2
            col = idx % 2

            frame = ttk.LabelFrame(images_frame, text=title, padding=5)
            frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

            if key == "info":
                # Info text widget
                text_widget = tk.Text(frame, height=15, wrap=tk.WORD, font=("Courier", 9))
                text_widget.pack(fill=tk.BOTH, expand=True)
                self.image_labels[key] = text_widget
            else:
                label = ttk.Label(frame, text="Waiting for data...", anchor="center")
                label.pack(fill=tk.BOTH, expand=True)
                self.image_labels[key] = label

            images_frame.rowconfigure(row, weight=1)
            images_frame.columnconfigure(col, weight=1)

        # Control buttons
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)

        ttk.Button(controls_frame, text="Refresh", command=self.refresh_now).pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(controls_frame, text="Status: Initializing...")
        self.status_label.pack(side=tk.LEFT, padx=20)

    def start_updates(self):
        """Start background update thread"""
        if not self.running and self.rs_available:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

    def stop_updates(self):
        """Stop background updates"""
        self.running = False

    def refresh_now(self):
        """Force immediate refresh"""
        self._fetch_and_display()

    def _update_loop(self):
        """Background update loop"""
        while self.running:
            try:
                self._fetch_and_display()
                time.sleep(0.15)  # ~6-7 Hz
            except Exception as e:
                print(f"[ERROR] RS Panel update loop: {e}")
                time.sleep(1.0)

    def _fetch_and_display(self):
        """Fetch data and update display"""
        try:
            # Get RealSense data
            rs_data = None
            if self.rs_adapter:
                rs_data = self.rs_adapter.capture_and_align()

            # Get API data
            api_data = self.client.get_data_batch()
            stats = self.client.get_stats()

            # Update display
            self.window.after(0, self._update_display, rs_data, api_data, stats)

        except Exception as e:
            print(f"[ERROR] RS Panel fetch: {e}")

    def _update_display(self, rs_data, api_data, stats):
        """Update display with fetched data"""
        try:
            # RGB feed from RealSense
            if rs_data and 'color' in rs_data:
                self._set_image('rgb', rs_data['color'])

            # Disparity from RealSense
            if rs_data and 'aligned_disparity' in rs_data:
                disp = rs_data['aligned_disparity']
                if disp is not None:
                    disp_colored = apply_colormap_for_depth(disp)
                    self._set_image('disparity', disp_colored)

            # ArUco detection from API
            if 'detected_frame' in api_data:
                self._set_image('aruco', api_data['detected_frame'])

            # Info panel
            info_text = self.image_labels['info']
            info_text.delete("1.0", tk.END)

            info_text.insert(tk.END, "=== REALSENSE DATA ===\n")
            if rs_data:
                info_text.insert(tk.END, f"Timestamp: {time.strftime('%H:%M:%S', time.localtime(rs_data.get('timestamp', 0)))}\n")
                info_text.insert(tk.END, f"Color: {'Available' if rs_data.get('color') is not None else 'N/A'}\n")
                info_text.insert(tk.END, f"Disparity: {'Available' if rs_data.get('aligned_disparity') is not None else 'N/A'}\n")
                info_text.insert(tk.END, f"Depth: {'Available' if rs_data.get('aligned_depth') is not None else 'N/A'}\n")
                if self.rs_adapter:
                    info_text.insert(tk.END, f"Extrinsics: {'Available' if self.rs_adapter.R_avp_rs_c is not None else 'N/A'}\n")
            else:
                info_text.insert(tk.END, "No RealSense data\n")

            info_text.insert(tk.END, "\n=== ARUCO PATTERN POSE ===\n")
            if 'pose' in api_data and api_data['pose']:
                pose = api_data['pose']
                info_text.insert(tk.END, f"Markers Detected: {pose.get('markers_detected', 0)}\n")
                if 'rvec' in pose:
                    rvec = pose['rvec']
                    info_text.insert(tk.END, f"Rotation Vector: [{rvec[0]:.4f}, {rvec[1]:.4f}, {rvec[2]:.4f}]\n")
                if 'tvec' in pose:
                    tvec = pose['tvec']
                    info_text.insert(tk.END, f"Translation Vector: [{tvec[0]:.4f}, {tvec[1]:.4f}, {tvec[2]:.4f}] m\n")
            else:
                info_text.insert(tk.END, "No pose data available\n")

            info_text.insert(tk.END, "\n=== COORDINATE TRANSFORMATION ===\n")
            if self.rs_adapter and self.rs_adapter.R_avp_rs_c is not None:
                R = self.rs_adapter.R_avp_rs_c
                t = self.rs_adapter.t_avp_rs_c if hasattr(self.rs_adapter, 't_avp_rs_c') else [0, 0, 0]
                info_text.insert(tk.END, "Rotation Matrix (AVP <- RS):\n")
                for i in range(3):
                    info_text.insert(tk.END, f"  [{R[i,0]:7.4f} {R[i,1]:7.4f} {R[i,2]:7.4f}]\n")
                info_text.insert(tk.END, f"Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]\n")
            else:
                info_text.insert(tk.END, "Transformation not calibrated\n")

            # Update status
            if rs_data:
                self.status_label.config(text="Status: Running ✓", foreground="green")
            else:
                self.status_label.config(text="Status: No Data", foreground="orange")

        except Exception as e:
            print(f"[ERROR] RS Panel display update: {e}")
            import traceback
            traceback.print_exc()

    def _set_image(self, key, img_array):
        """Set image in label"""
        if img_array is None:
            return

        try:
            # Convert to PIL Image
            if img_array.ndim == 2:
                pil_img = Image.fromarray(img_array)
            elif img_array.shape[2] == 3:
                img_rgb = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
            else:
                return

            # Resize to fit display
            pil_img.thumbnail((500, 400), Image.LANCZOS)

            # Create PhotoImage and display
            photo = ImageTk.PhotoImage(pil_img)
            self.image_labels[key].configure(image=photo, text="")
            self.photo_refs[key] = photo

        except Exception as e:
            print(f"[ERROR] RS Panel _set_image({key}): {e}")

    def on_close(self):
        """Handle window close"""
        self.running = False
        if self.rs_adapter:
            try:
                self.rs_adapter.stop()
            except Exception:
                pass
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        self.window.destroy()

# ------------------ Main ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedDebugViewer(root)
    root.mainloop()
