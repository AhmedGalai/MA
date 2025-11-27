#!/usr/bin/env python3
"""
Unified AVP API Debug Viewer
Enhanced debug client that displays API pipeline results with proper depth scaling.
Supports both RealSense and Transformers-based depth visualization.

FEATURES:
- Proper depth/disparity scaling for visibility (0-255 normalization for display)
- Shows timestamps for all data
- Indicates whether RealSense or Transformers depth is being used
- Color-coded depth maps for better visualization
- Real-time sync status monitoring
"""

import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
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
    DEFAULT_UI_HZ = APP_CONFIG.get("defaults", {}).get("ui_refresh_hz", 15)
except Exception:
    API_BASE_URL = "http://localhost:5000"
    DEFAULT_UI_HZ = 15

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
    """
    Normalize depth/disparity array to 0-255 range for proper display.

    This is crucial for tk_debugging_window display as raw depth values
    might be in different ranges (0-1 for normalized, 0-10000 for mm, etc.)
    """
    if img_array is None or img_array.size == 0:
        return None

    # Convert to float for processing
    if img_array.dtype != np.float32:
        img_array = img_array.astype(np.float32)

    # Normalize to 0-255 range
    img_min = img_array.min()
    img_max = img_array.max()

    if img_max - img_min < 1e-6:
        # Constant image
        return np.zeros_like(img_array, dtype=np.uint8)

    normalized = ((img_array - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    return normalized

def apply_colormap_for_depth(img_array, colormap=cv.COLORMAP_TURBO):
    """
    Apply colormap to depth/disparity for better visualization.

    Input can be raw depth values - this function handles normalization.
    """
    normalized = normalize_for_display(img_array)
    if normalized is None:
        return None

    # Apply colormap
    colored = cv.applyColorMap(normalized, colormap)
    return colored

# ------------------ API Client ------------------
class UnifiedAPIClient:
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

    def get_stats(self):
        """Get pipeline statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=1)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def get_data_batch(self):
        """Fetch all display data in one batch with timestamps"""
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

            # Get depth (PNG-encoded)
            try:
                r = self.session.get(f"{self.base_url}/depth", timeout=0.5)
                if r.status_code == 200:
                    data = r.json()
                    img_array = decode_base64_image(data['depth'])
                    if img_array is not None:
                        results['depth'] = img_array
                        results['depth_timestamp'] = data.get('timestamp')
            except Exception as e:
                print(f"[DEBUG] depth error: {e}")

            # Get disparity (PNG-encoded)
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

            results['fetch_time'] = fetch_time
            return results

        except Exception as e:
            print(f"[ERROR] get_data_batch: {e}")
            return {}

# ------------------ Main UI ------------------
class UnifiedDebugViewer:
    def __init__(self, root):
        self.root = root
        root.title("Unified AVP Pipeline Debug Viewer")
        root.geometry("1600x900")

        self.client = UnifiedAPIClient(API_BASE_URL)
        self.running = False
        self.update_thread = None
        self.photo_refs = {}  # Keep references to prevent GC

        # Build UI
        self._build_ui()

        # Check API connection
        if not self.client.health_check():
            messagebox.showwarning(
                "API Connection",
                f"Cannot connect to API at {API_BASE_URL}\n"
                "Make sure main_api.py is running."
            )
        else:
            self.start_updates()

        # Handle window close
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        """Build the UI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top: Image grid (2x3)
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(fill=tk.BOTH, expand=True)

        self.image_labels = {}
        image_titles = [
            ("RGB Frame", "rgb"),
            ("Mask", "mask"),
            ("Depth (Normalized)", "depth"),
            ("Disparity (Normalized)", "disparity"),
            ("Depth Colormap", "depth_color"),
            ("Disparity Colormap", "disparity_color")
        ]

        for idx, (title, key) in enumerate(image_titles):
            row = idx // 3
            col = idx % 3

            frame = ttk.LabelFrame(images_frame, text=title, padding=5)
            frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

            label = ttk.Label(frame, text="Waiting for data...", anchor="center")
            label.pack(fill=tk.BOTH, expand=True)

            self.image_labels[key] = label

            images_frame.rowconfigure(row, weight=1)
            images_frame.columnconfigure(col, weight=1)

        # Bottom: Info panel
        info_frame = ttk.LabelFrame(main_frame, text="Pipeline Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(10, 0))

        self.info_text = tk.Text(info_frame, height=12, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(10, 0))

        self.toggle_btn = ttk.Button(
            controls_frame,
            text="Pause Updates",
            command=self.toggle_updates
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=5)

        refresh_btn = ttk.Button(
            controls_frame,
            text="Refresh Now",
            command=self.refresh_now
        )
        refresh_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(controls_frame, text="Status: Connecting...")
        self.status_label.pack(side=tk.RIGHT, padx=5)

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
        """Background update loop"""
        update_interval = 1.0 / DEFAULT_UI_HZ

        while self.running:
            try:
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
        """Update image displays"""
        try:
            # RGB frame
            if 'rgb_frame' in data:
                self._set_image('rgb', data['rgb_frame'])

            # Mask
            if 'mask' in data:
                self._set_image('mask', data['mask'])

            # Depth (normalized for display)
            if 'depth' in data:
                depth_normalized = normalize_for_display(data['depth'])
                self._set_image('depth', depth_normalized)

                # Depth colormap
                depth_colored = apply_colormap_for_depth(data['depth'])
                self._set_image('depth_color', depth_colored)

            # Disparity (normalized for display)
            if 'disparity' in data:
                disparity_normalized = normalize_for_display(data['disparity'])
                self._set_image('disparity', disparity_normalized)

                # Disparity colormap
                disparity_colored = apply_colormap_for_depth(data['disparity'])
                self._set_image('disparity_color', disparity_colored)

        except Exception as e:
            print(f"[ERROR] _update_images: {e}")

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

            # Resize to fit display
            pil_img.thumbnail((400, 300), Image.LANCZOS)

            # Create PhotoImage and display
            photo = ImageTk.PhotoImage(pil_img)
            self.image_labels[key].configure(image=photo, text="")
            self.photo_refs[key] = photo  # Keep reference

        except Exception as e:
            print(f"[ERROR] _set_image({key}): {e}")

    def _update_info(self, data, config, stats):
        """Update info text panel"""
        try:
            self.info_text.delete("1.0", tk.END)

            # Configuration
            if config:
                use_realsense = config.get('use_realsense', False)
                use_random_pose = config.get('use_random_pose', True)

                self.info_text.insert(tk.END, "=== CONFIGURATION ===\n")
                self.info_text.insert(tk.END, f"Depth Mode: {'RealSense Hardware' if use_realsense else 'Transformers AI'}\n")
                self.info_text.insert(tk.END, f"Pose Mode: {'Mock/Random' if use_random_pose else 'Real API'}\n")
                self.info_text.insert(tk.END, f"HSV Center: {config.get('hsv_center', 'N/A')}\n")
                self.info_text.insert(tk.END, "\n")

            # Statistics
            if stats:
                self.info_text.insert(tk.END, "=== STATISTICS ===\n")
                cv_stats = stats.get('cv_pipeline', {})
                self.info_text.insert(tk.END, f"Frames Processed: {cv_stats.get('frames_processed', 0)}\n")
                self.info_text.insert(tk.END, f"ArUco Detections: {cv_stats.get('aruco_detections', 0)}\n")
                self.info_text.insert(tk.END, f"Pose Successes: {cv_stats.get('pose_successes', 0)}\n")
                self.info_text.insert(tk.END, f"Device: {cv_stats.get('device', 'N/A')}\n")
                self.info_text.insert(tk.END, f"RealSense Available: {cv_stats.get('realsense_available', False)}\n")
                self.info_text.insert(tk.END, "\n")

            # Current frame data timestamps
            self.info_text.insert(tk.END, "=== TIMESTAMPS (Sync Check) ===\n")
            fetch_time = data.get('fetch_time', 0)

            if 'depth_timestamp' in data and data['depth_timestamp']:
                age_ms = (fetch_time - data['depth_timestamp']) * 1000
                self.info_text.insert(tk.END, f"Depth Age: {age_ms:.1f} ms\n")

            if 'disparity_timestamp' in data and data['disparity_timestamp']:
                age_ms = (fetch_time - data['disparity_timestamp']) * 1000
                self.info_text.insert(tk.END, f"Disparity Age: {age_ms:.1f} ms\n")

            # Pose info
            if 'pose' in data and data['pose']:
                pose = data['pose']
                self.info_text.insert(tk.END, "\n=== POSE DATA ===\n")
                self.info_text.insert(tk.END, f"Markers Detected: {pose.get('markers_detected', 0)}\n")
                if 'timestamp' in pose:
                    age_ms = (fetch_time - pose['timestamp']) * 1000
                    self.info_text.insert(tk.END, f"Pose Age: {age_ms:.1f} ms\n")

            # Intrinsics
            if 'intrinsics' in data and data['intrinsics']:
                K = data['intrinsics'].get('K', [])
                if K:
                    self.info_text.insert(tk.END, "\n=== CAMERA INTRINSICS ===\n")
                    self.info_text.insert(tk.END, f"fx: {K[0][0]:.1f}, fy: {K[1][1]:.1f}\n")
                    self.info_text.insert(tk.END, f"cx: {K[0][2]:.1f}, cy: {K[1][2]:.1f}\n")

        except Exception as e:
            print(f"[ERROR] _update_info: {e}")

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

# ------------------ Main ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedDebugViewer(root)
    root.mainloop()
