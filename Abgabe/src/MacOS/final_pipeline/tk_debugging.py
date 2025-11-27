#!/usr/bin/env python3
"""
Final Pipeline Debug Viewer
Streamlined debugging client for the new final pipeline:
- RealSense RGB feed
- RealSense Depth feed (metric)
- AVP Mask feed
- RS Pose Overlay (6D pose in RealSense view)
- AVP RGB feed (captured screen)
- AVP Pose Overlay (transformed pose in AVP view)
- Pipeline statistics
- Save frame functionality
- Pose API test
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
import os
import json
from datetime import datetime

# ------------------ Configuration ------------------
DEFAULT_FINAL_API_HOST = "localhost"
DEFAULT_FINAL_API_PORT = 5001  # Final pipeline API port
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

def encode_image_to_base64(img_array):
    """Encode numpy array to base64 string"""
    try:
        if len(img_array.shape) == 2:
            # Grayscale
            pil_img = Image.fromarray(img_array)
        else:
            # RGB/BGR
            if img_array.shape[2] == 3:
                rgb = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            else:
                pil_img = Image.fromarray(img_array)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"[ERROR] encode_image_to_base64: {e}")
        return None

def draw_6d_pose_overlay(rgb_frame, pose_data, intrinsics_K):
    """
    Draw 6D pose overlay on RGB frame

    Args:
        rgb_frame: RGB image as numpy array
        pose_data: Pose data with 'rvec' and 'tvec'
        intrinsics_K: Camera intrinsics matrix (3x3)

    Returns:
        Frame with pose overlay drawn
    """
    if pose_data is None or intrinsics_K is None:
        return rgb_frame

    try:
        frame = rgb_frame.copy()

        # Extract pose
        rvec = np.array(pose_data.get('rvec', [0, 0, 0]), dtype=np.float32).reshape(3, 1)
        tvec = np.array(pose_data.get('tvec', [0, 0, 0]), dtype=np.float32).reshape(3, 1)

        # Define axis points in 3D (length = 0.1m)
        axis_length = 0.1
        axis_points_3d = np.float32([
            [0, 0, 0],  # Origin
            [axis_length, 0, 0],  # X-axis (red)
            [0, axis_length, 0],  # Y-axis (green)
            [0, 0, axis_length]   # Z-axis (blue)
        ])

        # Project 3D points to 2D
        img_points, _ = cv.projectPoints(axis_points_3d, rvec, tvec, intrinsics_K, np.zeros(5))
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

        # Draw confidence text
        confidence = pose_data.get('confidence', 0)
        cv.putText(frame, f"Conf: {confidence:.2f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame
    except Exception as e:
        print(f"[ERROR] draw_6d_pose_overlay: {e}")
        return rgb_frame


# ------------------ API Client ------------------
class FinalPipelineClient:
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

    def get_stats(self):
        """Get pipeline statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=1)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def process_frame(self, mask_base64, headset_pose=None, rgb_base64=None):
        """Process frame through pipeline"""
        try:
            data = {"mask": mask_base64}
            if headset_pose:
                data["headset_pose"] = headset_pose
            if rgb_base64:
                data["rgb"] = rgb_base64

            response = self.session.post(f"{self.base_url}/process", json=data, timeout=5)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"[ERROR] process_frame: {e}")
            return None

    def get_avp_frame(self):
        """Get latest AVP frame"""
        try:
            response = self.session.get(f"{self.base_url}/avp_frame", timeout=1)
            if response.status_code == 200:
                data = response.json()
                return decode_base64_image(data.get('frame'))
            return None
        except Exception:
            return None

    def get_mask(self):
        """Get latest mask"""
        try:
            response = self.session.get(f"{self.base_url}/mask", timeout=1)
            if response.status_code == 200:
                data = response.json()
                return decode_base64_image(data.get('mask'))
            return None
        except Exception:
            return None

    def get_pose_result(self):
        """Get latest pose estimation result"""
        try:
            response = self.session.get(f"{self.base_url}/pose_result", timeout=1)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None


# ------------------ Main Debugging Window ------------------
class FinalPipelineDebugger:
    def __init__(self, root):
        self.root = root
        self.root.title("Final Pipeline Debugger")
        self.root.geometry("1400x1100")  # Increased height for 3x2 grid

        # API connection
        self.api_host = DEFAULT_FINAL_API_HOST
        self.api_port = DEFAULT_FINAL_API_PORT
        self.client = FinalPipelineClient(f"http://{self.api_host}:{self.api_port}")

        # State
        self.running = False
        self.ui_refresh_hz = DEFAULT_UI_HZ
        self.save_next_frame = False

        # Data cache
        self.last_rgb = None
        self.last_depth = None
        self.last_mask = None
        self.last_pose = None
        self.last_stats = None
        self.last_avp_rgb = None
        self.last_pose_result = None

        # Build UI
        self._build_ui()

        # Start updates
        self.start_updates()

    def _build_ui(self):
        """Build the complete UI"""
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top: Controls
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        self._build_controls(control_frame)

        # Middle: Image Grid
        image_frame = ttk.Frame(main_container)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self._build_image_grid(image_frame)

        # Bottom: Info Panel
        info_frame = ttk.LabelFrame(main_container, text="Pipeline Info", padding="5")
        info_frame.pack(fill=tk.X)
        self._build_info_panel(info_frame)

    def _build_controls(self, parent):
        """Build control panel"""
        # Row 1: Connection
        conn_frame = ttk.LabelFrame(parent, text="API Connection", padding="5")
        conn_frame.pack(fill=tk.X, pady=5)

        ttk.Label(conn_frame, text="Host:").pack(side=tk.LEFT, padx=5)
        self.host_entry = ttk.Entry(conn_frame, width=15)
        self.host_entry.insert(0, self.api_host)
        self.host_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(conn_frame, text="Port:").pack(side=tk.LEFT, padx=5)
        self.port_entry = ttk.Entry(conn_frame, width=6)
        self.port_entry.insert(0, str(self.api_port))
        self.port_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(conn_frame, text="Connect", command=self.update_connection).pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(conn_frame, text="●", foreground="gray", font=("Arial", 16))
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Row 2: UI Controls
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

        # Pause/Resume button
        self.toggle_btn = ttk.Button(ui_frame, text="Pause Updates", command=self.toggle_updates)
        self.toggle_btn.pack(side=tk.LEFT, padx=10)

        # Refresh now button
        ttk.Button(ui_frame, text="Refresh Now", command=self.refresh_now).pack(side=tk.LEFT, padx=5)

        # Save next frame button
        ttk.Button(ui_frame, text="Save Next Frame", command=self.trigger_save_frame).pack(side=tk.LEFT, padx=5)

        # Test pose API button
        ttk.Button(ui_frame, text="Test Pose API", command=self.test_pose_api).pack(side=tk.LEFT, padx=5)

    def _build_image_grid(self, parent):
        """Build the image grid (3 rows x 2 columns)"""
        self.image_labels = {}
        image_titles = [
            ("RealSense RGB", "rgb"),
            ("RealSense Depth", "depth"),
            ("AVP Mask", "mask"),
            ("RS Pose Overlay", "pose_overlay_rs"),
            ("AVP RGB Feed", "avp_rgb"),
            ("AVP Pose Overlay", "avp_pose_overlay")
        ]

        for idx, (title, key) in enumerate(image_titles):
            row = idx // 2
            col = idx % 2

            frame = ttk.LabelFrame(parent, text=title, padding=5)
            frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

            label = ttk.Label(frame, text="Waiting for data...", anchor="center")
            label.pack(fill=tk.BOTH, expand=True)

            self.image_labels[key] = label

            parent.rowconfigure(row, weight=1)
            parent.columnconfigure(col, weight=1)

    def _build_info_panel(self, parent):
        """Build the info panel"""
        self.info_text = tk.Text(parent, height=8, wrap=tk.WORD, font=("Courier", 9))
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

    def on_refresh_change(self, value):
        """Handle refresh rate slider change"""
        self.ui_refresh_hz = int(float(value))
        self.refresh_label.config(text=f"{self.ui_refresh_hz} Hz")

    def toggle_updates(self):
        """Toggle auto-updates"""
        if self.running:
            self.running = False
            self.toggle_btn.config(text="Resume Updates")
        else:
            self.running = True
            self.toggle_btn.config(text="Pause Updates")
            threading.Thread(target=self.update_loop, daemon=True).start()

    def refresh_now(self):
        """Manually refresh data"""
        self.fetch_and_display_data()

    def trigger_save_frame(self):
        """Trigger save on next frame"""
        self.save_next_frame = True
        messagebox.showinfo("Save Frame", "Next frame will be saved")

    def test_pose_api(self):
        """Test pose API with dummy data"""
        # Create test mask
        test_mask = np.zeros((480, 640), dtype=np.uint8)
        test_mask[200:300, 300:400] = 255

        # Encode mask
        mask_base64 = encode_image_to_base64(test_mask)

        # Test pose
        test_pose = {
            "position": [0.0, 0.0, 0.5],
            "rotation": [0.0, 0.0, 0.0]
        }

        # Send request
        print("[TEST] Sending test pose request...")
        result = self.client.process_frame(mask_base64, headset_pose=test_pose)

        if result and result.get('success'):
            messagebox.showinfo("Pose API Test",
                f"Success!\nConfidence: {result.get('confidence', 0):.3f}\n" +
                f"Processing time: {result.get('processing_time_ms', 0):.1f}ms")
            print(f"[TEST] Result: {result}")
        else:
            error = result.get('error', 'Unknown error') if result else 'No response'
            messagebox.showerror("Pose API Test", f"Failed: {error}")
            print(f"[TEST] Error: {error}")

    # ------------------ Update Loop ------------------
    def start_updates(self):
        """Start the update loop"""
        if not self.running:
            self.running = True
            self.toggle_btn.config(text="Pause Updates")
            threading.Thread(target=self.update_loop, daemon=True).start()

    def update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                self.fetch_and_display_data()
                time.sleep(1.0 / self.ui_refresh_hz)
            except Exception as e:
                print(f"[ERROR] update_loop: {e}")
                time.sleep(1.0)

    def fetch_and_display_data(self):
        """Fetch data from RealSense and display"""
        try:
            # Check connection
            if self.client.health_check():
                self.status_label.config(foreground="green")
            else:
                self.status_label.config(foreground="red")
                return

            # Get statistics
            stats = self.client.get_stats()
            if stats:
                self.last_stats = stats
                self.update_info_panel()

            # Get AVP frame from main API
            avp_frame = self.client.get_avp_frame()
            if avp_frame is not None:
                self.last_avp_rgb = avp_frame
                self.update_avp_rgb_display(avp_frame)

            # Get mask from main API
            mask = self.client.get_mask()
            if mask is not None:
                self.last_mask = mask
                self.update_mask_display(mask)

            # Get pose result from main API
            pose_result = self.client.get_pose_result()
            if pose_result:
                self.last_pose_result = pose_result

            # For visualization, we need to get RealSense data directly
            # Since the API doesn't stream video, we'll show stats only
            # To show actual video, integrate with RealSense camera directly

            # Try to import RealSense module from final pipeline
            try:
                import sys
                import os
                pipeline_path = os.path.join(os.path.dirname(__file__))
                if pipeline_path not in sys.path:
                    sys.path.insert(0, pipeline_path)

                from realsense_depth import RealSenseDepth

                if not hasattr(self, 'realsense'):
                    self.realsense = RealSenseDepth()

                if self.realsense.available:
                    # Capture frame
                    rs_data = self.realsense.capture_frame()
                    if rs_data:
                        self.last_rgb = rs_data['rgb']
                        self.last_depth = rs_data['depth']

                        # Update displays
                        self.update_rgb_display(self.last_rgb)
                        self.update_depth_display(self.last_depth)

                        # Update pose overlay (RealSense view)
                        if self.last_pose_result and self.last_pose_result.get('success'):
                            pose_rs = self.last_pose_result.get('pose_rs_view')
                            if pose_rs:
                                self.update_pose_overlay_rs(self.last_rgb, pose_rs, rs_data['intrinsics'])

                        # Update AVP pose overlay
                        if self.last_avp_rgb is not None and self.last_pose_result and self.last_pose_result.get('success'):
                            pose_avp = self.last_pose_result.get('pose_avp_view')
                            if pose_avp:
                                self.update_avp_pose_overlay(self.last_avp_rgb, pose_avp)

                        # Save frame if requested
                        if self.save_next_frame:
                            self.save_frame_data(rs_data)
                            self.save_next_frame = False

            except Exception as e:
                print(f"[WARNING] RealSense direct access failed: {e}")

        except Exception as e:
            print(f"[ERROR] fetch_and_display_data: {e}")

    def update_rgb_display(self, rgb_frame):
        """Update RGB display"""
        if rgb_frame is None:
            return

        try:
            # Convert BGR to RGB
            rgb = cv.cvtColor(rgb_frame, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((640, 480))
            photo = ImageTk.PhotoImage(pil_img)

            self.image_labels["rgb"].config(image=photo, text="")
            self.image_labels["rgb"].image = photo
        except Exception as e:
            print(f"[ERROR] update_rgb_display: {e}")

    def update_depth_display(self, depth_frame):
        """Update depth display"""
        if depth_frame is None:
            return

        try:
            # Normalize depth to 0-255
            depth_normalized = cv.normalize(depth_frame, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv.applyColorMap(depth_normalized, cv.COLORMAP_JET)
            depth_rgb = cv.cvtColor(depth_colored, cv.COLOR_BGR2RGB)

            pil_img = Image.fromarray(depth_rgb)
            pil_img.thumbnail((640, 480))
            photo = ImageTk.PhotoImage(pil_img)

            self.image_labels["depth"].config(image=photo, text="")
            self.image_labels["depth"].image = photo
        except Exception as e:
            print(f"[ERROR] update_depth_display: {e}")

    def update_mask_display(self, mask):
        """Update mask display"""
        if mask is None:
            return

        try:
            # Convert mask to RGB for display
            if len(mask.shape) == 2:
                mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
            else:
                mask_rgb = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

            pil_img = Image.fromarray(mask_rgb)
            pil_img.thumbnail((640, 480))
            photo = ImageTk.PhotoImage(pil_img)

            self.image_labels["mask"].config(image=photo, text="")
            self.image_labels["mask"].image = photo
        except Exception as e:
            print(f"[ERROR] update_mask_display: {e}")

    def update_avp_rgb_display(self, avp_frame):
        """Update AVP RGB display"""
        if avp_frame is None:
            return

        try:
            # Convert BGR to RGB if needed
            if len(avp_frame.shape) == 3 and avp_frame.shape[2] == 3:
                rgb = cv.cvtColor(avp_frame, cv.COLOR_BGR2RGB)
            else:
                rgb = avp_frame

            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((640, 480))
            photo = ImageTk.PhotoImage(pil_img)

            self.image_labels["avp_rgb"].config(image=photo, text="")
            self.image_labels["avp_rgb"].image = photo
        except Exception as e:
            print(f"[ERROR] update_avp_rgb_display: {e}")

    def update_pose_overlay_rs(self, rgb_frame, pose_rs, intrinsics):
        """Update RealSense pose overlay"""
        if rgb_frame is None or pose_rs is None:
            return

        try:
            # Get camera matrix
            K = np.array(intrinsics['K'], dtype=np.float32) if isinstance(intrinsics, dict) else intrinsics

            # Draw overlay
            overlayed = draw_6d_pose_overlay(rgb_frame, pose_rs, K)

            # Convert BGR to RGB
            rgb = cv.cvtColor(overlayed, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((640, 480))
            photo = ImageTk.PhotoImage(pil_img)

            self.image_labels["pose_overlay_rs"].config(image=photo, text="")
            self.image_labels["pose_overlay_rs"].image = photo
        except Exception as e:
            print(f"[ERROR] update_pose_overlay_rs: {e}")

    def update_avp_pose_overlay(self, avp_frame, pose_avp):
        """Update AVP pose overlay with transformed pose"""
        if avp_frame is None or pose_avp is None:
            return

        try:
            # Estimate camera matrix from frame size
            h, w = avp_frame.shape[:2]
            K_avp = self._estimate_camera_matrix(w, h)

            # Draw overlay
            overlayed = draw_6d_pose_overlay(avp_frame, pose_avp, K_avp)

            # Convert BGR to RGB if needed
            if len(overlayed.shape) == 3 and overlayed.shape[2] == 3:
                rgb = cv.cvtColor(overlayed, cv.COLOR_BGR2RGB)
            else:
                rgb = overlayed

            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((640, 480))
            photo = ImageTk.PhotoImage(pil_img)

            self.image_labels["avp_pose_overlay"].config(image=photo, text="")
            self.image_labels["avp_pose_overlay"].image = photo
        except Exception as e:
            print(f"[ERROR] update_avp_pose_overlay: {e}")

    def _estimate_camera_matrix(self, width, height):
        """Estimate camera matrix from image dimensions"""
        # Assume typical focal length ~70% of image width
        fx = fy = width * 0.7
        cx = width / 2.0
        cy = height / 2.0

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K

    def update_info_panel(self):
        """Update the info panel with statistics"""
        if not self.last_stats:
            return

        try:
            self.info_text.delete(1.0, tk.END)

            info = f"""
╔══════════════════════════════════════════════════════════════╗
║                  FINAL PIPELINE STATUS                       ║
╚══════════════════════════════════════════════════════════════╝

Pipeline Status:
  • Calibrated:         {self.last_stats.get('calibrated', False)}
  • RealSense Available: {self.last_stats.get('realsense_available', False)}

Statistics:
  • Frames Processed:    {self.last_stats.get('frames_processed', 0)}
  • Successful Poses:    {self.last_stats.get('successful_poses', 0)}
  • Failed Poses:        {self.last_stats.get('failed_poses', 0)}
  • Avg Processing Time: {self.last_stats.get('avg_processing_time_ms', 0):.1f} ms

Success Rate: {self._compute_success_rate()}%
"""

            self.info_text.insert(1.0, info)
        except Exception as e:
            print(f"[ERROR] update_info_panel: {e}")

    def _compute_success_rate(self):
        """Compute success rate percentage"""
        if not self.last_stats:
            return 0

        total = self.last_stats.get('successful_poses', 0) + self.last_stats.get('failed_poses', 0)
        if total == 0:
            return 0

        return int((self.last_stats.get('successful_poses', 0) / total) * 100)

    def save_frame_data(self, rs_data):
        """Save current frame data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = "saved_frames"
            os.makedirs(save_dir, exist_ok=True)

            # Save RGB
            rgb_path = os.path.join(save_dir, f"rgb_{timestamp}.png")
            cv.imwrite(rgb_path, rs_data['rgb'])

            # Save depth (as numpy array for precision)
            depth_path = os.path.join(save_dir, f"depth_{timestamp}.npy")
            np.save(depth_path, rs_data['depth'])

            # Save depth visualization
            depth_viz_path = os.path.join(save_dir, f"depth_viz_{timestamp}.png")
            cv.imwrite(depth_viz_path, rs_data['depth_colormap'])

            # Save intrinsics
            intrinsics_path = os.path.join(save_dir, f"intrinsics_{timestamp}.json")
            with open(intrinsics_path, 'w') as f:
                json.dump(rs_data['intrinsics'], f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

            print(f"[SAVE] Frame saved to {save_dir}/")
            messagebox.showinfo("Save Frame", f"Frame saved to {save_dir}/\n{timestamp}")

        except Exception as e:
            print(f"[ERROR] save_frame_data: {e}")
            messagebox.showerror("Save Error", f"Failed to save frame: {e}")


def main():
    root = tk.Tk()
    app = FinalPipelineDebugger(root)
    root.mainloop()


if __name__ == "__main__":
    main()
