#!/usr/bin/env python3
"""
Screen Capture Program with UI
Captures a screen region and forwards raw RGB frames to the AVP API.
Provides UI with sliders to adjust capture parameters.
"""

import numpy as np
import cv2 as cv
from mss import mss
import requests
import time
import base64
import io
from PIL import Image, ImageTk
from dataclasses import dataclass
import argparse
import sys
import threading
import tkinter as tk
from tkinter import ttk
import queue

# ------------------ Configuration ------------------
@dataclass
class CaptureConfig:
    left: int = 934
    top: int = 100
    width: int = 812
    height: int = 1080
    fps: int = 30
    api_url: str = "http://localhost:5000"

# ------------------ Highlight Window ------------------
class HighlightWindow:
    def __init__(self):
        self.root = None
        self.canvas = None
        self.active = False

    def create(self, left, top, width, height):
        """Create highlight window"""
        if self.root is not None:
            self.close()

        self.root = tk.Toplevel()
        self.root.attributes('-alpha', 0.3)  # Semi-transparent
        self.root.attributes('-topmost', True)  # Always on top
        self.root.overrideredirect(True)  # No window decorations

        # Position and size
        self.root.geometry(f"{width}x{height}+{left}+{top}")

        # Create canvas with border
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=3, highlightbackground='red')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Make window click-through (Windows only)
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20)
        except Exception:
            pass

        self.active = True

    def update_position(self, left, top, width, height):
        """Update position and size"""
        if self.root and self.active:
            try:
                self.root.geometry(f"{width}x{height}+{left}+{top}")
            except Exception:
                pass

    def close(self):
        """Close highlight window"""
        if self.root:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None
            self.canvas = None
            self.active = False

# ------------------ Screen Capture UI ------------------
class ScreenCaptureUI:
    def __init__(self, config: CaptureConfig):
        self.config = config
        self.root = tk.Tk()
        self.root.title("Screen Capture Control")
        self.root.geometry("500x400")

        # State
        self.running = False
        self.frames_captured = 0
        self.frames_sent = 0
        self.frames_failed = 0
        self.session = requests.Session()
        self.capture_thread = None
        self.highlight_window = HighlightWindow()

        # Build UI
        self._build_ui()

        # Start highlight window
        self.update_highlight()

    def _build_ui(self):
        """Build the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title = ttk.Label(main_frame, text="Screen Capture Configuration", font=('Arial', 14, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=10)

        # Sliders
        row = 1

        # Left
        ttk.Label(main_frame, text="Left:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.left_var = tk.IntVar(value=self.config.left)
        self.left_slider = ttk.Scale(main_frame, from_=0, to=2560, variable=self.left_var,
                                      orient=tk.HORIZONTAL, length=300, command=self._on_config_change)
        self.left_slider.grid(row=row, column=1, pady=5)
        self.left_label = ttk.Label(main_frame, text=str(self.config.left))
        self.left_label.grid(row=row, column=2, padx=5)
        row += 1

        # Top
        ttk.Label(main_frame, text="Top:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.top_var = tk.IntVar(value=self.config.top)
        self.top_slider = ttk.Scale(main_frame, from_=0, to=1440, variable=self.top_var,
                                     orient=tk.HORIZONTAL, length=300, command=self._on_config_change)
        self.top_slider.grid(row=row, column=1, pady=5)
        self.top_label = ttk.Label(main_frame, text=str(self.config.top))
        self.top_label.grid(row=row, column=2, padx=5)
        row += 1

        # Width
        ttk.Label(main_frame, text="Width:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.width_var = tk.IntVar(value=self.config.width)
        self.width_slider = ttk.Scale(main_frame, from_=100, to=2560, variable=self.width_var,
                                       orient=tk.HORIZONTAL, length=300, command=self._on_config_change)
        self.width_slider.grid(row=row, column=1, pady=5)
        self.width_label = ttk.Label(main_frame, text=str(self.config.width))
        self.width_label.grid(row=row, column=2, padx=5)
        row += 1

        # Height
        ttk.Label(main_frame, text="Height:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.height_var = tk.IntVar(value=self.config.height)
        self.height_slider = ttk.Scale(main_frame, from_=100, to=1440, variable=self.height_var,
                                        orient=tk.HORIZONTAL, length=300, command=self._on_config_change)
        self.height_slider.grid(row=row, column=1, pady=5)
        self.height_label = ttk.Label(main_frame, text=str(self.config.height))
        self.height_label.grid(row=row, column=2, padx=5)
        row += 1

        # FPS
        ttk.Label(main_frame, text="FPS:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.fps_var = tk.IntVar(value=self.config.fps)
        self.fps_slider = ttk.Scale(main_frame, from_=1, to=60, variable=self.fps_var,
                                     orient=tk.HORIZONTAL, length=300, command=self._on_config_change)
        self.fps_slider.grid(row=row, column=1, pady=5)
        self.fps_label = ttk.Label(main_frame, text=str(self.config.fps))
        self.fps_label.grid(row=row, column=2, padx=5)
        row += 1

        # API URL
        ttk.Label(main_frame, text="API URL:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.api_url_var = tk.StringVar(value=self.config.api_url)
        self.api_url_entry = ttk.Entry(main_frame, textvariable=self.api_url_var, width=40)
        self.api_url_entry.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=15)

        self.start_button = ttk.Button(button_frame, text="Start Capture", command=self._start_capture)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Capture", command=self._stop_capture, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)

        self.highlight_var = tk.BooleanVar(value=True)
        self.highlight_check = ttk.Checkbutton(button_frame, text="Show Highlight",
                                                variable=self.highlight_var, command=self._toggle_highlight)
        self.highlight_check.grid(row=0, column=2, padx=5)

        row += 1

        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground='blue')
        self.status_label.grid(row=row, column=0, columnspan=3, pady=10)

    def _on_config_change(self, event=None):
        """Handle configuration slider changes"""
        self.config.left = self.left_var.get()
        self.config.top = self.top_var.get()
        self.config.width = self.width_var.get()
        self.config.height = self.height_var.get()
        self.config.fps = self.fps_var.get()

        # Update labels
        self.left_label.config(text=str(self.config.left))
        self.top_label.config(text=str(self.config.top))
        self.width_label.config(text=str(self.config.width))
        self.height_label.config(text=str(self.config.height))
        self.fps_label.config(text=str(self.config.fps))

        # Update highlight window
        self.update_highlight()

    def update_highlight(self):
        """Update highlight window position"""
        if self.highlight_var.get():
            self.highlight_window.create(
                self.config.left, self.config.top,
                self.config.width, self.config.height
            )
        else:
            self.highlight_window.close()

    def _toggle_highlight(self):
        """Toggle highlight window"""
        self.update_highlight()

    def _start_capture(self):
        """Start capture thread"""
        if self.running:
            return

        # Update API URL
        self.config.api_url = self.api_url_var.get()

        # Check API availability
        try:
            response = self.session.get(f"{self.config.api_url}/health", timeout=2)
            if response.status_code != 200:
                self.status_var.set("ERROR: API not available")
                return
        except Exception as e:
            self.status_var.set(f"ERROR: Cannot connect to API - {e}")
            return

        # Start capture
        self.running = True
        self.frames_captured = 0
        self.frames_sent = 0
        self.frames_failed = 0

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        # Update UI
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Capturing...")

        # Disable sliders during capture
        self.left_slider.config(state=tk.DISABLED)
        self.top_slider.config(state=tk.DISABLED)
        self.width_slider.config(state=tk.DISABLED)
        self.height_slider.config(state=tk.DISABLED)
        self.fps_slider.config(state=tk.DISABLED)
        self.api_url_entry.config(state=tk.DISABLED)

    def _stop_capture(self):
        """Stop capture thread"""
        self.running = False

        # Update UI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Stopped")

        # Enable sliders
        self.left_slider.config(state=tk.NORMAL)
        self.top_slider.config(state=tk.NORMAL)
        self.width_slider.config(state=tk.NORMAL)
        self.height_slider.config(state=tk.NORMAL)
        self.fps_slider.config(state=tk.NORMAL)
        self.api_url_entry.config(state=tk.NORMAL)

    def _encode_frame(self, frame_bgr):
        """Encode frame to base64 JPEG"""
        try:
            img_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return img_str
        except Exception as e:
            print(f"[ERROR] Encoding frame: {e}")
            return None

    def _send_frame(self, frame_bgr):
        """Send frame to API"""
        try:
            frame_str = self._encode_frame(frame_bgr)
            if frame_str is None:
                return False

            payload = {"frame": frame_str}
            response = self.session.post(
                f"{self.config.api_url}/receive_frame",
                json=payload,
                timeout=2
            )

            return response.status_code == 200

        except requests.exceptions.Timeout:
            return False
        except requests.exceptions.ConnectionError:
            return False
        except Exception as e:
            print(f"[ERROR] Sending frame: {e}")
            return False

    def _capture_loop(self):
        """Main capture loop"""
        print("[INFO] Starting capture loop")

        start_time = time.time()
        last_stats_time = start_time

        mon = {
            "left": self.config.left,
            "top": self.config.top,
            "width": self.config.width,
            "height": self.config.height
        }

        with mss() as sct:
            while self.running:
                loop_start = time.time()

                # Capture frame
                try:
                    frame = np.asarray(sct.grab(mon))
                    frame_bgr = frame[..., :3].copy()
                    self.frames_captured += 1
                except Exception as e:
                    print(f"[ERROR] Capture failed: {e}")
                    time.sleep(0.25)
                    continue

                # Send frame to API
                success = self._send_frame(frame_bgr)
                if success:
                    self.frames_sent += 1
                else:
                    self.frames_failed += 1

                # Update stats every second
                current_time = time.time()
                if current_time - last_stats_time >= 1.0:
                    elapsed = current_time - start_time
                    actual_fps = self.frames_sent / elapsed if elapsed > 0 else 0
                    success_rate = (self.frames_sent / self.frames_captured * 100) if self.frames_captured > 0 else 0

                    status_text = (f"Capturing | Sent: {self.frames_sent} | "
                                   f"Failed: {self.frames_failed} | "
                                   f"FPS: {actual_fps:.1f} | "
                                   f"Success: {success_rate:.1f}%")

                    self.status_var.set(status_text)
                    last_stats_time = current_time

                # Throttle to target FPS
                loop_elapsed = time.time() - loop_start
                target_delay = 1.0 / self.config.fps
                if loop_elapsed < target_delay:
                    time.sleep(target_delay - loop_elapsed)

        print("[INFO] Capture loop stopped")

    def run(self):
        """Run the UI"""
        self.root.mainloop()

# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser(description="Screen Capture with UI for AVP API")
    parser.add_argument("--left", type=int, default=934, help="Initial left position")
    parser.add_argument("--top", type=int, default=100, help="Initial top position")
    parser.add_argument("--width", type=int, default=812, help="Initial width")
    parser.add_argument("--height", type=int, default=1080, help="Initial height")
    parser.add_argument("--fps", type=int, default=30, help="Initial FPS")
    parser.add_argument("--api", type=str, default="http://localhost:5000", help="API base URL")

    args = parser.parse_args()

    config = CaptureConfig(
        left=args.left,
        top=args.top,
        width=args.width,
        height=args.height,
        fps=args.fps,
        api_url=args.api
    )

    app = ScreenCaptureUI(config)
    app.run()

if __name__ == "__main__":
    main()
