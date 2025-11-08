#!/usr/bin/env python3
"""Tkinter panel for visualizing RealSense disparity pipeline."""

import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

from realsense_adapter import RealSenseDisparityAdapter


class RealSenseDebugPanel:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("RealSense Debug Panel")
        self.root.geometry("1400x900")

        self.adapter = RealSenseDisparityAdapter()
        if not self.adapter.available:
            messagebox.showwarning(
                "RealSense",
                "pyrealsense2 not available or camera not detected."
            )

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._setup_widgets()
        self._start_thread()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    def _setup_widgets(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        grid = ttk.Frame(frame)
        grid.pack(fill=tk.BOTH, expand=True)

        self.labels = {}
        self.images = {}
        titles = [
            ("RGB Feed", "rgb"),
            ("RS Disparity", "disp"),
            ("Pattern View", "pattern"),
            ("Transformed Disparity", "disp_xform"),
        ]
        for idx, (title, key) in enumerate(titles):
            holder = ttk.LabelFrame(grid, text=title)
            holder.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=6, pady=6)
            img_label = ttk.Label(holder)
            img_label.pack(fill=tk.BOTH, expand=True)
            self.labels[key] = img_label
            grid.columnconfigure(idx % 2, weight=1)
            grid.rowconfigure(idx // 2, weight=1)

        self.info = tk.Text(frame, height=10, font=("Menlo", 11))
        self.info.pack(fill=tk.BOTH, expand=False, pady=(10, 0))

    # ------------------------------------------------------------------
    def _start_thread(self):
        def worker():
            while not self._stop.is_set():
                data = None
                try:
                    data = self.adapter.capture_and_process()
                except Exception as exc:
                    self._write_info(f"Capture failed: {exc}\n")
                if data:
                    self.root.after(0, self._update_ui, data)
                time.sleep(0.2)
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def _update_ui(self, data):
        self._update_image(self.labels["rgb"], data.get("color_frame"))
        self._update_image(self.labels["disp"], data.get("disparity"), cmap=True)
        self._update_image(self.labels["pattern"], data.get("pattern_view"))
        self._update_image(self.labels["disp_xform"], data.get("transformed_disparity"), cmap=True)

        info_lines = [
            f"Timestamp: {time.strftime('%H:%M:%S', time.localtime(data.get('timestamp', 0)))}",
            f"Pattern detected: {data.get('pattern_pose') is not None}",
        ]
        transform = data.get("transform")
        if isinstance(transform, np.ndarray):
            info_lines.append("Transform (first row): " + np.array2string(transform[0], precision=3))
        snapshot = data.get("main_api") or {}
        if snapshot.get("head_pose"):
            info_lines.append(f"Head pose: {snapshot['head_pose']}")
        if snapshot.get("avp_pose"):
            info_lines.append(f"AVP pose rvec: {snapshot['avp_pose'].get('rvec')}")
        self._write_info("\n".join(info_lines) + "\n")

    def _update_image(self, widget: ttk.Label, frame, cmap: bool = False):
        if frame is None:
            return
        if cmap:
            frame = cvt_color_map(frame)
        image = Image.fromarray(frame)
        image = image.resize((int(widget.winfo_width() or 640), int(widget.winfo_height() or 360)))
        photo = ImageTk.PhotoImage(image=image)
        widget.configure(image=photo)
        widget.image = photo

    def _write_info(self, text: str):
        self.info.delete("1.0", tk.END)
        self.info.insert(tk.END, text)

    def _on_close(self):
        self._stop.set()
        if self.adapter:
            try:
                self.adapter.stop()
            except Exception:
                pass
        self.root.destroy()


def cvt_color_map(frame: np.ndarray) -> np.ndarray:
    import cv2 as cv

    if frame.dtype != np.uint8:
        frame = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return cv.applyColorMap(frame, cv.COLORMAP_TURBO)


if __name__ == "__main__":  # pragma: no cover - manual tool
    root = tk.Tk()
    RealSenseDebugPanel(root)
    root.mainloop()
