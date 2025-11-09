#!/usr/bin/env python3
"""Tk UI to visualize AVP‑aligned disparity from RealSense (adjusted)."""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

from .realsense_adapter_adjusted import RealSenseToAVPAligner


class RSAdjustedPanel:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("RealSense → AVP Aligned Disparity (Adjusted)")
        root.geometry("1400x900")

        self.adapter = RealSenseToAVPAligner()
        if not self.adapter.available:
            messagebox.showwarning("RealSense", "pyrealsense2 not available or camera not detected.")

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._imgs = {}

        self._build()
        self._start()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        grid = ttk.Frame(frame)
        grid.pack(fill=tk.BOTH, expand=True)

        self.labels = {}
        titles = [("RS Color", "color"), ("Aligned Disparity", "disp"), ("Aligned Depth", "depth")]
        for idx, (title, key) in enumerate(titles):
            box = ttk.LabelFrame(grid, text=title)
            box.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=6, pady=6)
            lbl = ttk.Label(box, text="Waiting...")
            lbl.pack(fill=tk.BOTH, expand=True)
            self.labels[key] = lbl
            grid.columnconfigure(idx % 2, weight=1)
            grid.rowconfigure(idx // 2, weight=1)

        self.info = tk.Text(frame, height=10)
        self.info.pack(fill=tk.BOTH, expand=False, pady=(8, 0))

    def _start(self):
        def worker():
            while not self._stop.is_set():
                data = self.adapter.capture_and_align()
                if data:
                    self.root.after(0, self._update, data)
                time.sleep(0.15)
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def _update(self, data):
        self._set_img(self.labels["color"], data.get("color"))
        self._set_img(self.labels["disp"], data.get("aligned_disparity"), cmap=True)
        self._set_img(self.labels["depth"], self._colorize_depth(data.get("aligned_depth")))
        ts = data.get("timestamp", 0)
        self.info.delete("1.0", tk.END)
        self.info.insert(tk.END, f"Timestamp: {time.strftime('%H:%M:%S', time.localtime(ts))}\n")
        self.info.insert(tk.END, f"Extrinsics available: {self.adapter.R_avp_rs_c is not None}\n")

    def _set_img(self, label: ttk.Label, arr, cmap: bool = False):
        if arr is None:
            return
        img = arr
        if isinstance(arr, np.ndarray):
            if cmap:
                import cv2 as cv
                if img.dtype != np.uint8:
                    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
                img = cv.applyColorMap(img, cv.COLORMAP_TURBO)
            if img.ndim == 2:
                from PIL import Image
                img = Image.fromarray(img)
            else:
                from PIL import Image
                img = Image.fromarray(img[:, :, ::-1])  # BGR->RGB if needed
        photo = ImageTk.PhotoImage(image=img)
        label.configure(image=photo, text="")
        self._imgs[label] = photo

    @staticmethod
    def _colorize_depth(depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if depth is None:
            return None
        import cv2 as cv
        d = depth.copy()
        if d.size == 0:
            return None
        if d.dtype != np.float32:
            d = d.astype(np.float32)
        # Convert to colormap via normalized inverse depth
        inv = np.where(d > 0, 1.0 / d, 0.0)
        inv = cv.normalize(inv, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        return cv.applyColorMap(inv, cv.COLORMAP_TURBO)

    def _on_close(self):
        self._stop.set()
        try:
            self.adapter.stop()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":  # pragma: no cover
    root = tk.Tk()
    RSAdjustedPanel(root)
    root.mainloop()

