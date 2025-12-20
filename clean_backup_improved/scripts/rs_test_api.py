#!/usr/bin/env python3
"""
Minimal RealSense RGBD test API (single endpoint).
Run: py scripts/rs_test_api.py --host 0.0.0.0 --port 9000
GET /rgbd -> { rgb, depth, timestamp }
"""

from __future__ import annotations

import argparse
import base64
import time

import cv2
import numpy as np
from flask import Flask, jsonify

from realsense_client import RealSenseClient


def encode_jpeg_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return base64.b64encode(buf).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RealSense RGBD test API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    rs = RealSenseClient(width=args.width, height=args.height, fps=args.fps)
    if not rs.start():
        raise SystemExit("RealSense start failed")

    app = Flask(__name__)

    @app.route("/rgbd", methods=["GET"])
    def rgbd():
        frame = rs.capture()
        if frame is None:
            return jsonify({"error": "No frame available"}), 503
        rgb = frame["rgb"]
        depth = frame["depth"]
        ts = frame.get("timestamp", time.time())

        rgb_b64 = encode_jpeg_b64(rgb)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        depth_b64 = encode_jpeg_b64(depth_color)
        return jsonify(
            {
                "rgb": f"data:image/jpeg;base64,{rgb_b64}",
                "depth": f"data:image/jpeg;base64,{depth_b64}",
                "timestamp": ts,
            }
        )

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        rs.stop()


if __name__ == "__main__":
    main()
