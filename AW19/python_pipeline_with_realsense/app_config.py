#!/usr/bin/env python3
"""
Central configuration for hosts, ports, and defaults.
Edit this file to change runtime addresses and default parameters.
"""

APP_CONFIG = {
    "main_api": {
        "host": "localhost",
        "port": 8000,
        "base_url": "http://localhost:8000",
    },
    "pose_api": {
        "host": "localhost",
        "port": 5000,
        "base_url": "http://localhost:5000",
        "route": "/pose",
    },
    "defaults": {
        "model_name": "cube.ply",
        "estimate_depth": True,
        "use_random_pose": True,
        "ui_refresh_hz": 30,
        "roi_hsv_center": [90, 128, 128],
        "tolerances": {"h": 12, "s": 50, "v": 50},
    },
    "capture": {
        "left": 934,
        "top": 100,
        "width": 812,
        "height": 1080,
        "fps": 30,
    }
}
