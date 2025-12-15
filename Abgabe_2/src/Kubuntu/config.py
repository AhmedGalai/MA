#!/usr/bin/env python3
"""
Configuration module for the pose estimation pipeline.

This module serves as the single source of truth for all configuration parameters.
Import with: from config import CONFIG
"""

import os
import cv2

# Single configuration dictionary - the source of truth for all settings
CONFIG = {
    # =========================================================================
    # Network Configuration
    # =========================================================================
    "network": {
        "main_api_host": os.getenv("MAIN_API_HOST", "192.168.178.68"),
        "main_api_port": int(os.getenv("MAIN_API_PORT", "8000")),
        "foundationpose_url": os.getenv(
            "FOUNDATIONPOSE_URL",
            "http://localhost:5000/foundationpose"
        ),
    },

    # =========================================================================
    # ArUco Board Configuration
    # =========================================================================
    "aruco": {
        "dictionary_type": cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        ),
        "dictionary_enum": cv2.aruco.DICT_4X4_50,
        "rows": 3,
        "cols": 4,
        "marker_size_m": 0.030,          # 30mm
        "marker_separation_m": 0.010,    # 10mm
    },

    # =========================================================================
    # RealSense Camera Configuration
    # =========================================================================
    "realsense": {
        "resolution_width": 640,
        "resolution_height": 480,
        "fps": 30,
        "enable_color_alignment": True,
    },

    # =========================================================================
    # File Paths
    # =========================================================================
    "paths": {
        "models_dir": os.path.join(os.path.dirname(__file__), "models"),
        "extrinsics_dir": os.path.join(os.path.dirname(__file__), "extrinsics"),
        "calibration_file": os.path.join(
            os.path.dirname(__file__),
            "extrinsics",
            "T_world_rs.json"
        ),
    },

    # =========================================================================
    # UxPlay Configuration (Docker-based AirPlay receiver)
    # =========================================================================
    "uxplay": {
        "enabled": True,
        "frame_dir": os.path.join(os.path.dirname(__file__), "frames"),
        "docker_compose_file": os.path.join(
            os.path.dirname(__file__),
            "docker-compose.yml"
        ),
        "device_name": "Kubuntu Backend",
        "max_frame_age": 2.0,  # seconds - max age for frame to be considered valid
    },

    # =========================================================================
    # Processing Parameters
    # =========================================================================
    "processing": {
        "head_pose_update_timeout": 1.0,  # seconds
        "default_model": "cube.ply",
    },

    # =========================================================================
    # Debug Mode (enable via DEBUG_MODE environment variable)
    # =========================================================================
    "debug": {
        "enabled": os.getenv("DEBUG_MODE", "false").lower() == "true",
    },
}


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================
# These allow existing code using the old flat structure to continue working

MAIN_API_HOST = CONFIG["network"]["main_api_host"]
MAIN_API_PORT = CONFIG["network"]["main_api_port"]
FOUNDATIONPOSE_URL = CONFIG["network"]["foundationpose_url"]

ARUCO_DICT = "DICT_4X4_50"
ARUCO_ROWS = CONFIG["aruco"]["rows"]
ARUCO_COLS = CONFIG["aruco"]["cols"]
MARKER_SIZE_M = CONFIG["aruco"]["marker_size_m"]
SEPARATION_M = CONFIG["aruco"]["marker_separation_m"]

RS_WIDTH = CONFIG["realsense"]["resolution_width"]
RS_HEIGHT = CONFIG["realsense"]["resolution_height"]
RS_FPS = CONFIG["realsense"]["fps"]

MODELS_DIR = CONFIG["paths"]["models_dir"]
EXTRINSICS_DIR = CONFIG["paths"]["extrinsics_dir"]
EXTRINSICS_FILE = CONFIG["paths"]["calibration_file"]

HEAD_POSE_MAX_AGE = CONFIG["processing"]["head_pose_update_timeout"]
DEFAULT_MODEL = CONFIG["processing"]["default_model"]

DEBUG_MODE = CONFIG["debug"]["enabled"]


if __name__ == "__main__":
    # Simple usage demonstration
    print("Configuration loaded successfully!")
    print(f"\nNetwork Configuration:")
    print(f"  Main API: {CONFIG['network']['main_api_host']}:{CONFIG['network']['main_api_port']}")
    print(f"  FoundationPose URL: {CONFIG['network']['foundationpose_url']}")

    print(f"\nArUco Board Configuration:")
    print(f"  Board size: {CONFIG['aruco']['rows']}x{CONFIG['aruco']['cols']}")
    print(f"  Marker size: {CONFIG['aruco']['marker_size_m']*1000}mm")
    print(f"  Marker separation: {CONFIG['aruco']['marker_separation_m']*1000}mm")

    print(f"\nRealSense Configuration:")
    print(f"  Resolution: {CONFIG['realsense']['resolution_width']}x{CONFIG['realsense']['resolution_height']}")
    print(f"  FPS: {CONFIG['realsense']['fps']}")

    print(f"\nFile Paths:")
    print(f"  Models: {CONFIG['paths']['models_dir']}")
    print(f"  Calibration: {CONFIG['paths']['calibration_file']}")

    print(f"\nProcessing Parameters:")
    print(f"  Head pose timeout: {CONFIG['processing']['head_pose_update_timeout']}s")
    print(f"  Default model: {CONFIG['processing']['default_model']}")
