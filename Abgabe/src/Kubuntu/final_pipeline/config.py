"""
Configuration for Final Pipeline
"""

import os

# RealSense Configuration
REALSENSE_CONFIG = {
    "width": 640,
    "height": 480,
    "fps": 30,
    "depth_format": "z16",
    "color_format": "bgr8"
}

# ArUco Marker Configuration
ARUCO_CONFIG = {
    "dictionary": "DICT_4X4_50",  # ArUco dictionary
    "marker_size_m": 0.030,  # 3cm markers
    "separation_m": 0.010,   # 1cm separation
    "board_rows": 3,
    "board_cols": 4,
    # For headset-mounted marker
    "headset_marker_id": 0,
    "headset_marker_size_m": 0.050  # 5cm marker on headset
}

# Kalman Filter Configuration (for pose correction)
KALMAN_CONFIG = {
    "process_noise": 0.01,  # Process noise covariance
    "measurement_noise": 0.1,  # Measurement noise covariance
    "initial_uncertainty": 1.0  # Initial state uncertainty
}

# Pose Estimation Configuration
POSE_ESTIMATION_CONFIG = {
    "min_points": 4,  # Minimum points for PnP
    "ransac_threshold": 3.0,  # RANSAC inlier threshold (pixels)
    "ransac_iterations": 100,
    "confidence_threshold": 0.7  # Minimum confidence for valid pose
}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.join(BASE_DIR, "calibration")
os.makedirs(CALIBRATION_DIR, exist_ok=True)

CALIBRATION_FILES = {
    "headset_to_world": os.path.join(CALIBRATION_DIR, "headset_to_world.json"),
    "realsense_to_world": os.path.join(CALIBRATION_DIR, "realsense_to_world.json"),
    "avp_to_realsense": os.path.join(CALIBRATION_DIR, "avp_to_realsense.json")
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5001,
    "debug": False
}
