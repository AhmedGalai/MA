#!/usr/bin/env python3
"""
RealSense camera manager for RGB-D capture.
"""

import numpy as np
import threading
from typing import Optional, Dict, Tuple
import config

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    rs = None
    REALSENSE_AVAILABLE = False
    print("[RealSense] pyrealsense2 not available")


class RealSenseManager:
    """Manages RealSense camera for RGB-D capture."""

    def __init__(self):
        """Initialize RealSense camera."""
        self.pipeline = None
        self.align = None
        self.profile = None
        self.intrinsics = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.available = False
        self.lock = threading.Lock()

        if not REALSENSE_AVAILABLE:
            print("[RealSense] Cannot initialize - pyrealsense2 not available")
            return

        try:
            self._start_camera()
            self.available = True
            print("[RealSense] Camera initialized successfully")
        except Exception as e:
            print(f"[RealSense] Failed to initialize: {e}")
            self.stop()

    def _start_camera(self):
        """Start RealSense pipeline."""
        self.pipeline = rs.pipeline()
        cfg = rs.config()

        # Configure streams
        cfg.enable_stream(
            rs.stream.depth,
            config.RS_WIDTH,
            config.RS_HEIGHT,
            rs.format.z16,
            config.RS_FPS
        )
        cfg.enable_stream(
            rs.stream.color,
            config.RS_WIDTH,
            config.RS_HEIGHT,
            rs.format.bgr8,
            config.RS_FPS
        )

        # Start pipeline
        self.profile = self.pipeline.start(cfg)

        # Create align object (align depth to color)
        self.align = rs.align(rs.stream.color)

        # Get camera intrinsics
        self._extract_intrinsics()

    def _extract_intrinsics(self):
        """Extract camera intrinsics from RealSense profile."""
        intrinsics = (
            self.profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        self.intrinsics = intrinsics

        # Convert to camera matrix format
        self.camera_matrix = np.array([
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Extract distortion coefficients
        coeffs = intrinsics.coeffs if hasattr(intrinsics, 'coeffs') else [0, 0, 0, 0, 0]
        self.dist_coeffs = np.array(coeffs[:5], dtype=np.float64).reshape(-1, 1)

        print(f"[RealSense] Camera matrix:\n{self.camera_matrix}")

    def capture(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Capture aligned RGB-D frame.

        Returns:
            Dictionary with 'rgb', 'depth', and 'timestamp' or None on failure
        """
        if not self.available or self.pipeline is None:
            return None

        with self.lock:
            try:
                # Wait for frames
                frames = self.pipeline.wait_for_frames(timeout_ms=1500)

                # Align depth to color
                aligned_frames = self.align.process(frames)

                # Get aligned frames
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    return None

                # Convert to numpy arrays
                rgb = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())

                # Convert depth from uint16 to float32 meters
                depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
                depth = depth.astype(np.float32) * depth_scale

                return {
                    'rgb': rgb,
                    'depth': depth,
                    'depth_scale': depth_scale,
                    'timestamp': frames.get_timestamp() / 1000.0  # Convert to seconds
                }

            except Exception as e:
                print(f"[RealSense] Capture failed: {e}")
                return None

    def get_camera_info(self) -> Dict:
        """
        Get camera information.

        Returns:
            Dictionary with camera parameters
        """
        if not self.available:
            return {'available': False}

        return {
            'available': True,
            'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            'dist_coeffs': self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            'width': config.RS_WIDTH,
            'height': config.RS_HEIGHT,
            'fps': config.RS_FPS
        }

    def stop(self):
        """Stop RealSense pipeline."""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self.align = None
            self.available = False
            print("[RealSense] Camera stopped")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
