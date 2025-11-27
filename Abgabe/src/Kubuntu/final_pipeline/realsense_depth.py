"""
RealSense Depth Camera Interface
Provides metric depth from fixed RealSense D435/D455 camera
"""

import numpy as np
import cv2 as cv
from typing import Optional, Dict, Tuple
import time

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARNING] pyrealsense2 not available")

from .config import REALSENSE_CONFIG


class RealSenseDepth:
    """
    RealSense depth camera interface
    Captures aligned depth + color frames with metric depth values
    """

    def __init__(self):
        """Initialize RealSense pipeline"""
        self.pipeline = None
        self.align = None
        self.profile = None
        self.intrinsics = None
        self.available = False

        if not REALSENSE_AVAILABLE:
            print("[ERROR] RealSense not available - install pyrealsense2")
            return

        try:
            self._initialize_camera()
            self.available = True
            print("[RealSense] Camera initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize RealSense: {e}")
            self.available = False

    def _initialize_camera(self):
        """Start RealSense pipeline and get intrinsics"""
        # Create pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Configure streams
        config.enable_stream(
            rs.stream.depth,
            REALSENSE_CONFIG["width"],
            REALSENSE_CONFIG["height"],
            getattr(rs.format, REALSENSE_CONFIG["depth_format"]),
            REALSENSE_CONFIG["fps"]
        )
        config.enable_stream(
            rs.stream.color,
            REALSENSE_CONFIG["width"],
            REALSENSE_CONFIG["height"],
            getattr(rs.format, REALSENSE_CONFIG["color_format"]),
            REALSENSE_CONFIG["fps"]
        )

        # Start pipeline
        self.profile = self.pipeline.start(config)

        # Create alignment object (align depth to color frame)
        self.align = rs.align(rs.stream.color)

        # Get color stream intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # Extract intrinsic parameters
        self.intrinsics = {
            "width": color_intrinsics.width,
            "height": color_intrinsics.height,
            "fx": color_intrinsics.fx,
            "fy": color_intrinsics.fy,
            "ppx": color_intrinsics.ppx,
            "ppy": color_intrinsics.ppy,
            "coeffs": color_intrinsics.coeffs,
            "K": np.array([
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float32),
            "dist": np.array(color_intrinsics.coeffs[:5], dtype=np.float32).reshape(-1, 1)
        }

        print(f"[RealSense] Intrinsics: fx={self.intrinsics['fx']:.1f}, fy={self.intrinsics['fy']:.1f}")
        print(f"[RealSense] Principal point: ({self.intrinsics['ppx']:.1f}, {self.intrinsics['ppy']:.1f})")

    def capture_frame(self) -> Optional[Dict]:
        """
        Capture aligned RGB + Depth frame

        Returns:
            Dictionary containing:
                - rgb: RGB image (H, W, 3) uint8
                - depth: Depth map (H, W) uint16 in millimeters
                - depth_colormap: Visualization of depth
                - timestamp: Capture timestamp
        """
        if not self.available:
            return None

        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            timestamp = time.time()

            # Align depth to color
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("[WARNING] Failed to get aligned frames")
                return None

            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())  # uint16, millimeters
            color_image = np.asanyarray(color_frame.get_data())  # uint8, BGR

            # Create depth colormap for visualization
            depth_colormap = cv.applyColorMap(
                cv.convertScaleAbs(depth_image, alpha=0.03),
                cv.COLORMAP_JET
            )

            return {
                "rgb": color_image,
                "depth": depth_image,
                "depth_colormap": depth_colormap,
                "timestamp": timestamp,
                "intrinsics": self.intrinsics
            }

        except Exception as e:
            print(f"[ERROR] Frame capture failed: {e}")
            return None

    def get_depth_at_pixel(self, depth_map: np.ndarray, u: int, v: int) -> Optional[float]:
        """
        Get depth value at specific pixel location

        Args:
            depth_map: Depth map (uint16, millimeters)
            u, v: Pixel coordinates

        Returns:
            Depth in meters, or None if invalid
        """
        h, w = depth_map.shape
        if u < 0 or u >= w or v < 0 or v >= h:
            return None

        depth_mm = depth_map[v, u]
        if depth_mm == 0:
            return None

        return depth_mm / 1000.0  # Convert to meters

    def deproject_pixel_to_point(self, u: int, v: int, depth: float) -> Optional[np.ndarray]:
        """
        Deproject pixel + depth to 3D point in camera frame

        Args:
            u, v: Pixel coordinates
            depth: Depth value in meters

        Returns:
            3D point [x, y, z] in camera frame, or None if invalid
        """
        if self.intrinsics is None or depth <= 0:
            return None

        # Deproject using pinhole camera model
        x = (u - self.intrinsics["ppx"]) * depth / self.intrinsics["fx"]
        y = (v - self.intrinsics["ppy"]) * depth / self.intrinsics["fy"]
        z = depth

        return np.array([x, y, z], dtype=np.float32)

    def deproject_depth_to_pointcloud(self, depth_map: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Deproject entire depth map to 3D point cloud

        Args:
            depth_map: Depth map (uint16, millimeters)
            mask: Optional binary mask to filter points

        Returns:
            Point cloud (N, 3) where N is number of valid points
        """
        if self.intrinsics is None:
            return np.array([])

        h, w = depth_map.shape
        points = []

        # Create pixel grid
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

        # Apply mask if provided
        if mask is not None:
            valid_mask = (depth_map > 0) & (mask > 0)
        else:
            valid_mask = depth_map > 0

        # Get valid pixel coordinates and depths
        valid_u = u_coords[valid_mask]
        valid_v = v_coords[valid_mask]
        valid_depths = depth_map[valid_mask] / 1000.0  # Convert to meters

        # Deproject all valid points
        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
        ppx, ppy = self.intrinsics["ppx"], self.intrinsics["ppy"]

        x = (valid_u - ppx) * valid_depths / fx
        y = (valid_v - ppy) * valid_depths / fy
        z = valid_depths

        points = np.stack([x, y, z], axis=1)

        return points

    def get_intrinsics_matrix(self) -> Optional[np.ndarray]:
        """Get camera intrinsics matrix K (3x3)"""
        if self.intrinsics is None:
            return None
        return self.intrinsics["K"]

    def get_distortion_coeffs(self) -> Optional[np.ndarray]:
        """Get distortion coefficients"""
        if self.intrinsics is None:
            return None
        return self.intrinsics["dist"]

    def stop(self):
        """Stop RealSense pipeline"""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                print("[RealSense] Pipeline stopped")
            except Exception as e:
                print(f"[WARNING] Error stopping pipeline: {e}")
        self.pipeline = None
        self.align = None
        self.available = False

    def __del__(self):
        """Cleanup on deletion"""
        self.stop()


# Test functionality
if __name__ == "__main__":
    print("Testing RealSense Depth Camera...")

    camera = RealSenseDepth()

    if not camera.available:
        print("RealSense camera not available!")
        exit(1)

    print("\nCapturing 10 frames...")
    for i in range(10):
        data = camera.capture_frame()
        if data:
            print(f"Frame {i+1}: RGB shape={data['rgb'].shape}, Depth range=[{data['depth'].min()}, {data['depth'].max()}]mm")
        else:
            print(f"Frame {i+1}: Failed to capture")

    camera.stop()
    print("\nTest complete!")
