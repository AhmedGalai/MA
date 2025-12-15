"""
Intel RealSense Camera Interface for RGB+Depth Capture

This module provides a client for interfacing with Intel RealSense cameras,
enabling aligned RGB and depth frame capture with intrinsic calibration data.
"""

import logging
import numpy as np
import pyrealsense2 as rs
from config import CONFIG


logger = logging.getLogger(__name__)


class RealSenseClient:
    """
    Intel RealSense camera client for capturing aligned RGB and depth frames.

    This class handles the initialization, configuration, and operation of
    Intel RealSense cameras. It provides frame capture with automatic alignment
    between depth and color streams, and retrieves camera intrinsic parameters.
    """

    def __init__(self, width=640, height=480, fps=30):
        """
        Initialize RealSense camera client configuration.

        Args:
            width (int): Frame width in pixels. Default: 640
            height (int): Frame height in pixels. Default: 480
            fps (int): Frames per second. Default: 30
        """
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.config = None
        self.align_to = None
        self.align = None
        self.is_running = False
        self.intrinsics = None

        logger.info(f"RealSenseClient initialized with resolution {width}x{height} @ {fps} fps")

    def start(self) -> bool:
        """
        Start the RealSense pipeline and configure streams.

        Configures the depth stream (Z16 format) and color stream (BGR8 format),
        and creates an alignment object to align depth frames to color frames.

        Returns:
            bool: True on successful startup, False on failure.
        """
        try:
            # Create pipeline and configuration objects
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Configure color stream (BGR8)
            self.config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps
            )

            # Configure depth stream (Z16)
            self.config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps
            )

            # Start the pipeline
            profile = self.pipeline.start(self.config)

            # Create alignment object (align depth to color)
            self.align_to = rs.stream.color
            self.align = rs.align(self.align_to)

            # Extract intrinsics from color stream
            color_profile = profile.get_stream(rs.stream.color)
            intr = color_profile.as_video_stream_profile().get_intrinsics()

            # Build intrinsics matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            self.intrinsics = np.array([
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1]
            ], dtype=np.float32)

            self.is_running = True
            logger.info("RealSense pipeline started successfully")
            return True

        except RuntimeError as e:
            logger.error(f"Failed to start RealSense pipeline: {e}")
            self.is_running = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error during RealSense startup: {e}")
            self.is_running = False
            return False

    def stop(self):
        """
        Stop the RealSense pipeline.

        Safe to call multiple times. Does not raise exceptions.
        """
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                self.is_running = False
                logger.info("RealSense pipeline stopped")
            except Exception as e:
                logger.warning(f"Error stopping RealSense pipeline: {e}")
        else:
            logger.debug("Pipeline was not running, no action taken")

    def capture(self) -> dict or None:
        """
        Capture aligned RGB and depth frames from the camera.

        Waits for frames with a 1000ms timeout and aligns depth to color frame.
        Depth values are converted to meters (float32).

        Returns:
            dict: Dictionary containing:
                - 'rgb': uint8 ndarray of shape (H, W, 3) in BGR format
                - 'depth': float32 ndarray of shape (H, W) in meters
                - 'K': 3x3 float32 intrinsics matrix
                - 'timestamp': float timestamp in seconds
            None: On capture failure or timeout
        """
        if not self.is_running:
            logger.warning("Cannot capture: pipeline is not running")
            return None

        try:
            # Wait for a coherent pair of frames with 1000ms timeout
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)

            # Align depth to color frame
            aligned_frames = self.align.process(frames)

            # Extract aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                logger.warning("Failed to get aligned frames")
                return None

            # Get timestamp from color frame
            timestamp = color_frame.get_timestamp() / 1000.0  # Convert ms to seconds

            # Convert to numpy arrays
            rgb = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
            depth_raw = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)

            # Convert depth from millimeters to meters (float32)
            depth = (depth_raw.astype(np.float32) / 1000.0)

            # Ensure RGB has shape (H, W, 3)
            if len(rgb.shape) != 3 or rgb.shape[2] != 3:
                logger.error(f"Unexpected RGB shape: {rgb.shape}")
                return None

            # Ensure depth has shape (H, W)
            if len(depth.shape) != 2:
                logger.error(f"Unexpected depth shape: {depth.shape}")
                return None

            result = {
                'rgb': rgb,
                'depth': depth,
                'K': self.intrinsics.copy(),
                'timestamp': timestamp
            }

            logger.debug(f"Frame captured successfully at timestamp {timestamp:.3f}s")
            return result

        except RuntimeError as e:
            logger.error(f"RealSense capture error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during frame capture: {e}")
            return None

    def get_intrinsics(self) -> np.ndarray:
        """
        Get the camera intrinsics matrix.

        Returns the 3x3 intrinsics matrix from the color stream in the format:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

        Returns:
            np.ndarray: 3x3 float32 intrinsics matrix, or None if not initialized.
        """
        if self.intrinsics is None:
            logger.warning("Intrinsics not available - pipeline may not be started")
            return None

        return self.intrinsics.copy()
