#!/usr/bin/env python3
"""
UxPlay Integration Module for Main API

This module manages UxPlay AirPlay receiver and integrates it with the main API.
It captures RGB frames from visionOS devices and forwards them to the main API.
"""

import os
import sys
import subprocess
import threading
import time
import signal
import logging
import base64
import requests
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Callable
from config import CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UxPlayManager:
    """
    Manages UxPlay AirPlay receiver process and frame capture.

    Captures RGB frames from UxPlay stdout and forwards them to the main API
    for processing and pose estimation.
    """

    def __init__(self,
                 uxplay_binary: str = None,
                 device_name: str = "AirPlay-Pipeline",
                 main_api_url: str = None,
                 frame_callback: Optional[Callable] = None,
                 auto_forward_to_api: bool = True):
        """
        Initialize UxPlay manager.

        Args:
            uxplay_binary: Path to uxplay binary. Auto-detected if None.
            device_name: AirPlay device name visible to iOS/visionOS devices
            main_api_url: Main API URL for forwarding frames
            frame_callback: Optional callback function(frame) for custom processing
            auto_forward_to_api: Automatically forward frames to main API
        """
        # Find uxplay binary
        if uxplay_binary is None:
            uxplay_binary = self._find_uxplay_binary()

        self.uxplay_binary = uxplay_binary
        self.device_name = device_name

        # API configuration
        if main_api_url is None:
            host = CONFIG["network"]["main_api_host"]
            port = CONFIG["network"]["main_api_port"]
            self.main_api_url = f"http://{host}:{port}"
        else:
            self.main_api_url = main_api_url

        self.frame_callback = frame_callback
        self.auto_forward_to_api = auto_forward_to_api

        # Process management
        self.process = None
        self.running = False
        self.capture_thread = None

        # Frame statistics
        self.frames_captured = 0
        self.frames_forwarded = 0
        self.last_frame_time = None
        self.fps = 0.0

        # Resolution (will be detected from stream)
        self.frame_width = None
        self.frame_height = None

        logger.info(f"UxPlayManager initialized: device_name={device_name}")
        logger.info(f"  Binary: {self.uxplay_binary}")
        logger.info(f"  API URL: {self.main_api_url}")

    def _find_uxplay_binary(self) -> str:
        """
        Automatically find uxplay binary.

        Returns:
            Path to uxplay binary

        Raises:
            FileNotFoundError: If uxplay binary not found
        """
        # Common locations
        search_paths = [
            "/usr/local/bin/uxplay",
            "/usr/bin/uxplay",
            "/opt/homebrew/bin/uxplay",
            # Xcode derived data path (from the running process we saw)
            "/Users/match-mac/Library/Developer/Xcode/DerivedData/ClientAPI-dmfpkvzlxafeggfagbqgzfzjkjmy/Build/Products/Debug/ClientAPI.app/Contents/Resources/uxplay"
        ]

        for path in search_paths:
            if os.path.exists(path):
                logger.info(f"Found uxplay binary: {path}")
                return path

        # Try which command
        try:
            result = subprocess.run(
                ["which", "uxplay"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                logger.info(f"Found uxplay binary via 'which': {path}")
                return path
        except Exception as e:
            logger.warning(f"'which uxplay' failed: {e}")

        raise FileNotFoundError(
            "Could not find uxplay binary. Please install UxPlay or "
            "specify the path manually."
        )

    def start(self, width: int = 1920, height: int = 1080):
        """
        Start UxPlay process and begin frame capture.

        Args:
            width: Video width (default: 1920)
            height: Video height (default: 1080)
        """
        if self.running:
            logger.warning("UxPlay is already running")
            return

        self.frame_width = width
        self.frame_height = height

        # Build uxplay command
        # Output BGR frames to stdout using GStreamer
        cmd = [
            self.uxplay_binary,
            "-n", self.device_name,
            "-vsync", "no",
            "-as", "0",  # Disable audio
            "-vs", f"videoconvert ! video/x-raw,format=BGR ! fdsink fd=1 sync=false"
        ]

        logger.info(f"Starting UxPlay: {' '.join(cmd)}")

        try:
            # Start process with stdout capture
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # Unbuffered
            )

            self.running = True

            # Start frame capture thread
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True
            )
            self.capture_thread.start()

            logger.info("UxPlay started successfully")
            logger.info(f"AirPlay device '{self.device_name}' is now discoverable")

        except Exception as e:
            logger.error(f"Failed to start UxPlay: {e}")
            self.running = False
            raise

    def _capture_loop(self):
        """
        Main capture loop - reads frames from UxPlay stdout.

        Runs in separate thread.
        """
        logger.info("Frame capture thread started")

        # Calculate frame size (BGR = 3 bytes per pixel)
        if self.frame_width is None or self.frame_height is None:
            # Default to 1920x1080 if not specified
            self.frame_width = 1920
            self.frame_height = 1080

        frame_size = self.frame_width * self.frame_height * 3
        fps_counter_start = time.time()
        fps_frame_count = 0

        # Buffer to accumulate partial reads
        frame_buffer = bytearray()
        first_frame_logged = False

        try:
            while self.running:
                # Read available data (may be less than frame_size)
                chunk = self.process.stdout.read(frame_size - len(frame_buffer))

                if not chunk:
                    # End of stream or process terminated
                    if self.running:
                        logger.warning("Stream ended unexpectedly")
                    break

                # Accumulate data in buffer
                frame_buffer.extend(chunk)

                # Check if we have a complete frame
                if len(frame_buffer) >= frame_size:
                    # Extract one frame
                    frame_data = bytes(frame_buffer[:frame_size])
                    frame_buffer = frame_buffer[frame_size:]  # Keep excess data

                    if not first_frame_logged:
                        logger.info("First frame received from UxPlay stream")
                        first_frame_logged = True

                    # Convert to numpy array
                    frame = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = frame.reshape((self.frame_height, self.frame_width, 3))

                    # Update statistics
                    self.frames_captured += 1
                    self.last_frame_time = time.time()
                    fps_frame_count += 1

                    # Calculate FPS every second
                    if time.time() - fps_counter_start >= 1.0:
                        self.fps = fps_frame_count / (time.time() - fps_counter_start)
                        fps_counter_start = time.time()
                        fps_frame_count = 0

                    # Process frame
                    self._process_frame(frame)

        except Exception as e:
            if self.running:
                logger.error(f"Error in capture loop: {e}", exc_info=True)
        finally:
            logger.info("Frame capture thread stopped")

    def _process_frame(self, frame: np.ndarray):
        """
        Process captured frame.

        Args:
            frame: BGR frame from UxPlay
        """
        # Call custom callback if provided
        if self.frame_callback is not None:
            try:
                self.frame_callback(frame)
            except Exception as e:
                logger.error(f"Error in frame callback: {e}")

        # Auto-forward to main API
        if self.auto_forward_to_api:
            self._forward_frame_to_api(frame)

    def _forward_frame_to_api(self, frame: np.ndarray, purpose: str = "general"):
        """
        Forward frame to main API /receive_frame endpoint.

        Args:
            frame: BGR frame to send
            purpose: Purpose tag for the frame
        """
        try:
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Send to API
            payload = {
                "rgb_frame": f"data:image/jpeg;base64,{frame_b64}",
                "purpose": purpose,
                "timestamp": time.time()
            }

            response = requests.post(
                f"{self.main_api_url}/receive_frame",
                json=payload,
                timeout=2.0
            )

            if response.status_code == 200:
                self.frames_forwarded += 1
                if self.frames_forwarded % 30 == 0:  # Log every 30 frames
                    logger.debug(f"Forwarded {self.frames_forwarded} frames to API")
            else:
                logger.warning(f"API returned status {response.status_code}")

        except requests.exceptions.Timeout:
            logger.warning("Timeout forwarding frame to API")
        except requests.exceptions.ConnectionError:
            logger.warning("Connection error - is main_api.py running?")
        except Exception as e:
            logger.error(f"Error forwarding frame: {e}")

    def stop(self):
        """Stop UxPlay process and frame capture."""
        if not self.running:
            logger.warning("UxPlay is not running")
            return

        logger.info("Stopping UxPlay...")
        self.running = False

        # Terminate process
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process did not terminate, killing...")
                self.process.kill()
                self.process.wait()

            self.process = None

        # Wait for capture thread
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)

        logger.info("UxPlay stopped")
        logger.info(f"Statistics: {self.frames_captured} frames captured, "
                   f"{self.frames_forwarded} forwarded")

    def is_running(self) -> bool:
        """Check if UxPlay is running."""
        return self.running and self.process is not None

    def get_stats(self) -> dict:
        """
        Get capture statistics.

        Returns:
            Dictionary with capture stats
        """
        return {
            "running": self.running,
            "frames_captured": self.frames_captured,
            "frames_forwarded": self.frames_forwarded,
            "fps": self.fps,
            "last_frame_time": self.last_frame_time,
            "resolution": f"{self.frame_width}x{self.frame_height}" if self.frame_width else None
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def main():
    """Command-line interface for UxPlay manager."""
    import argparse

    parser = argparse.ArgumentParser(
        description="UxPlay Integration Module - AirPlay to API bridge"
    )
    parser.add_argument(
        "--device-name",
        default="AirPlay-Pipeline",
        help="AirPlay device name (default: AirPlay-Pipeline)"
    )
    parser.add_argument(
        "--api-url",
        help="Main API URL (default: from config)"
    )
    parser.add_argument(
        "--uxplay-binary",
        help="Path to uxplay binary (auto-detected if not specified)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Video width (default: 1920)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Video height (default: 1080)"
    )
    parser.add_argument(
        "--no-forward",
        action="store_true",
        help="Don't auto-forward frames to API (for testing)"
    )

    args = parser.parse_args()

    # Create manager
    manager = UxPlayManager(
        uxplay_binary=args.uxplay_binary,
        device_name=args.device_name,
        main_api_url=args.api_url,
        auto_forward_to_api=not args.no_forward
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nShutting down...")
        manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start UxPlay
    print(f"Starting UxPlay AirPlay receiver...")
    print(f"  Device name: {args.device_name}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  API URL: {manager.main_api_url}")
    print(f"  Auto-forward: {not args.no_forward}")
    print()
    print("Waiting for visionOS device to connect...")
    print("(Press Ctrl+C to stop)")
    print()

    try:
        manager.start(width=args.width, height=args.height)

        # Monitor and print stats
        while True:
            time.sleep(5)
            stats = manager.get_stats()
            if stats["frames_captured"] > 0:
                print(f"[Stats] Captured: {stats['frames_captured']} | "
                      f"Forwarded: {stats['frames_forwarded']} | "
                      f"FPS: {stats['fps']:.1f}")

    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
