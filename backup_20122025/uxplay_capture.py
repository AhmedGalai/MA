#!/usr/bin/env python3
"""
UxPlay Frame Capture Service
Monitors UxPlay frame output and provides on-demand capture functionality.
"""

import os
import time
import shutil
import requests
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UxPlayCapture:
    """
    Captures frames from UxPlay Docker container output.

    UxPlay writes frames to a shared directory, and this class
    reads them on-demand and forwards to the main API.
    """

    def __init__(self,
                 frame_dir: str = "./frames",
                 api_url: str = "http://localhost:8000"):
        """
        Initialize UxPlay capture service.

        Args:
            frame_dir: Directory where UxPlay writes frames
            api_url: URL of main API server
        """
        self.frame_dir = Path(frame_dir)
        self.latest_frame_path = self.frame_dir / "latest.jpg"
        self.api_url = api_url

        # Create frame directory if it doesn't exist
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"UxPlayCapture initialized: frame_dir={frame_dir}, api_url={api_url}")

    def is_frame_available(self, max_age_s: float = 10.0) -> bool:
        """
        Check if latest frame exists and is recent.

        Returns:
            True if frame is available and recent (< 2 seconds old)
        """
        if not self.latest_frame_path.exists():
            logger.warning(f"Frame file not found: {self.latest_frame_path}")
            return False

        # Check if frame was updated in last 2 seconds
        mtime = self.latest_frame_path.stat().st_mtime
        age = time.time() - mtime

        if age > max_age_s:
            logger.warning(f"Frame is stale (age: {age:.2f}s, max_age={max_age_s}s)")
            return False

        return True

    def capture_frame(self, allow_stale: bool = True, max_age_s: float = 10.0) -> Optional[np.ndarray]:
        """
        Capture current frame from UxPlay output.

        Returns:
            Frame as numpy array (BGR) or None if capture failed
        """
        if not self.is_frame_available(max_age_s=max_age_s):
            if not allow_stale:
                logger.error("No recent frame available from UxPlay")
                return None
            # If stale is allowed, continue and try to read whatever is there
            logger.warning("Using stale UxPlay frame (capture_frame allow_stale=True)")

        try:
            frame = cv2.imread(str(self.latest_frame_path))
            if frame is None:
                logger.error("Failed to read frame file")
                return None

            logger.info(f"Frame captured: shape={frame.shape}")
            return frame

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None

    def send_frame_to_api(self, frame: np.ndarray,
                          purpose: str = "general") -> bool:
        """
        Send captured frame to main API.

        Args:
            frame: Frame to send (BGR numpy array)
            purpose: Purpose of capture (for logging)

        Returns:
            True if sent successfully
        """
        try:
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Send to API
            payload = {
                "rgb_frame": f"data:image/jpeg;base64,{frame_b64}",
                "purpose": purpose
            }

            response = requests.post(
                f"{self.api_url}/receive_frame",
                json=payload,
                timeout=5.0
            )

            if response.status_code == 200:
                logger.info(f"Frame sent successfully for purpose: {purpose}")
                return True
            else:
                logger.error(f"API returned status {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            logger.error("Timeout sending frame to API")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("Connection error - is main_api.py running?")
            return False
        except Exception as e:
            logger.error(f"Error sending frame to API: {e}")
            return False

    def capture_and_send(self, purpose: str = "general") -> bool:
        """
        Capture frame and send to API in one operation.

        Args:
            purpose: Purpose of capture

        Returns:
            True if successful
        """
        frame = self.capture_frame(allow_stale=True)
        if frame is None:
            return False

        return self.send_frame_to_api(frame, purpose)


def main():
    """Command-line interface for UxPlay capture."""
    import argparse

    parser = argparse.ArgumentParser(description="UxPlay Frame Capture Service")
    parser.add_argument("--action", choices=["capture", "monitor"],
                        default="capture",
                        help="Action to perform")
    parser.add_argument("--purpose", default="general",
                        help="Purpose of capture (for logging)")
    parser.add_argument("--frame-dir", default="./frames",
                        help="Directory where UxPlay writes frames")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Main API URL")

    args = parser.parse_args()

    capture = UxPlayCapture(
        frame_dir=args.frame_dir,
        api_url=args.api_url
    )

    if args.action == "capture":
        # Single capture
        logger.info(f"Attempting single capture for purpose: {args.purpose}")
        success = capture.capture_and_send(args.purpose)

        if success:
            print("[SUCCESS] Frame captured and sent to API")
            return 0
        else:
            print("[FAILED] Could not capture/send frame")
            return 1

    elif args.action == "monitor":
        # Monitor mode (for testing)
        print("[INFO] Monitoring UxPlay frames... (Ctrl+C to stop)")
        try:
            while True:
                if capture.is_frame_available():
                    age = time.time() - capture.latest_frame_path.stat().st_mtime
                    print(f"[INFO] Frame available (age: {age:.2f}s)")
                else:
                    print("[WARN] No recent frame")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[INFO] Monitoring stopped")
            return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
