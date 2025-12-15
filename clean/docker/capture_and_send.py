#!/usr/bin/env python3
"""
UxPlay Frame Capture and Forwarding Script

Captures frames from UxPlay video output and sends them to the main API.
Continuously discards old frames and only keeps the latest one.
"""

import cv2
import base64
import time
import requests
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
API_HOST = os.getenv('API_HOST', 'host.docker.internal')
API_PORT = os.getenv('API_PORT', '8000')
API_URL = f"http://{API_HOST}:{API_PORT}"

# Frame capture configuration
CAPTURE_FPS = 10  # Capture ~10 frames per second
CAPTURE_INTERVAL = 1.0 / CAPTURE_FPS

# Video capture device (UxPlay creates a virtual video device)
# Try different sources
VIDEO_SOURCES = [
    0,  # Default camera
    '/tmp/uxplay_video.fifo',  # Named pipe if UxPlay configured to use it
    'tcp://127.0.0.1:5000',  # TCP stream if UxPlay streams via TCP
]


def find_video_source():
    """
    Try to find a working video source from UxPlay.

    Returns:
        VideoCapture object or None
    """
    logger.info("Searching for UxPlay video source...")

    for source in VIDEO_SOURCES:
        logger.info(f"Trying video source: {source}")
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    logger.info(f"✓ Found working video source: {source}")
                    return cap
                else:
                    logger.warning(f"  Source opened but no frames: {source}")
                    cap.release()
            else:
                logger.warning(f"  Could not open source: {source}")
        except Exception as e:
            logger.warning(f"  Error trying source {source}: {e}")

    return None


def encode_frame_to_base64(frame):
    """
    Encode OpenCV frame to base64 JPEG.

    Args:
        frame: OpenCV frame (numpy array)

    Returns:
        str: Base64-encoded JPEG image with data URL prefix
    """
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Convert to base64
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    # Add data URL prefix
    return f"data:image/jpeg;base64,{frame_b64}"


def send_frame_to_api(frame_b64):
    """
    Send frame to main API.

    Args:
        frame_b64: Base64-encoded frame with data URL prefix

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        payload = {
            'rgb_frame': frame_b64,
            'timestamp': time.time(),
            'purpose': 'general'
        }

        response = requests.post(
            f"{API_URL}/receive_frame",
            json=payload,
            timeout=2
        )

        if response.status_code == 200:
            return True
        else:
            logger.warning(f"API returned status {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        logger.warning("API request timed out")
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to API at {API_URL}")
        return False
    except Exception as e:
        logger.error(f"Error sending frame: {e}")
        return False


def main():
    """Main capture and forward loop."""
    logger.info("=" * 60)
    logger.info("UxPlay Frame Capture & Forward Service")
    logger.info("=" * 60)
    logger.info(f"API URL: {API_URL}")
    logger.info(f"Capture FPS: {CAPTURE_FPS}")
    logger.info("")

    # Wait for UxPlay to start
    logger.info("Waiting for UxPlay to start...")
    time.sleep(5)

    # Find video source
    cap = find_video_source()

    if cap is None:
        logger.error("No video source found! UxPlay may not be streaming yet.")
        logger.info("This service will keep running and retry periodically.")
        logger.info("Connect your visionOS device to start streaming.")

        # Keep retrying
        while True:
            time.sleep(10)
            logger.info("Retrying video source detection...")
            cap = find_video_source()
            if cap is not None:
                break

    logger.info("✓ Video capture initialized")
    logger.info("Starting frame capture loop...")
    logger.info("")

    frame_count = 0
    success_count = 0
    last_log_time = time.time()

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()

            if not ret or frame is None:
                logger.warning("Failed to capture frame, reconnecting...")
                cap.release()
                time.sleep(2)
                cap = find_video_source()
                if cap is None:
                    logger.error("Lost video source, retrying in 10s...")
                    time.sleep(10)
                continue

            # Encode frame
            frame_b64 = encode_frame_to_base64(frame)

            # Send to API
            success = send_frame_to_api(frame_b64)

            frame_count += 1
            if success:
                success_count += 1

            # Log progress every 10 seconds
            now = time.time()
            if now - last_log_time >= 10.0:
                success_rate = (success_count / frame_count * 100) if frame_count > 0 else 0
                logger.info(
                    f"Stats: {frame_count} frames captured, "
                    f"{success_count} sent ({success_rate:.1f}% success)"
                )
                last_log_time = now

            # Wait before next capture
            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        if cap is not None:
            cap.release()
        logger.info("Capture service stopped")


if __name__ == "__main__":
    main()
