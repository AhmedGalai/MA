#!/usr/bin/env python3
"""
UxPlay Integration Example

This script demonstrates how to integrate UxPlay with the main API.
It starts UxPlay and automatically forwards all frames from visionOS devices
to the main API for processing.
"""

import sys
import time
import signal
import logging
from uxplay_module import UxPlayManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def custom_frame_callback(frame):
    """
    Optional: Custom processing for each frame.

    This is called for every frame before it's forwarded to the API.
    You can add custom logic here (e.g., face detection, preprocessing, etc.)

    Args:
        frame: BGR numpy array from UxPlay
    """
    # Example: Log frame shape
    # logger.debug(f"Received frame: {frame.shape}")
    pass


def main():
    """Run UxPlay integration with main API."""

    print("=" * 60)
    print("UxPlay Integration for Pose Estimation Pipeline")
    print("=" * 60)
    print()
    print("This service creates an AirPlay receiver that:")
    print("  1. Receives video from visionOS devices")
    print("  2. Extracts RGB frames in real-time")
    print("  3. Forwards frames to the main API")
    print()
    print("Make sure main_api.py is running before connecting devices!")
    print()
    print("=" * 60)
    print()

    # Create UxPlay manager
    manager = UxPlayManager(
        device_name="AirPlay-Pipeline",
        auto_forward_to_api=True,
        frame_callback=custom_frame_callback  # Optional
    )

    # Handle shutdown signals
    def signal_handler(sig, frame):
        print("\n\nShutdown signal received...")
        manager.stop()
        print("UxPlay stopped successfully")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start UxPlay
        print("Starting UxPlay AirPlay receiver...")
        manager.start(width=1920, height=1080)

        print()
        print("✓ UxPlay is running!")
        print()
        print("Instructions:")
        print("  1. On your visionOS device, open Control Center")
        print("  2. Tap 'Screen Mirroring' or 'AirPlay'")
        print("  3. Select 'AirPlay-Pipeline' from the list")
        print("  4. Frames will automatically be sent to the main API")
        print()
        print("Press Ctrl+C to stop")
        print()
        print("-" * 60)

        # Monitor stats
        last_stats_time = time.time()

        while True:
            time.sleep(1)

            # Print stats every 10 seconds
            if time.time() - last_stats_time >= 10:
                stats = manager.get_stats()

                if stats["frames_captured"] > 0:
                    print(f"\n[Statistics]")
                    print(f"  Frames captured:  {stats['frames_captured']}")
                    print(f"  Frames forwarded: {stats['frames_forwarded']}")
                    print(f"  Current FPS:      {stats['fps']:.1f}")
                    print(f"  Resolution:       {stats['resolution']}")
                    print()
                else:
                    print("[Waiting for visionOS device to connect...]")

                last_stats_time = time.time()

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        manager.stop()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
