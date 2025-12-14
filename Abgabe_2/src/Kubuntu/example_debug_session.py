#!/usr/bin/env python3
"""
Example Debug Session Script

This script demonstrates how to programmatically use the DebugViewer
to monitor the pose estimation pipeline with custom configurations.

Usage:
    python3 example_debug_session.py [--api-url http://localhost:8000]

Author: Debug Tools
Date: 2025-12-14
"""

import sys
from pathlib import Path

# Add src/Kubuntu to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from debug_viewer import DebugViewer
import logging

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_minimal():
    """Minimal example - just launch the debug viewer."""
    logger.info("Starting minimal debug session...")

    # Create viewer with defaults
    viewer = DebugViewer()

    # Run the GUI
    viewer.run()


def example_custom_api():
    """Example with custom API URL."""
    api_url = "http://192.168.1.100:8000"  # Change to your API server
    logger.info(f"Connecting to API at {api_url}")

    viewer = DebugViewer(api_url=api_url, width=1400, height=900)
    viewer.run()


def example_with_logging():
    """Example with detailed logging."""
    logger.info("Starting debug session with verbose logging...")

    viewer = DebugViewer()

    # You can add custom event handlers here
    # For example, save statistics periodically:

    def save_stats_callback():
        """Save statistics to file (would need implementation)."""
        stats = viewer.stats
        logger.info(f"Current stats: {stats['total_frames']} frames, "
                   f"{stats['successful_estimates']} successful")

    # Launch viewer
    viewer.run()


def example_programmatic_control():
    """
    Example showing how to control viewer programmatically.

    Note: This is a concept example. The actual implementation would
    need to account for threading and tkinter's main loop.
    """
    logger.info("Starting programmatic control example...")

    viewer = DebugViewer()

    # Example: Set initial refresh rate
    viewer.update_rate_hz = 3.0

    # Example: Update statistics manually
    viewer.stats['total_frames'] = 100
    viewer.stats['successful_estimates'] = 95

    # Launch viewer
    viewer.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Example Debug Session for Pose Estimation Pipeline"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=['minimal', 'custom', 'logging', 'programmatic'],
        default='minimal',
        help="Which example to run (default: minimal)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Custom API URL (default: http://127.0.0.1:8000)"
    )

    args = parser.parse_args()

    try:
        if args.example == 'minimal':
            example_minimal()
        elif args.example == 'custom':
            if args.api_url:
                print(f"Using custom API URL: {args.api_url}")
            example_custom_api()
        elif args.example == 'logging':
            example_with_logging()
        elif args.example == 'programmatic':
            example_programmatic_control()

    except KeyboardInterrupt:
        logger.info("Debug session interrupted by user")
    except Exception as e:
        logger.error(f"Error during debug session: {e}", exc_info=True)
