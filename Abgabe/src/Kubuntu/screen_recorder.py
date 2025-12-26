#!/usr/bin/env python3
"""
Screen recorder that captures the screen until keyboard interrupt.
Saves video to ./debugger_screencaptures/

FIXED: Memory-efficient version that writes frames directly to file
instead of storing them in memory.
"""

import cv2
import numpy as np
import mss
import os
from datetime import datetime
import signal
import sys
import time
import argparse


class ScreenRecorder:
    def __init__(self, output_dir="./debugger_screencaptures", fps=10.0,
                 scale=1.0, show_preview=False):
        """
        Initialize screen recorder with FPS limiting and direct-to-file writing.

        Args:
            output_dir: Directory to save recordings
            fps: Target frames per second (default: 10, lower uses less memory)
            scale: Scale factor for recording (0.5 = half size, saves memory/disk)
            show_preview: Show live preview window (uses more CPU)
        """
        self.output_dir = output_dir
        self.fps = fps
        self.scale = scale
        self.show_preview = show_preview
        self.recording = True
        self.frame_count = 0
        self.video_writer = None

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\nKeyboard interrupt received. Stopping recording...")
        self.recording = False

    def record(self):
        """Record the screen until interrupted - writes directly to file"""
        print("=" * 60)
        print("Screen Recording (Memory-Safe Mode)")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Target FPS: {self.fps}")
        print(f"Scale: {self.scale * 100:.0f}%")
        print(f"Preview: {'Enabled' if self.show_preview else 'Disabled'}")
        print("=" * 60)
        print("Press Ctrl+C to stop recording")
        print()

        # Calculate frame interval for FPS limiting
        frame_interval = 1.0 / self.fps

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")

        start_time = time.time()
        last_status_time = start_time

        try:
            with mss.mss() as sct:
                # Get the primary monitor
                monitor = sct.monitors[1]

                print(f"Monitor resolution: {monitor['width']}x{monitor['height']}")

                # Calculate scaled dimensions
                width = int(monitor['width'] * self.scale)
                height = int(monitor['height'] * self.scale)

                print(f"Recording resolution: {width}x{height}")
                print()

                # Initialize video writer on first frame
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    output_path, fourcc, self.fps, (width, height)
                )

                if not self.video_writer.isOpened():
                    print("ERROR: Could not open video writer!")
                    return

                print(f"Recording to: {output_path}")
                print()

                next_frame_time = time.time()

                while self.recording:
                    current_time = time.time()

                    # FPS limiting - only capture when it's time for next frame
                    if current_time < next_frame_time:
                        time.sleep(0.001)  # Small sleep to prevent busy-waiting
                        continue

                    next_frame_time += frame_interval

                    # Capture the screen
                    screenshot = sct.grab(monitor)

                    # Convert to numpy array
                    frame = np.array(screenshot)

                    # Convert BGRA to BGR (remove alpha channel)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Scale if needed
                    if self.scale != 1.0:
                        frame = cv2.resize(frame, (width, height),
                                         interpolation=cv2.INTER_AREA)

                    # Write frame directly to file (no memory accumulation!)
                    self.video_writer.write(frame)
                    self.frame_count += 1

                    # Show preview if enabled
                    if self.show_preview:
                        preview = cv2.resize(frame, (960, 540))  # 16:9 preview
                        cv2.putText(preview, f"Recording... Frame: {self.frame_count}",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                  (0, 0, 255), 2)
                        cv2.imshow("Screen Recording Preview", preview)
                        cv2.waitKey(1)

                    # Print status every 5 seconds
                    if current_time - last_status_time >= 5.0:
                        duration = current_time - start_time
                        actual_fps = self.frame_count / duration if duration > 0 else 0
                        print(f"Recording... {self.frame_count} frames | "
                              f"Duration: {duration:.1f}s | "
                              f"Actual FPS: {actual_fps:.1f}")
                        last_status_time = current_time

        finally:
            # Clean up
            if self.video_writer is not None:
                self.video_writer.release()

            if self.show_preview:
                cv2.destroyAllWindows()

            # Print final statistics
            total_duration = time.time() - start_time
            print()
            print("=" * 60)
            print("Recording Complete")
            print("=" * 60)
            print(f"Total frames: {self.frame_count}")
            print(f"Duration: {total_duration:.2f} seconds")
            print(f"Average FPS: {self.frame_count / total_duration:.2f}")
            print(f"Video saved to: {output_path}")
            print(f"File size: {self._get_file_size(output_path)}")
            print("=" * 60)

    def _get_file_size(self, filepath):
        """Get human-readable file size"""
        if not os.path.exists(filepath):
            return "N/A"

        size_bytes = os.path.getsize(filepath)

        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0

        return f"{size_bytes:.2f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Screen recorder that writes directly to file (memory-safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic recording at 10 fps (default, memory-safe)
  python screen_recorder.py

  # Higher quality at 15 fps
  python screen_recorder.py --fps 15

  # Lower quality, half resolution (saves disk space)
  python screen_recorder.py --fps 10 --scale 0.5

  # With live preview window
  python screen_recorder.py --fps 10 --preview

  # Slow motion recording
  python screen_recorder.py --fps 5 --scale 0.5
        """
    )

    parser.add_argument("--fps", type=float, default=10.0,
                       help="Target frames per second (default: 10, lower = less CPU/memory)")
    parser.add_argument("--scale", type=float, default=1.0,
                       help="Scale factor for video size (0.5 = half, 1.0 = full, default: 1.0)")
    parser.add_argument("--output-dir", type=str, default="./debugger_screencaptures",
                       help="Output directory (default: ./debugger_screencaptures)")
    parser.add_argument("--preview", action="store_true",
                       help="Show live preview window (uses more CPU)")

    args = parser.parse_args()

    # Validate arguments
    if args.fps <= 0 or args.fps > 60:
        print("ERROR: FPS must be between 0 and 60")
        sys.exit(1)

    if args.scale <= 0 or args.scale > 1.0:
        print("ERROR: Scale must be between 0 and 1.0")
        sys.exit(1)

    recorder = ScreenRecorder(
        output_dir=args.output_dir,
        fps=args.fps,
        scale=args.scale,
        show_preview=args.preview
    )
    recorder.record()


if __name__ == "__main__":
    main()
