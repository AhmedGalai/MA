#!/usr/bin/env python3
"""
RealSense Camera Diagnostic Tool

Tests RealSense camera connection and frame capture to diagnose issues.
"""

import pyrealsense2 as rs
import numpy as np
import sys

def test_realsense():
    """Test RealSense camera connection and frame capture."""

    print("=" * 60)
    print("RealSense Camera Diagnostic Tool")
    print("=" * 60)
    print()

    # Step 1: Check for connected devices
    print("Step 1: Checking for connected RealSense devices...")
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("❌ ERROR: No RealSense devices found!")
        print("   - Check USB connection")
        print("   - Try different USB port (USB 3.0 recommended)")
        print("   - Check if device is recognized: lsusb | grep Intel")
        return False

    print(f"✓ Found {len(devices)} RealSense device(s)")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device.get_info(rs.camera_info.name)}")
        print(f"    Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"    Firmware: {device.get_info(rs.camera_info.firmware_version)}")
    print()

    # Step 2: Check if device is already in use
    print("Step 2: Checking if device is available...")
    try:
        pipeline = rs.pipeline()
        config = rs.config()

        # Try to configure streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        print("✓ Device is not in use by another process")
    except RuntimeError as e:
        print(f"❌ ERROR: Device might be in use - {e}")
        print("   - Close any other programs using the camera")
        print("   - Kill any stuck processes: pkill -9 python")
        return False
    print()

    # Step 3: Start pipeline
    print("Step 3: Starting RealSense pipeline...")
    try:
        profile = pipeline.start(config)
        print("✓ Pipeline started successfully")

        # Get device info
        device = profile.get_device()
        print(f"  Active device: {device.get_info(rs.camera_info.name)}")
    except RuntimeError as e:
        print(f"❌ ERROR: Failed to start pipeline - {e}")
        return False
    print()

    # Step 4: Test frame capture without warm-up
    print("Step 4: Testing frame capture (first attempt, may timeout)...")
    try:
        frames = pipeline.wait_for_frames(timeout_ms=2000)
        print("✓ First frame captured successfully!")
    except RuntimeError as e:
        print(f"⚠ WARNING: First frame timeout (this is sometimes normal) - {e}")
    print()

    # Step 5: Warm up camera
    print("Step 5: Warming up camera (skipping 30 frames)...")
    successful_frames = 0
    failed_frames = 0

    for i in range(30):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            successful_frames += 1
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/30 frames captured")
        except RuntimeError as e:
            failed_frames += 1
            print(f"  ⚠ Frame {i+1} timeout")

    print(f"  Warm-up complete: {successful_frames} successful, {failed_frames} failed")

    if successful_frames == 0:
        print("❌ ERROR: No frames captured during warm-up!")
        print("   - Camera might be defective")
        print("   - Try unplugging and replugging the camera")
        print("   - Check USB bandwidth (disconnect other USB devices)")
        pipeline.stop()
        return False

    print()

    # Step 6: Test stable frame capture
    print("Step 6: Testing stable frame capture (10 frames)...")
    successful = 0
    failed = 0

    for i in range(10):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)

            # Check if we got both color and depth
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if depth_frame and color_frame:
                successful += 1

                # Get frame data
                depth_data = np.asanyarray(depth_frame.get_data())
                color_data = np.asanyarray(color_frame.get_data())

                if i == 0:
                    print(f"  Frame info:")
                    print(f"    Color: {color_data.shape}, dtype: {color_data.dtype}")
                    print(f"    Depth: {depth_data.shape}, dtype: {depth_data.dtype}")
            else:
                failed += 1
                print(f"  ⚠ Frame {i+1}: Missing color or depth")

        except RuntimeError as e:
            failed += 1
            print(f"  ⚠ Frame {i+1}: Timeout - {e}")

    print(f"  Capture test: {successful}/10 successful")
    print()

    # Step 7: Clean up
    print("Step 7: Stopping pipeline...")
    try:
        pipeline.stop()
        print("✓ Pipeline stopped cleanly")
    except Exception as e:
        print(f"⚠ Warning during stop: {e}")
    print()

    # Final verdict
    print("=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    if successful >= 8:
        print("✓ SUCCESS: Camera is working properly!")
        print("  Your RealSense camera is functioning correctly.")
        print("  The API should work with this camera.")
        return True
    elif successful >= 5:
        print("⚠ PARTIAL: Camera is working but unstable")
        print("  The camera works but has frequent timeouts.")
        print("  Recommendations:")
        print("    - Use a high-quality USB 3.0 cable")
        print("    - Connect to USB 3.0 port (not USB 2.0)")
        print("    - Reduce USB bandwidth by disconnecting other devices")
        print("    - Update RealSense firmware")
        return True
    else:
        print("❌ FAILURE: Camera is not working")
        print("  The camera cannot capture frames reliably.")
        print("  Recommendations:")
        print("    - Check physical connection")
        print("    - Try different USB port")
        print("    - Restart computer")
        print("    - Update RealSense SDK")
        print("    - Check camera permissions: sudo usermod -a -G video $USER")
        return False


if __name__ == "__main__":
    print()
    success = test_realsense()
    print()

    if success:
        sys.exit(0)
    else:
        sys.exit(1)
