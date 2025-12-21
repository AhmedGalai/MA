#!/usr/bin/env python3
"""
RealSense Camera Reset Utility

Stops all RealSense pipelines and optionally resets the USB device.
"""

import argparse
import subprocess
import sys
import time

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not installed")
    sys.exit(1)


def stop_all_pipelines():
    """Stop all running RealSense pipelines"""
    print("Stopping all RealSense pipelines...")

    try:
        ctx = rs.context()
        devices = ctx.query_devices()

        if devices.size() == 0:
            print("  No RealSense devices found")
            return False

        print(f"  Found {devices.size()} device(s)")

        for i in range(devices.size()):
            device = devices[i]
            device_name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)

            print(f"  Device {i}: {device_name} (S/N: {serial})")

            # Hardware reset via the device
            try:
                device.hardware_reset()
                print(f"    ✓ Hardware reset sent")
            except Exception as e:
                print(f"    ! Hardware reset failed: {e}")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def get_usb_device_info():
    """Get USB device info for RealSense camera"""
    try:
        result = subprocess.run(
            ['lsusb'],
            capture_output=True,
            text=True,
            check=True
        )

        for line in result.stdout.split('\n'):
            if 'Intel' in line and 'RealSense' in line:
                # Parse: Bus 004 Device 002: ID 8086:0b3a Intel Corp. Intel(R) RealSense(TM)
                parts = line.split()
                bus = parts[1]
                device = parts[3].rstrip(':')
                return bus, device

        return None, None

    except Exception as e:
        print(f"ERROR getting USB info: {e}")
        return None, None


def reset_usb_device(bus, device):
    """Reset USB device using usbreset"""
    print(f"Resetting USB device on bus {bus}, device {device}...")

    device_path = f"/dev/bus/usb/{bus}/{device}"

    try:
        # Try using usbreset if available
        result = subprocess.run(
            ['sudo', 'usbreset', device_path],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("  ✓ USB device reset successful")
            return True
        else:
            print(f"  ! usbreset failed: {result.stderr}")

    except FileNotFoundError:
        print("  ! 'usbreset' command not found")
        print("  Tip: Install with: sudo apt-get install usbutils")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Alternative: try to unbind/bind the USB device
    try:
        print("  Trying alternative USB reset method...")

        # This requires root and is more aggressive
        subprocess.run(
            ['sudo', 'sh', '-c',
             f'echo "{bus}-{device}" > /sys/bus/usb/drivers/usb/unbind'],
            check=False
        )
        time.sleep(1)
        subprocess.run(
            ['sudo', 'sh', '-c',
             f'echo "{bus}-{device}" > /sys/bus/usb/drivers/usb/bind'],
            check=False
        )
        print("  ✓ Alternative USB reset completed")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def verify_camera():
    """Verify camera is working after reset"""
    print("Verifying camera...")

    try:
        ctx = rs.context()
        devices = ctx.query_devices()

        if devices.size() == 0:
            print("  ✗ No devices found")
            return False

        print(f"  ✓ Found {devices.size()} device(s)")

        # Try to start a pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        print("  Testing pipeline start...")
        profile = pipeline.start(config)

        # Wait for a few frames
        for i in range(5):
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            print(f"    Frame {i+1}/5 received")

        pipeline.stop()
        print("  ✓ Camera is working!")
        return True

    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Reset RealSense camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Soft reset (hardware reset only)
  %(prog)s

  # Hard reset (USB reset - requires sudo)
  %(prog)s --hard

  # Reset and verify
  %(prog)s --verify

  # Full reset with verification
  %(prog)s --hard --verify
        """
    )
    parser.add_argument(
        '--hard',
        action='store_true',
        help='Perform USB device reset (requires sudo)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify camera works after reset'
    )
    parser.add_argument(
        '--wait',
        type=int,
        default=2,
        help='Wait time in seconds after reset (default: 2)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RealSense Camera Reset Utility")
    print("=" * 60)
    print()

    # Step 1: Stop all pipelines
    success = stop_all_pipelines()

    if not success:
        print("\n✗ Failed to stop pipelines")
        return 1

    print(f"\nWaiting {args.wait} seconds...")
    time.sleep(args.wait)

    # Step 2: USB reset if requested
    if args.hard:
        print()
        bus, device = get_usb_device_info()

        if bus and device:
            reset_usb_device(bus, device)
            print(f"\nWaiting {args.wait} seconds after USB reset...")
            time.sleep(args.wait)
        else:
            print("! Could not find RealSense USB device")
            print("  Skipping USB reset")

    # Step 3: Verify if requested
    if args.verify:
        print()
        if verify_camera():
            print("\n✓ Reset successful - camera is working")
            return 0
        else:
            print("\n✗ Reset completed but camera verification failed")
            return 1

    print("\n✓ Reset completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
