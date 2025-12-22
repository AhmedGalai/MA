#!/usr/bin/env python3
"""
Quick RealSense detection check for macOS (pyrealsense2-macosx) or standard pyrealsense2.
"""

from __future__ import annotations

import sys


def load_rs():
    try:
        import pyrealsense2 as rs  # type: ignore
        return rs
    except ModuleNotFoundError:
        import pyrealsense2_macosx as rs  # type: ignore
        return rs


def main() -> int:
    try:
        rs = load_rs()
    except ModuleNotFoundError as exc:
        print(f"RealSense SDK module not found: {exc}")
        return 2

    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        count = devices.size()
        if count == 0:
            print("No RealSense devices detected.")
            return 1

        print(f"Detected {count} RealSense device(s):")
        for i in range(count):
            dev = devices[i]
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            fw = dev.get_info(rs.camera_info.firmware_version)
            usb = dev.get_info(rs.camera_info.usb_type_descriptor)
            print(f"- {name} | S/N {serial} | FW {fw} | USB {usb}")
        return 0
    except Exception as exc:
        print(f"RealSense detection failed: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
