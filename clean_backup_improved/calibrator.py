#!/usr/bin/env python3
"""
UxPlay Camera Calibration Tool

Calibrates camera intrinsics (K) and distortion coefficients (dist) for UxPlay feed
using ArUco marker board detection.

Usage:
    python calibrator.py [--show-preview] [--output intrinsics.json]
"""
import json, time, argparse
import numpy as np
import cv2
from pathlib import Path

DICT_NAME = "DICT_4X4_250"
ROWS, COLS = 3, 4              # markersY, markersX
MARKER_SIZE = 0.03             # meters
SEP = 0.01                     # meters

# ---- plug in how you get frames (from your UxPlayCapture) ----
from basic_main_api_with_uxplay import UxPlayCapture, find_uxplay_binary

def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    try:
        board = cv2.aruco.GridBoard_create(COLS, ROWS, MARKER_SIZE, SEP, dictionary)
    except Exception:
        board = cv2.aruco.GridBoard((COLS, ROWS), MARKER_SIZE, SEP, dictionary)
    return dictionary, board

def detect(dictionary, frame_bgr):
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(frame_bgr)
    except Exception:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(frame_bgr, dictionary, parameters=params)
    return corners, ids

def assess_calibration_quality(ret, dist):
    """Assess the quality of calibration based on reprojection error and distortion coefficients"""
    quality = "GOOD"
    warnings = []

    # Check reprojection error
    if ret > 1.0:
        quality = "POOR"
        warnings.append(f"High reprojection error: {ret:.3f} px (should be < 1.0)")
    elif ret > 0.5:
        quality = "FAIR"
        warnings.append(f"Moderate reprojection error: {ret:.3f} px (target < 0.5)")

    # Check distortion coefficients magnitude
    k1, k2, p1, p2, k3 = dist.reshape(-1)[:5] if len(dist.reshape(-1)) >= 5 else (dist.reshape(-1).tolist() + [0]*5)[:5]

    if abs(k1) > 0.5 or abs(k2) > 0.5:
        warnings.append(f"Large radial distortion detected (k1={k1:.3f}, k2={k2:.3f})")

    if abs(p1) > 0.01 or abs(p2) > 0.01:
        warnings.append(f"Tangential distortion detected (p1={p1:.3f}, p2={p2:.3f})")

    return quality, warnings


def save_calibration_images(frames, K, dist, output_dir="calibration_output"):
    """Save sample frames showing before/after undistortion"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nSaving calibration samples to {output_dir}/")

    # Save 3 sample frames
    for i, frame in enumerate(frames[:3]):
        # Original
        cv2.imwrite(str(output_path / f"sample_{i}_original.jpg"), frame)

        # Undistorted
        h, w = frame.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, K, dist, None, new_K)
        cv2.imwrite(str(output_path / f"sample_{i}_undistorted.jpg"), undistorted)

    print(f"  Saved {min(3, len(frames))} sample pairs (original + undistorted)")


def main():
    parser = argparse.ArgumentParser(description="Calibrate UxPlay camera intrinsics and distortion")
    parser.add_argument("--show-preview", action="store_true", help="Show live preview window during capture")
    parser.add_argument("--output", default="intrinsics.json", help="Output calibration file")
    parser.add_argument("--min-samples", type=int, default=15, help="Minimum number of samples required")
    parser.add_argument("--save-samples", action="store_true", help="Save sample images showing undistortion")
    args = parser.parse_args()

    uxplay_bin = find_uxplay_binary(None)
    cap = UxPlayCapture(uxplay_binary=uxplay_bin, device_name="AirPlay-Pipeline", width=1920, height=1080)
    cap.start()

    dictionary, board = make_board()

    all_corners = []
    all_ids = []
    marker_counts = []
    all_frames = []  # Store frames for later visualization
    img_size = None

    print("=" * 70)
    print("UxPlay Camera Calibration")
    print("=" * 70)
    print(f"ArUco Dictionary: {DICT_NAME}")
    print(f"Board: {COLS}x{ROWS} markers, {MARKER_SIZE}m size, {SEP}m separation")
    print(f"Target samples: {args.min_samples}+")
    print()
    print("Instructions:")
    print("  1. Show the ArUco board to the camera")
    print("  2. Move it to different positions, angles, and distances")
    print("  3. Cover all areas of the frame (corners, edges, center)")
    print("  4. Press Ctrl+C when you have enough samples")
    print()
    print("Collecting frames...")
    print()

    if args.show_preview:
        cv2.namedWindow("Calibration Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration Preview", 960, 540)

    try:
        frame_count = 0
        while True:
            frame, ts = cap.get_latest()
            if frame is None:
                time.sleep(0.02)
                continue

            frame_count += 1
            img_size = (frame.shape[1], frame.shape[0])

            corners, ids = detect(dictionary, frame)

            # Show preview if requested
            if args.show_preview and frame_count % 3 == 0:  # Update every 3rd frame to reduce lag
                preview = frame.copy()
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(preview, corners, ids)

                # Add status text
                status_text = f"Samples: {len(all_corners)}/{args.min_samples}+ | Markers: {len(ids) if ids is not None else 0}"
                cv2.putText(preview, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Calibration Preview", preview)
                cv2.waitKey(1)

            if ids is not None and len(ids) >= 6:  # require some markers
                all_corners.append(corners)
                all_ids.append(ids)
                marker_counts.append(len(ids))
                all_frames.append(frame.copy())
                print(f"✓ Sample {len(all_corners):3d} - {len(ids):2d} markers detected")

            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\n\nCalibration capture stopped by user")
    finally:
        cap.stop()
        if args.show_preview:
            cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print(f"Captured {len(all_corners)} samples")
    print()

    if len(all_corners) < args.min_samples:
        raise SystemExit(f"✗ Not enough samples. Need at least {args.min_samples}, got {len(all_corners)}")

    # OpenCV wants concatenated lists + per-frame counts
    corners_concat = [c for frame_corners in all_corners for c in frame_corners]
    ids_concat = np.vstack(all_ids)
    counter = np.array(marker_counts, dtype=np.int32)

    # Calibrate
    if not hasattr(cv2.aruco, "calibrateCameraAruco"):
        raise SystemExit("✗ Your OpenCV build lacks cv2.aruco.calibrateCameraAruco")

    print("Running calibration...")
    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
        corners_concat, ids_concat, counter, board, img_size, None, None
    )
    print("✓ Calibration complete")
    print()

    # Assess quality
    quality, warnings = assess_calibration_quality(ret, dist)

    print("=" * 70)
    print("Calibration Results")
    print("=" * 70)
    print(f"Quality: {quality}")
    print(f"Reprojection Error: {ret:.4f} pixels")
    print()
    print("Camera Matrix (K):")
    print(K)
    print()
    print("Distortion Coefficients (k1, k2, p1, p2, k3):")
    print(dist.reshape(-1))
    print()

    if warnings:
        print("⚠ Warnings:")
        for w in warnings:
            print(f"  - {w}")
        print()

    # Save calibration
    out = {
        "image_size": [img_size[0], img_size[1]],
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "reproj_error": float(ret),
        "quality": quality,
        "num_samples": len(all_corners),
        "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ Saved calibration to: {args.output}")

    # Save sample images if requested
    if args.save_samples and all_frames:
        save_calibration_images(all_frames, K, dist)

    print()
    print("=" * 70)
    print("Next steps:")
    print(f"  1. Review the calibration quality: {quality}")
    if quality != "GOOD":
        print("  2. If quality is poor, re-run with more varied samples")
    print(f"  3. The system will use {args.output} for undistortion")
    print("=" * 70)

if __name__ == "__main__":
    main()
