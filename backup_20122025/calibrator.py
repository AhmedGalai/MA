#!/usr/bin/env python3
import json, time
import numpy as np
import cv2

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

def main():
    uxplay_bin = find_uxplay_binary(None)
    cap = UxPlayCapture(uxplay_binary=uxplay_bin, device_name="AirPlay-Pipeline", width=1920, height=1080)
    cap.start()

    dictionary, board = make_board()

    all_corners = []
    all_ids = []
    marker_counts = []
    img_size = None

    print("Collecting frames... press Ctrl+C when you have ~30-80 good views.")
    try:
        while True:
            frame, ts = cap.get_latest()
            if frame is None:
                time.sleep(0.02)
                continue
            img_size = (frame.shape[1], frame.shape[0])

            corners, ids = detect(dictionary, frame)
            if ids is not None and len(ids) >= 6:  # require some markers
                all_corners.append(corners)
                all_ids.append(ids)
                marker_counts.append(len(ids))
                print(f"got sample {len(all_corners)} (markers {len(ids)})")
            time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()

    if len(all_corners) < 15:
        raise SystemExit("Not enough samples. Get more varied views.")

    # OpenCV wants concatenated lists + per-frame counts
    corners_concat = [c for frame_corners in all_corners for c in frame_corners]
    ids_concat = np.vstack(all_ids)
    counter = np.array(marker_counts, dtype=np.int32)

    # Calibrate
    if not hasattr(cv2.aruco, "calibrateCameraAruco"):
        raise SystemExit("Your OpenCV build lacks cv2.aruco.calibrateCameraAruco")

    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
        corners_concat, ids_concat, counter, board, img_size, None, None
    )

    out = {
        "image_size": [img_size[0], img_size[1]],
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "reproj_error": float(ret),
    }
    with open("intrinsics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote intrinsics.json")
    print("Reproj error:", ret)
    print("K:\n", K)
    print("dist:\n", dist.reshape(-1))

if __name__ == "__main__":
    main()
