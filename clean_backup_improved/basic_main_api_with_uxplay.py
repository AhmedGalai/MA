#!/usr/bin/env python3
"""
basic_main_api_with_uxplay.py

Single-process Flask API that:
- runs UxPlay (AirPlay receiver)
- captures raw BGR frames from UxPlay stdout (no network forwarding)
- processes frames at a configurable FPS (AruCo detection + board pose)
- serves VisionOS clients:
  - RGB feed endpoint (base64 JPEG)
  - Pose endpoint (base64 JPEG + ArUco board pose as 4x4 matrix + rvec/tvec + ids)

Notes:
- UxPlay raw video output requires you to know width/height ahead of time.
- Pose estimation needs camera intrinsics K. This file uses a simple default K
  (good enough to get started; replace with real intrinsics when you have them).
"""

import os
import sys
import time
import signal
import base64
import logging
import threading
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

from rich import print, pretty
pretty.install()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("basic_main_api_with_uxplay")

# Head pose storage (from visionOS)
head_pose_latest = None
head_pose_meta = {
    "receive_time": None,
    "reception_count": 0,
    "last_reception_time": None,
    "reception_rate": None,
}
head_pose_lock = threading.Lock()


# -----------------------------
# ArUco config (edit as needed)
# -----------------------------
@dataclass
class ArucoConfig:
    dictionary_name: str = "DICT_4X4_50"  # common default
    rows: int = 3                        # markersY
    cols: int = 4                        # markersX
    marker_size_m: float = 0.03          # marker side length in meters
    separation_m: float = 0.01           # gap between markers in meters
    draw_axes: bool = True
    axis_length_m: float = 0.06          # axis length in meters (visualization)


# -----------------------------
# UxPlay capture
# -----------------------------
class UxPlayCapture:
    """
    Spawns UxPlay and reads raw BGR frames from stdout.

    Important: You MUST supply width/height correctly.
    """

    def __init__(
        self,
        uxplay_binary: str,
        device_name: str,
        width: int,
        height: int,
    ):
        self.uxplay_binary = uxplay_binary
        self.device_name = device_name
        self.width = int(width)
        self.height = int(height)

        self.process: Optional[subprocess.Popen] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: Optional[float] = None
        self.frames_received = 0

    def start(self) -> None:
        if self.running:
            return
        frame_pipeline = "videoconvert ! video/x-raw,format=BGR ! fdsink fd=1 sync=false"
        cmd = [
            self.uxplay_binary,
            "-n", self.device_name,
            "-vsync", "no",
            "-as", "0",  # disable audio
            "-vs", frame_pipeline,
        ]

        logger.info("Starting UxPlay: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self.process.stdout is None:
            raise RuntimeError("UxPlay stdout pipe is None")

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            except Exception:
                pass
            self.process = None

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._latest_frame is None:
                return None, None
            return self._latest_frame.copy(), self._latest_ts

    def _read_loop(self) -> None:
        logger.info("UxPlay capture thread started")
        w, h = self.width, self.height
        frame_size = w * h * 3  # BGR
        buf = bytearray()

        try:
            while self.running and self.process and self.process.stdout:
                chunk = self.process.stdout.read(frame_size - len(buf))
                if not chunk:
                    if self.running:
                        logger.warning("UxPlay stream ended")
                    break

                buf.extend(chunk)
                if len(buf) < frame_size:
                    continue

                frame_data = bytes(buf[:frame_size])
                buf = buf[frame_size:]

                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, 3))

                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = time.time()
                    self.frames_received += 1

        except Exception as e:
            if self.running:
                logger.error("Error in UxPlay read loop: %s", e, exc_info=True)
        finally:
            logger.info("UxPlay capture thread stopped")


# -----------------------------
# ArUco processing (rate-limited)
# -----------------------------
def _get_aruco_dictionary(name: str):
    name = name.strip()
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _make_grid_board(cfg: ArucoConfig, dictionary):
    # OpenCV API differs slightly across versions; try both.
    try:
        # older OpenCV
        return cv2.aruco.GridBoard_create(
            cfg.cols, cfg.rows, cfg.marker_size_m, cfg.separation_m, dictionary
        )
    except Exception:
        # newer OpenCV
        return cv2.aruco.GridBoard(
            (cfg.cols, cfg.rows), cfg.marker_size_m, cfg.separation_m, dictionary
        )


def _load_intrinsics(path: str):
    import json, numpy as np
    with open(path, "r") as f:
        d = json.load(f)
    K = np.array(d["K"], dtype=np.float32)
    dist = np.array(d.get("dist", [0,0,0,0,0]), dtype=np.float32).reshape(-1, 1)
    return K, dist


def _rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = tvec.reshape(3).astype(np.float32)
    return T


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [x,y,z,w].
    """
    R = R.astype(np.float64)
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([x, y, z, w], dtype=np.float32)


class ArucoProcessor:
    """
    Runs ArUco detection at a fixed FPS, using the latest captured frame.
    Stores:
      - latest raw frame (for /rgb)
      - latest annotated frame (for /aruco)
      - latest board pose (for overlay)
    """

    def __init__(self, capture: UxPlayCapture, cfg: ArucoConfig, process_fps: float):
        self.capture = capture
        self.cfg = cfg
        self.process_fps = float(process_fps)

        self.dictionary = _get_aruco_dictionary(cfg.dictionary_name)
        self.board = _make_grid_board(cfg, self.dictionary)

        # in ArucoProcessor.__init__ after width/height known:
        self.K, self.dist = _load_intrinsics("intrinsics.json")
        print("Loaded JSON intrinsics:")
        print("K:\n", self.K)
        print("dist:\n", self.dist)



        # detection params
        try:
            self.detector_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
            self._use_new_api = True
        except Exception:
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self._use_new_api = False
            self.detector = None

        self.running = False
        self.thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self.latest_rgb_jpeg_b64: Optional[str] = None
        self.latest_rgb_ts: Optional[float] = None

        self.latest_aruco_jpeg_b64: Optional[str] = None
        self.latest_aruco_ts: Optional[float] = None

        self.latest_pose: Optional[Dict[str, Any]] = None  # pose + ids + corners count
        self.processed_frames = 0
        self.detected_frames = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def get_rgb(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "rgb": None if self.latest_rgb_jpeg_b64 is None else f"data:image/jpeg;base64,{self.latest_rgb_jpeg_b64}",
                "timestamp": self.latest_rgb_ts,
            }

    def get_aruco(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "rgb": None if self.latest_aruco_jpeg_b64 is None else f"data:image/jpeg;base64,{self.latest_aruco_jpeg_b64}",
                "timestamp": self.latest_aruco_ts,
                "pose": self.latest_pose,
            }

    def _encode_jpeg_b64(self, bgr: np.ndarray, quality: int = 85) -> str:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64.b64encode(buf).decode("utf-8")

    def _detect(self, frame_bgr: np.ndarray):
        if self._use_new_api and self.detector is not None:
            corners, ids, _rejected = self.detector.detectMarkers(frame_bgr)
        else:
            corners, ids, _rejected = cv2.aruco.detectMarkers(
                frame_bgr, self.dictionary, parameters=self.detector_params
            )
        return corners, ids

    def _estimate_board_pose(self, corners, ids, frame_bgr=None):
        """
        Try cv2.aruco.estimatePoseBoard if available, otherwise fall back to solvePnP
        with board corner correspondences. If that is unavailable, fall back to
        per-marker pose (first detected marker).
        """
        # Normalize ids to 1D ints for matching
        ids = ids.flatten().astype(int)
        # Preferred API
        if hasattr(cv2.aruco, "estimatePoseBoard"):
            try:
                return cv2.aruco.estimatePoseBoard(
                    corners, ids, self.board, self.K, self.dist, None, None
                )
            except Exception:
                pass

        # Fallback: single-marker pose (first marker)
        try:
            if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.cfg.marker_size_m, self.K, self.dist
                )
                if rvecs is not None and len(rvecs) > 0:
                    return True, rvecs[0].reshape(3, 1), tvecs[0].reshape(3, 1)
        except Exception:
            pass

        # Fallback: manual single-marker solvePnP using the first marker
        try:
            s = float(self.cfg.marker_size_m)
            obj_pts = np.array([
                [0.0, 0.0, 0.0],
                [s, 0.0, 0.0],
                [s, s, 0.0],
                [0.0, s, 0.0],
            ], dtype=np.float32)

            img_pts = np.asarray(corners[0], dtype=np.float32).reshape(-1, 2)
            if img_pts.shape[0] >= 4:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    self.K,
                    self.dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
                )
                if ok:
                    return ok, rvec, tvec
        except Exception:
            pass

        # Fallback: manual solvePnP using board.objPoints mapping to detected ids
        try:
            obj_pts = []
            img_pts = []
            board_ids = np.array(self.board.ids).flatten() if hasattr(self.board, "ids") else np.array([])
            for corner, marker_id in zip(corners, ids.flatten()):
                if board_ids.size == 0:
                    continue
                matches = np.where(board_ids == marker_id)[0]
                if len(matches) == 0:
                    continue
                obj = np.asarray(self.board.objPoints[matches[0]], dtype=np.float32).reshape(-1, 3)
                img = np.asarray(corner, dtype=np.float32).reshape(-1, 2)
                obj_pts.append(obj)
                img_pts.append(img)

            if len(obj_pts) >= 1:
                obj_pts = np.vstack(obj_pts)
                img_pts = np.vstack(img_pts)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    self.K,
                    self.dist,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if ok:
                    return ok, rvec, tvec
        except Exception:
            pass

        return False, None, None

    def _loop(self):
        logger.info("Aruco processor thread started (fps=%.2f)", self.process_fps)
        period = 1.0 / max(0.1, self.process_fps)

        while self.running:
            t0 = time.time()

            frame, ts = self.capture.get_latest()
            if frame is None or ts is None:
                time.sleep(0.01)
                continue

            # Always publish latest RGB feed (cheap)
            try:
                rgb_b64 = self._encode_jpeg_b64(frame, quality=85)
            except Exception as e:
                logger.warning("RGB encode failed: %s", e)
                rgb_b64 = None

            # Run ArUco detection
            annotated = frame.copy()
            pose_payload = {
                "detected": False,
                "marker_ids": [],
                "board_pose_camera_T_4x4": None,
                "rvec": None,
                "tvec": None,
                "quaternion_xyzw": None,
                "num_markers": 0,
                "K": self.K.tolist(),
            }

            try:
                corners, ids = self._detect(frame)
                if ids is not None and len(ids) > 0:
                    pose_payload["marker_ids"] = ids.flatten().astype(int).tolist()
                    pose_payload["num_markers"] = int(len(ids))

                    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

                    # Board pose estimate (new API first, then fallback)
                    retval, rvec, tvec = self._estimate_board_pose(corners, ids, frame_bgr=annotated)

                    if rvec is not None and tvec is not None:
                        T = _rvec_tvec_to_T(rvec, tvec)
                        quat = _rotmat_to_quat_xyzw(T[:3, :3])

                        pose_payload["detected"] = True
                        pose_payload["board_pose_camera_T_4x4"] = T.tolist()
                        pose_payload["rvec"] = rvec.reshape(3).astype(float).tolist()
                        pose_payload["tvec"] = tvec.reshape(3).astype(float).tolist()
                        pose_payload["quaternion_xyzw"] = quat.astype(float).tolist()

                        # Only draw marker highlights (axes removed to keep RGB clean)

                        self.detected_frames += 1

            except Exception as e:
                logger.warning("Aruco detection failed: %s", e)

            # Encode annotated frame
            try:
                aruco_b64 = self._encode_jpeg_b64(annotated, quality=85)
            except Exception as e:
                logger.warning("Aruco encode failed: %s", e)
                aruco_b64 = None

            with self._lock:
                self.latest_rgb_jpeg_b64 = rgb_b64
                self.latest_rgb_ts = ts

                self.latest_aruco_jpeg_b64 = aruco_b64
                self.latest_aruco_ts = ts

                self.latest_pose = pose_payload
                self.processed_frames += 1

            # sleep to maintain processing FPS
            elapsed = time.time() - t0
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        logger.info("Aruco processor thread stopped")


# -----------------------------
# Flask API
# -----------------------------
def create_app(capture: UxPlayCapture, processor: ArucoProcessor) -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "uxplay_running": capture.running,
            "frames_received": capture.frames_received,
            "processed_frames": processor.processed_frames,
            "detected_frames": processor.detected_frames,
            "resolution": f"{capture.width}x{capture.height}",
            "process_fps": processor.process_fps,
        }), 200

    @app.route("/rgb", methods=["GET"])
    def rgb():
        """
        VisionOS: poll this for the background RGB feed.
        Returns: { rgb: data:image/jpeg;base64,..., timestamp: float }
        """
        out = processor.get_rgb()
        if out["rgb"] is None:
            return jsonify({"error": "No frame available yet"}), 503
        return jsonify(out), 200

    @app.route("/aruco", methods=["GET"])
    def aruco():
        """
        VisionOS: poll this for RGB + board pose for overlay.

        Returns:
          {
            rgb: data:image/jpeg;base64,...,
            timestamp: float,
            pose: {
              detected: bool,
              marker_ids: [..],
              board_pose_camera_T_4x4: [[..],[..],[..],[..]] or null,
              rvec: [..] or null,
              tvec: [..] or null,
              quaternion_xyzw: [..] or null,
              num_markers: int,
              K: [[..],[..],[..]]
            }
          }
        """
        out = processor.get_aruco()
        if out["rgb"] is None:
            return jsonify({"error": "No frame available yet"}), 503
        return jsonify(out), 200

    @app.route("/head_pose", methods=["POST"])
    def head_pose():
        """
        Accept head pose updates from visionOS.

        Expected JSON:
          position: [x, y, z]
          quaternion: [x, y, z, w]
          timestamp: float (optional)
        """
        data = request.get_json(force=True, silent=True) or {}
        position = data.get("position")
        quaternion = data.get("quaternion")
        timestamp = float(data.get("timestamp", time.time()))

        if not (isinstance(position, list) and len(position) == 3
                and isinstance(quaternion, list) and len(quaternion) == 4):
            return jsonify({"error": "Invalid payload"}), 400

        with head_pose_lock:
            global head_pose_latest, head_pose_meta
            head_pose_latest = {
                "position": [float(x) for x in position],
                "quaternion": [float(x) for x in quaternion],
                "timestamp": timestamp
            }
            now = time.time()
            head_pose_meta["receive_time"] = now
            head_pose_meta["reception_count"] += 1
            if head_pose_meta["last_reception_time"] is not None:
                delta = now - head_pose_meta["last_reception_time"]
                if delta > 0:
                    head_pose_meta["reception_rate"] = 1.0 / delta
            head_pose_meta["last_reception_time"] = now

        return jsonify({"status": "ok"}), 200

    @app.route("/get_head_pose", methods=["GET"])
    def get_head_pose():
        """
        Return the most recent head pose (if any) along with age and reception stats.
        """
        with head_pose_lock:
            if head_pose_latest is None:
                return jsonify({"error": "No head pose received yet"}), 404
            latest = dict(head_pose_latest)
            meta = dict(head_pose_meta)

        age = None
        if meta.get("receive_time") is not None:
            age = time.time() - meta["receive_time"]

        latest.update({
            "age_seconds": age,
            "reception_count": meta.get("reception_count", 0),
            "reception_rate": meta.get("reception_rate"),
        })
        return jsonify(latest), 200

    @app.route("/mjpeg", methods=["GET"])
    def mjpeg():
        """
        MJPEG stream with detection highlights; includes pose axes/text overlay if available.
        """
        K = processor.K.copy()
        dist = processor.dist.copy()

        def draw_axes(frame, rvec, tvec, length=0.12):
            try:
                axes = np.float32([
                    [0, 0, 0],
                    [length, 0, 0],
                    [0, length, 0],
                    [0, 0, length],
                ])
                img_pts, _ = cv2.projectPoints(axes, rvec, tvec, K, dist)
                pts = img_pts.reshape(-1, 2).astype(int)
                origin = tuple(pts[0])
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
                for i in range(1, 4):
                    cv2.line(frame, origin, tuple(pts[i]), colors[i-1], 2, cv2.LINE_AA)
            except Exception:
                pass
            return frame

        def gen():
            while True:
                out = processor.get_aruco()
                if out["rgb"] is None:
                    time.sleep(0.05)
                    continue

                frame_b64 = out["rgb"].split(",", 1)[1]
                frame = cv2.imdecode(np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8), cv2.IMREAD_COLOR)

                pose = out.get("pose") or {}
                if pose.get("detected") and pose.get("rvec") and pose.get("tvec"):
                    rvec = np.array(pose["rvec"], dtype=np.float32).reshape(3, 1)
                    tvec = np.array(pose["tvec"], dtype=np.float32).reshape(3, 1)
                    frame = draw_axes(frame, rvec, tvec, length=processor.cfg.axis_length_m)
                    text = f"Pose (m): x={tvec[0][0]:.3f} y={tvec[1][0]:.3f} z={tvec[2][0]:.3f}"
                    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    time.sleep(0.05)
                    continue
                jpg = buf.tobytes()

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                       jpg + b"\r\n")
                time.sleep(0.03)  # stream throttle

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


# -----------------------------
# Utilities
# -----------------------------
def find_uxplay_binary(user_path: Optional[str]) -> str:
    if user_path and os.path.exists(user_path):
        return user_path

    candidates = [
        "/usr/local/bin/uxplay",
        "/usr/bin/uxplay",
        "/opt/homebrew/bin/uxplay",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # fallback: which
    try:
        r = subprocess.run(["which", "uxplay"], capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            p = r.stdout.strip()
            if p:
                return p
    except Exception:
        pass

    raise FileNotFoundError("uxplay binary not found. Install UxPlay or pass --uxplay-binary.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Basic Main API with UxPlay (no forwarding)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--uxplay-binary", default=None)
    parser.add_argument("--device-name", default="AirPlay-Pipeline")

    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)

    parser.add_argument("--fps", type=float, default=15.0, help="AruCo processing FPS (not capture FPS)")
    parser.add_argument("--aruco-dict", default="DICT_4X4_250")
    parser.add_argument("--aruco-rows", type=int, default=3)
    parser.add_argument("--aruco-cols", type=int, default=4)
    parser.add_argument("--marker-size-m", type=float, default=0.03)
    parser.add_argument("--separation-m", type=float, default=0.01)
    parser.add_argument("--no-axes", action="store_true")

    args = parser.parse_args()

    uxplay_bin = find_uxplay_binary(args.uxplay_binary)

    ar_cfg = ArucoConfig(
        dictionary_name=args.aruco_dict,
        rows=args.aruco_rows,
        cols=args.aruco_cols,
        marker_size_m=args.marker_size_m,
        separation_m=args.separation_m,
        draw_axes=not args.no_axes,
    )

    capture = UxPlayCapture(
        uxplay_binary=uxplay_bin,
        device_name=args.device_name,
        width=args.width,
        height=args.height,
    )
    capture.start()

    processor = ArucoProcessor(
        capture=capture,
        cfg=ar_cfg,
        process_fps=args.fps,
    )
    processor.start()

    app = create_app(capture, processor)

    def shutdown(*_):
        logger.info("Shutting down...")
        processor.stop()
        capture.stop()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("API running on http://%s:%d", args.host, args.port)
    logger.info("Endpoints: /health  /rgb  /aruco  (optional /mjpeg)")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
