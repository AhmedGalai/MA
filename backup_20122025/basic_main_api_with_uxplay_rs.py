#!/usr/bin/env python3
"""
basic_main_api_with_uxplay_rs.py

Single-process Flask API that:
- runs UxPlay (AirPlay receiver)
- captures raw BGR frames from UxPlay stdout (no network forwarding)
- processes frames at a configurable FPS (AruCo detection + board pose)
- serves VisionOS clients and debug UI:
  - RGB feed endpoint (base64 JPEG)
  - Pose endpoint (base64 JPEG + ArUco board pose as 4x4 matrix + rvec/tvec + ids)

Debug additions:
- /mjpeg?view=raw|overlay|mask
- /debug HTML page showing AVP + RS views + HSV controls
- /hsv_config GET/POST to control HSV mean/stddev threshold mask

Notes:
- UxPlay raw video output requires you to know width/height ahead of time.
- Pose estimation needs camera intrinsics K/dist from intrinsics.json
"""

import os
import time
import signal
import base64
import logging
import threading
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

from rich import print, pretty
pretty.install()

from config import CONFIG
from realsense_client import RealSenseClient
from aruco_detector import ArucoDetector
from aruco_calibration import load_calibration
from coordinate_manager import CoordinateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("basic_main_api_with_uxplay")

# -----------------------------
# Model selection (VisionOS UI)
# -----------------------------
selected_model = None

# -----------------------------
# Head pose storage (from visionOS)
# -----------------------------
head_pose_latest = None
head_pose_meta = {
    "receive_time": None,
    "reception_count": 0,
    "last_reception_time": None,
    "reception_rate": None,
}
head_pose_lock = threading.Lock()

rs_capture = None
rs_lock = threading.Lock()
rs_latest = {
    "rgb": None,
    "depth": None,
    "K": None,
    "timestamp": None,
}

rs_aruco_lock = threading.Lock()
rs_aruco_latest = {
    "overlay": None,
    "pose_matrix": None,
    "rvec": None,
    "tvec": None,
    "marker_ids": [],
    "markers_detected": 0,
    "timestamp": None,
}

avp_pose_lock = threading.Lock()
avp_pose_latest = {
    "pose_matrix": None,
    "rvec": None,
    "tvec": None,
    "timestamp": None,
}

coord_lock = threading.Lock()
coordinate_manager: Optional[CoordinateManager] = None

# -----------------------------
# ArUco config (edit as needed)
# -----------------------------
@dataclass
class ArucoConfig:
    dictionary_name: str = "DICT_4X4_50"
    rows: int = 3
    cols: int = 4
    marker_size_m: float = 0.03
    separation_m: float = 0.01
    draw_axes: bool = True
    axis_length_m: float = 0.06


# -----------------------------
# HSV filter config (debug mask)
# -----------------------------
@dataclass
class HsvFilterConfig:
    mean_h: int = 90   # OpenCV H: 0..179 (cyan)
    mean_s: int = 255  # 0..255
    mean_v: int = 255  # 0..255
    std_h: int = 10
    std_s: int = 40
    std_v: int = 40
    enabled: bool = True


hsv_cfg = HsvFilterConfig()
hsv_lock = threading.Lock()


# -----------------------------
# UxPlay capture
# -----------------------------
class UxPlayCapture:
    """
    Spawns UxPlay and reads raw BGR frames from stdout.
    Important: You MUST supply width/height correctly.
    """

    def __init__(self, uxplay_binary: str, device_name: str, width: int, height: int):
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
# RealSense capture (RGB-D)
# -----------------------------
class RealSenseCapture:
    def __init__(self, width: int, height: int, fps: int):
        self.client = RealSenseClient(width=width, height=height, fps=fps)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.latest: Dict[str, Any] = {
            "rgb": None,
            "depth": None,
            "K": None,
            "timestamp": None,
        }

    def start(self) -> None:
        if self.running:
            return
        ok = self.client.start()
        if not ok:
            logger.warning("RealSense start failed")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.client.stop()

    def get_latest(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.latest)

    def _loop(self) -> None:
        while self.running:
            frame = self.client.capture()
            if frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self.latest = {
                    "rgb": frame.get("rgb"),
                    "depth": frame.get("depth"),
                    "K": frame.get("K"),
                    "timestamp": frame.get("timestamp", time.time()),
                }


# -----------------------------
# ArUco processing helpers
# -----------------------------
def _get_aruco_dictionary(name: str):
    name = name.strip()
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _make_grid_board(cfg: ArucoConfig, dictionary):
    try:
        return cv2.aruco.GridBoard_create(
            cfg.cols, cfg.rows, cfg.marker_size_m, cfg.separation_m, dictionary
        )
    except Exception:
        return cv2.aruco.GridBoard(
            (cfg.cols, cfg.rows), cfg.marker_size_m, cfg.separation_m, dictionary
        )


def _load_intrinsics(path: str):
    import json
    with open(path, "r") as f:
        d = json.load(f)
    K = np.array(d["K"], dtype=np.float32)
    dist = np.array(d.get("dist", [0, 0, 0, 0, 0]), dtype=np.float32).reshape(-1, 1)
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


def _T_to_rvec_tvec(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = T[:3, :3]
    tvec = T[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R.astype(np.float32))
    return rvec, tvec


def _draw_axes(frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
               K: np.ndarray, dist: np.ndarray, length: float,
               label: Optional[str] = None) -> np.ndarray:
    try:
        axes = np.float32(
            [
                [0, 0, 0],
                [length, 0, 0],
                [0, length, 0],
                [0, 0, length],
            ]
        )
        img_pts, _ = cv2.projectPoints(axes, rvec, tvec, K, dist)
        pts = img_pts.reshape(-1, 2).astype(int)
        origin = tuple(pts[0])
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
        for i in range(1, 4):
            cv2.line(frame, origin, tuple(pts[i]), colors[i - 1], 2, cv2.LINE_AA)
        if label:
            cv2.putText(frame, label, (origin[0] + 6, origin[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    except Exception:
        pass
    return frame


def _transform_depth_rs_to_avp(depth_rs: np.ndarray,
                               K_rs: np.ndarray,
                               T_avp_rs: np.ndarray,
                               K_avp: np.ndarray,
                               target_size: Tuple[int, int]) -> np.ndarray:
    h_rs, w_rs = depth_rs.shape
    h_avp, w_avp = target_size
    u, v = np.meshgrid(np.arange(w_rs), np.arange(h_rs))
    z = depth_rs.reshape(-1)
    valid = z > 0

    u = u.reshape(-1)[valid].astype(np.float32)
    v = v.reshape(-1)[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    fx_rs, fy_rs = K_rs[0, 0], K_rs[1, 1]
    cx_rs, cy_rs = K_rs[0, 2], K_rs[1, 2]
    X_rs = (u - cx_rs) * z / fx_rs
    Y_rs = (v - cy_rs) * z / fy_rs
    points_rs = np.vstack([X_rs, Y_rs, z, np.ones_like(z)])
    points_avp = T_avp_rs @ points_rs

    fx_avp, fy_avp = K_avp[0, 0], K_avp[1, 1]
    cx_avp, cy_avp = K_avp[0, 2], K_avp[1, 2]
    X_avp, Y_avp, Z_avp = points_avp[0], points_avp[1], points_avp[2]
    valid_avp = Z_avp > 0.01
    X_avp = X_avp[valid_avp]
    Y_avp = Y_avp[valid_avp]
    Z_avp = Z_avp[valid_avp]

    u_avp = (X_avp * fx_avp / Z_avp + cx_avp).astype(np.int32)
    v_avp = (Y_avp * fy_avp / Z_avp + cy_avp).astype(np.int32)

    valid_px = (u_avp >= 0) & (u_avp < w_avp) & (v_avp >= 0) & (v_avp < h_avp)
    u_avp = u_avp[valid_px]
    v_avp = v_avp[valid_px]
    Z_avp = Z_avp[valid_px]

    depth_avp = np.zeros((h_avp, w_avp), dtype=np.float32)
    depth_avp[v_avp, u_avp] = Z_avp
    return depth_avp


# -----------------------------
# HSV mask helpers
# -----------------------------
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _compute_hsv_mask(frame_bgr: np.ndarray, cfg: HsvFilterConfig) -> np.ndarray:
    """
    Binary mask from mean±std in HSV space.
    Handles hue wrap-around (H is circular).
    Returns mask uint8 (0 or 255).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mh, ms, mv = int(cfg.mean_h), int(cfg.mean_s), int(cfg.mean_v)
    sh, ss, sv = int(cfg.std_h), int(cfg.std_s), int(cfg.std_v)

    lo_s = _clamp(ms - ss, 0, 255)
    hi_s = _clamp(ms + ss, 0, 255)
    lo_v = _clamp(mv - sv, 0, 255)
    hi_v = _clamp(mv + sv, 0, 255)

    lo_h = mh - sh
    hi_h = mh + sh

    # Hue range is 0..179
    if 0 <= lo_h and hi_h <= 179:
        lower = np.array([lo_h, lo_s, lo_v], dtype=np.uint8)
        upper = np.array([hi_h, hi_s, hi_v], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    lo_h_wrapped = lo_h % 180
    hi_h_wrapped = hi_h % 180

    if lo_h < 0:
        # [0..hi_h] OR [lo_h_wrapped..179]
        lower1 = np.array([0, lo_s, lo_v], dtype=np.uint8)
        upper1 = np.array([_clamp(hi_h, 0, 179), hi_s, hi_v], dtype=np.uint8)
        lower2 = np.array([lo_h_wrapped, lo_s, lo_v], dtype=np.uint8)
        upper2 = np.array([179, hi_s, hi_v], dtype=np.uint8)
        return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    # hi_h > 179:
    # [lo_h..179] OR [0..hi_h_wrapped]
    lower1 = np.array([_clamp(lo_h, 0, 179), lo_s, lo_v], dtype=np.uint8)
    upper1 = np.array([179, hi_s, hi_v], dtype=np.uint8)
    lower2 = np.array([0, lo_s, lo_v], dtype=np.uint8)
    upper2 = np.array([hi_h_wrapped, hi_s, hi_v], dtype=np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))


def _encode_jpeg_b64(frame_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf).decode("utf-8")


# -----------------------------
# ArUco processing (rate-limited)
# -----------------------------
class ArucoProcessor:
    """
    Runs ArUco detection at a fixed FPS, using the latest captured frame.
    Stores:
      - latest raw frame (for /rgb)
      - latest annotated frame (for /aruco)
      - latest board pose payload (for overlay)
    """

    def __init__(self, capture: UxPlayCapture, cfg: ArucoConfig, process_fps: float):
        self.capture = capture
        self.cfg = cfg
        self.process_fps = float(process_fps)

        self.dictionary = _get_aruco_dictionary(cfg.dictionary_name)
        self.board = _make_grid_board(cfg, self.dictionary)

        self.K, self.dist = _load_intrinsics("intrinsics.json")
        print("Loaded JSON intrinsics:")
        print("K:\n", self.K)
        print("dist:\n", self.dist)

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

        self.latest_pose: Optional[Dict[str, Any]] = None
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

    def get_pose(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return None if self.latest_pose is None else dict(self.latest_pose)

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
        Try cv2.aruco.estimatePoseBoard if available, otherwise fall back to
        per-marker pose and solvePnP fallbacks.
        """
        ids = ids.flatten().astype(int)

        if hasattr(cv2.aruco, "estimatePoseBoard"):
            try:
                return cv2.aruco.estimatePoseBoard(
                    corners, ids, self.board, self.K, self.dist, None, None
                )
            except Exception:
                pass

        try:
            if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.cfg.marker_size_m, self.K, self.dist
                )
                if rvecs is not None and len(rvecs) > 0:
                    return True, rvecs[0].reshape(3, 1), tvecs[0].reshape(3, 1)
        except Exception:
            pass

        try:
            s = float(self.cfg.marker_size_m)
            obj_pts = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [s, 0.0, 0.0],
                    [s, s, 0.0],
                    [0.0, s, 0.0],
                ],
                dtype=np.float32,
            )

            img_pts = np.asarray(corners[0], dtype=np.float32).reshape(-1, 2)
            if img_pts.shape[0] >= 4:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    self.K,
                    self.dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                    if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
                    else cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    return ok, rvec, tvec
        except Exception:
            pass

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
                    flags=cv2.SOLVEPNP_ITERATIVE,
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

            try:
                rgb_b64 = self._encode_jpeg_b64(frame, quality=85)
            except Exception as e:
                logger.warning("RGB encode failed: %s", e)
                rgb_b64 = None

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

                    retval, rvec, tvec = self._estimate_board_pose(corners, ids, frame_bgr=annotated)
                    if rvec is not None and tvec is not None:
                        T = _rvec_tvec_to_T(rvec, tvec)
                        quat = _rotmat_to_quat_xyzw(T[:3, :3])

                        pose_payload["detected"] = True
                        pose_payload["board_pose_camera_T_4x4"] = T.tolist()
                        pose_payload["rvec"] = rvec.reshape(3).astype(float).tolist()
                        pose_payload["tvec"] = tvec.reshape(3).astype(float).tolist()
                        pose_payload["quaternion_xyzw"] = quat.astype(float).tolist()

                        self.detected_frames += 1
                        with avp_pose_lock:
                            avp_pose_latest["pose_matrix"] = T.copy()
                            avp_pose_latest["rvec"] = rvec.reshape(3).copy()
                            avp_pose_latest["tvec"] = tvec.reshape(3).copy()
                            avp_pose_latest["timestamp"] = ts
                        with coord_lock:
                            if coordinate_manager is not None:
                                try:
                                    coordinate_manager.set_avp_calibration(T)
                                except Exception:
                                    pass

            except Exception as e:
                logger.warning("Aruco detection failed: %s", e)

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

            elapsed = time.time() - t0
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        logger.info("Aruco processor thread stopped")


def _start_rs_aruco_worker() -> threading.Thread:
    detector = ArucoDetector()
    rs_cfg = CONFIG["aruco"]

    def loop():
        while True:
            if rs_capture is None:
                time.sleep(0.05)
                continue
            latest = rs_capture.get_latest()
            rgb = latest.get("rgb")
            K = latest.get("K")
            ts = latest.get("timestamp") or time.time()
            if rgb is None or K is None:
                time.sleep(0.02)
                continue

            overlay = rgb.copy()
            rvec = None
            tvec = None
            pose_matrix = None
            markers = 0
            marker_ids = []

            try:
                corners, ids = detector.detect_markers(overlay)
                if corners is not None and ids is not None:
                    markers = len(ids)
                    marker_ids = ids.flatten().tolist()
                    cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
                    pose = detector.estimate_board_pose(corners, ids, K, np.zeros((5, 1), dtype=np.float32))
                    if pose is not None:
                        rvec = pose[0].reshape(3, 1)
                        tvec = pose[1].reshape(3, 1)
                        pose_matrix = ArucoDetector.pose_to_transformation_matrix(rvec, tvec)
                        overlay = _draw_axes(overlay, rvec, tvec, K, np.zeros((5, 1), dtype=np.float32),
                                             length=rs_cfg["marker_size_m"] * 2.0,
                                             label="RS Aruco")
                cv2.putText(overlay, "RS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
            except Exception as e:
                logger.warning("RS ArUco detection failed: %s", e)

            with rs_aruco_lock:
                rs_aruco_latest.update(
                    {
                        "overlay": overlay,
                        "pose_matrix": pose_matrix,
                        "rvec": None if rvec is None else rvec.reshape(3),
                        "tvec": None if tvec is None else tvec.reshape(3),
                        "marker_ids": marker_ids,
                        "markers_detected": markers,
                        "timestamp": ts,
                    }
                )

            time.sleep(0.1)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def _get_T_avp_rs() -> Optional[np.ndarray]:
    with coord_lock:
        if coordinate_manager is not None and coordinate_manager.is_calibrated():
            try:
                return coordinate_manager.get_T_avp_rs()
            except Exception:
                pass

    with avp_pose_lock:
        T_world_avp = avp_pose_latest.get("pose_matrix")
    with rs_aruco_lock:
        T_world_rs = rs_aruco_latest.get("pose_matrix")

    if T_world_avp is None or T_world_rs is None:
        return None

    try:
        return np.linalg.inv(T_world_avp) @ T_world_rs
    except Exception:
        return None


# -----------------------------
# Flask API
# -----------------------------
def create_app(capture: UxPlayCapture, processor: ArucoProcessor) -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        with hsv_lock:
            hsv_state = vars(hsv_cfg).copy()

        with coord_lock:
            calibrated = coordinate_manager is not None and coordinate_manager.is_calibrated()

        rs_connected = rs_capture is not None and rs_capture.running

        return jsonify(
            {
                "status": "ok",
                "uxplay_running": capture.running,
                "rs_connected": rs_connected,
                "calibrated": calibrated,
                "frames_received": capture.frames_received,
                "processed_frames": processor.processed_frames,
                "detected_frames": processor.detected_frames,
                "resolution": f"{capture.width}x{capture.height}",
                "process_fps": processor.process_fps,
                "hsv_filter": hsv_state,
            }
        ), 200

    @app.route("/models", methods=["GET"])
    def models():
        """
        List available 3D models.
        """
        try:
            models_dir = Path(CONFIG["paths"]["models"])
            if not models_dir.exists():
                return jsonify({"models": []}), 200
            model_files = sorted([f.name for f in models_dir.glob("*.ply")])
            return jsonify({"models": model_files}), 200
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/select_model", methods=["POST"])
    def select_model():
        """
        Set the active model (compat endpoint for VisionOS client).
        """
        global selected_model
        try:
            data = request.get_json() or {}
            name = data.get("model_name") or data.get("name")
            if not name:
                return jsonify({"error": "model_name is required"}), 400
            selected_model = str(name)
            logger.info(f"Selected model set to {selected_model}")
            return jsonify({"success": True, "model_name": selected_model}), 200
        except Exception as e:
            logger.error(f"Error selecting model: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/rgb", methods=["GET"])
    def rgb():
        out = processor.get_rgb()
        if out["rgb"] is None:
            return jsonify({"error": "No frame available yet"}), 503
        return jsonify(out), 200

    @app.route("/aruco", methods=["GET"])
    def aruco():
        out = processor.get_aruco()
        if out["rgb"] is None:
            return jsonify({"error": "No frame available yet"}), 503
        return jsonify(out), 200

    @app.route("/rgb_frame", methods=["GET"])
    def rgb_frame():
        frame, ts = capture.get_latest()
        if frame is None:
            return jsonify({"error": "No frame available yet"}), 503
        rgb_b64 = _encode_jpeg_b64(frame, quality=85)
        return jsonify({"frame": f"data:image/jpeg;base64,{rgb_b64}", "timestamp": ts}), 200

    @app.route("/intrinsics", methods=["GET"])
    def intrinsics():
        return jsonify({"K": processor.K.tolist()}), 200

    @app.route("/get_rgbd_frame", methods=["GET"])
    def get_rgbd_frame():
        if rs_capture is None:
            return jsonify({"error": "RealSense not connected"}), 503
        latest = rs_capture.get_latest()
        rgb = latest.get("rgb")
        depth = latest.get("depth")
        ts = latest.get("timestamp")
        if rgb is None or depth is None:
            return jsonify({"error": "No RealSense frame available yet"}), 503

        rgb_b64 = _encode_jpeg_b64(rgb, quality=85)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        depth_b64 = _encode_jpeg_b64(depth_colormap, quality=85)
        return jsonify(
            {
                "rgb": f"data:image/jpeg;base64,{rgb_b64}",
                "depth": f"data:image/jpeg;base64,{depth_b64}",
                "timestamp": ts,
            }
        ), 200

    @app.route("/get_rs_aruco_frame", methods=["GET"])
    def get_rs_aruco_frame():
        with rs_aruco_lock:
            overlay = rs_aruco_latest.get("overlay")
            markers = rs_aruco_latest.get("markers_detected", 0)
            marker_ids = rs_aruco_latest.get("marker_ids", [])
            ts = rs_aruco_latest.get("timestamp")
            pose_matrix = rs_aruco_latest.get("pose_matrix")

        if overlay is None:
            return jsonify({"error": "No RS ArUco frame yet"}), 503
        rgb_b64 = _encode_jpeg_b64(overlay, quality=85)
        return jsonify(
            {
                "rgb": f"data:image/jpeg;base64,{rgb_b64}",
                "markers_detected": markers,
                "marker_ids": marker_ids,
                "timestamp": ts,
                "pose_matrix": None if pose_matrix is None else pose_matrix.tolist(),
            }
        ), 200

    @app.route("/get_aruco_frame", methods=["GET"])
    def get_aruco_frame_alias():
        return get_rs_aruco_frame()

    @app.route("/get_avp_latest_frame", methods=["GET"])
    def get_avp_latest_frame():
        frame, ts = capture.get_latest()
        if frame is None:
            return jsonify({"error": "No AVP frame yet"}), 503
        rgb_b64 = _encode_jpeg_b64(frame, quality=85)
        age = None if ts is None else max(0.0, time.time() - ts)
        return jsonify(
            {
                "rgb": f"data:image/jpeg;base64,{rgb_b64}",
                "timestamp": ts,
                "age_seconds": age,
                "width": capture.width,
                "height": capture.height,
            }
        ), 200

    @app.route("/get_avp_aruco_frame", methods=["GET"])
    def get_avp_aruco_frame():
        out = processor.get_aruco()
        if out["rgb"] is None:
            return jsonify({"error": "No AVP ArUco frame yet"}), 503
        pose = out.get("pose") or {}
        return jsonify(
            {
                "rgb": out["rgb"],
                "timestamp": out["timestamp"],
                "markers_detected": 0 if not pose.get("detected") else pose.get("num_markers", 0),
                "marker_ids": pose.get("marker_ids", []),
                "pose_matrix": pose.get("board_pose_camera_T_4x4"),
            }
        ), 200

    @app.route("/get_avp_rs_overlay", methods=["GET"])
    def get_avp_rs_overlay():
        frame, ts = capture.get_latest()
        if frame is None:
            return jsonify({"error": "No AVP frame yet"}), 503
        out = frame.copy()
        K_avp = processor.K
        dist = processor.dist

        with avp_pose_lock:
            avp_pose = avp_pose_latest.get("pose_matrix")

        if avp_pose is not None:
            rvec, tvec = _T_to_rvec_tvec(avp_pose)
            out = _draw_axes(out, rvec, tvec, K_avp, dist, processor.cfg.axis_length_m, label="Aruco")

        T_avp_rs = _get_T_avp_rs()
        if T_avp_rs is not None:
            rvec, tvec = _T_to_rvec_tvec(T_avp_rs)
            out = _draw_axes(out, rvec, tvec, K_avp, dist, processor.cfg.axis_length_m, label="RS")

        rgb_b64 = _encode_jpeg_b64(out, quality=85)
        return jsonify(
            {
                "rgb": f"data:image/jpeg;base64,{rgb_b64}",
                "timestamp": ts,
            }
        ), 200

    @app.route("/get_avp_mask_frame", methods=["GET"])
    def get_avp_mask_frame():
        frame, ts = capture.get_latest()
        if frame is None:
            return jsonify({"error": "No AVP frame yet"}), 503
        with hsv_lock:
            cfg = HsvFilterConfig(**vars(hsv_cfg))
        if cfg.enabled:
            mask = _compute_hsv_mask(frame, cfg)
            out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            out = np.zeros_like(frame)
        rgb_b64 = _encode_jpeg_b64(out, quality=85)
        return jsonify(
            {
                "rgb": f"data:image/jpeg;base64,{rgb_b64}",
                "timestamp": ts,
            }
        ), 200

    @app.route("/get_transformed_depth", methods=["GET"])
    def get_transformed_depth():
        if rs_capture is None:
            return jsonify({"error": "RealSense not connected"}), 503
        latest = rs_capture.get_latest()
        depth = latest.get("depth")
        K_rs = latest.get("K")
        if depth is None or K_rs is None:
            return jsonify({"error": "No RealSense depth yet"}), 503
        T_avp_rs = _get_T_avp_rs()
        if T_avp_rs is None:
            return jsonify({"error": "Missing calibration for RS->AVP transform"}), 503

        K_avp = processor.K
        depth_avp = _transform_depth_rs_to_avp(depth, K_rs, T_avp_rs, K_avp, (capture.height, capture.width))
        depth_norm = cv2.normalize(depth_avp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        depth_b64 = _encode_jpeg_b64(depth_colormap, quality=85)
        min_depth = float(np.min(depth_avp[depth_avp > 0])) if np.any(depth_avp > 0) else 0.0
        max_depth = float(np.max(depth_avp)) if depth_avp.size else 0.0
        return jsonify(
            {
                "depth_colormap": f"data:image/jpeg;base64,{depth_b64}",
                "timestamp": latest.get("timestamp", time.time()),
                "transformation_applied": True,
                "min_depth": min_depth,
                "max_depth": max_depth,
            }
        ), 200

    @app.route("/get_intrinsics", methods=["GET"])
    def get_intrinsics():
        rs_K = None
        if rs_capture is not None:
            latest = rs_capture.get_latest()
            K = latest.get("K")
            if K is not None:
                rs_K = K.tolist()
        return jsonify(
            {
                "rs": {
                    "K": rs_K,
                    "calculated": rs_K is not None,
                    "method": "realsense",
                    "timestamp": time.time(),
                },
                "avp": {
                    "K": processor.K.tolist(),
                    "calculated": True,
                    "method": "intrinsics.json",
                    "timestamp": time.time(),
                },
            }
        ), 200

    @app.route("/get_transformation", methods=["GET"])
    def get_transformation():
        T_avp_rs = _get_T_avp_rs()
        with avp_pose_lock:
            T_world_avp = avp_pose_latest.get("pose_matrix")
        with rs_aruco_lock:
            T_world_rs = rs_aruco_latest.get("pose_matrix")

        if coordinate_manager is not None and coordinate_manager.is_calibrated():
            try:
                T_world_rs = coordinate_manager.get_T_world_rs()
                T_world_avp = coordinate_manager.T_world_avp
            except Exception:
                pass

        calibrated = T_avp_rs is not None
        return jsonify(
            {
                "T_avp_rs": None if T_avp_rs is None else T_avp_rs.tolist(),
                "T_world_rs": None if T_world_rs is None else T_world_rs.tolist(),
                "T_world_avp": None if T_world_avp is None else T_world_avp.tolist(),
                "calibrated": calibrated,
                "message": "ok" if calibrated else "Missing calibration",
            }
        ), 200

    @app.route("/get_rs_pose_in_avp", methods=["GET"])
    def get_rs_pose_in_avp():
        T_avp_rs = _get_T_avp_rs()
        with head_pose_lock:
            age = None
            if head_pose_meta.get("receive_time") is not None:
                age = time.time() - head_pose_meta["receive_time"]

        if T_avp_rs is None:
            return jsonify({"calibrated": False, "message": "Missing calibration"}), 200

        position = T_avp_rs[:3, 3].tolist()
        quat = _rotmat_to_quat_xyzw(T_avp_rs[:3, :3]).astype(float).tolist()
        return jsonify(
            {
                "calibrated": True,
                "position": position,
                "quaternion": quat,
                "T_avp_rs": T_avp_rs.tolist(),
                "head_pose_age": age,
            }
        ), 200

    @app.route("/head_pose", methods=["POST"])
    def head_pose():
        data = request.get_json(force=True, silent=True) or {}
        position = data.get("position")
        quaternion = data.get("quaternion")
        timestamp = float(data.get("timestamp", time.time()))

        if not (isinstance(position, list) and len(position) == 3 and isinstance(quaternion, list) and len(quaternion) == 4):
            return jsonify({"error": "Invalid payload"}), 400

        with head_pose_lock:
            global head_pose_latest, head_pose_meta
            head_pose_latest = {
                "position": [float(x) for x in position],
                "quaternion": [float(x) for x in quaternion],
                "timestamp": timestamp,
            }
            now = time.time()
            head_pose_meta["receive_time"] = now
            head_pose_meta["reception_count"] += 1
            if head_pose_meta["last_reception_time"] is not None:
                delta = now - head_pose_meta["last_reception_time"]
                if delta > 0:
                    head_pose_meta["reception_rate"] = 1.0 / delta
            head_pose_meta["last_reception_time"] = now
        with coord_lock:
            if coordinate_manager is not None:
                try:
                    coordinate_manager.update_head_pose(position, quaternion, timestamp)
                except Exception:
                    pass

        return jsonify({"status": "ok"}), 200

    @app.route("/get_head_pose", methods=["GET"])
    def get_head_pose():
        with head_pose_lock:
            if head_pose_latest is None:
                return jsonify({"error": "No head pose received yet"}), 404
            latest = dict(head_pose_latest)
            meta = dict(head_pose_meta)

        age = None
        if meta.get("receive_time") is not None:
            age = time.time() - meta["receive_time"]

        latest.update(
            {
                "age_seconds": age,
                "reception_count": meta.get("reception_count", 0),
                "reception_rate": meta.get("reception_rate"),
            }
        )
        return jsonify(latest), 200

    @app.route("/hsv_config", methods=["GET", "POST"])
    def hsv_config_route():
        """
        GET: returns current HSV mean/std config
        POST: updates fields: mean_h, mean_s, mean_v, std_h, std_s, std_v, enabled
        """
        if request.method == "GET":
            with hsv_lock:
                return jsonify(
                    {
                        "mean_h": hsv_cfg.mean_h,
                        "mean_s": hsv_cfg.mean_s,
                        "mean_v": hsv_cfg.mean_v,
                        "std_h": hsv_cfg.std_h,
                        "std_s": hsv_cfg.std_s,
                        "std_v": hsv_cfg.std_v,
                        "enabled": hsv_cfg.enabled,
                    }
                ), 200

        data = request.get_json(force=True, silent=True) or {}
        with hsv_lock:
            for k in ["mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v"]:
                if k in data:
                    try:
                        v = int(float(data[k]))
                    except Exception:
                        continue
                    if k == "mean_h":
                        hsv_cfg.mean_h = _clamp(v, 0, 179)
                    elif k in ("mean_s", "mean_v"):
                        setattr(hsv_cfg, k, _clamp(v, 0, 255))
                    elif k == "std_h":
                        hsv_cfg.std_h = _clamp(v, 0, 90)
                    else:
                        setattr(hsv_cfg, k, _clamp(v, 0, 127))

            if "enabled" in data:
                hsv_cfg.enabled = bool(data["enabled"])

            return jsonify(
                {
                    "status": "ok",
                    "mean_h": hsv_cfg.mean_h,
                    "mean_s": hsv_cfg.mean_s,
                    "mean_v": hsv_cfg.mean_v,
                    "std_h": hsv_cfg.std_h,
                    "std_s": hsv_cfg.std_s,
                    "std_v": hsv_cfg.std_v,
                    "enabled": hsv_cfg.enabled,
                }
            ), 200

    @app.route("/debug", methods=["GET"])
    def debug_page():
        html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Debug Views</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 14px; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    .panel { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
    .hdr { padding: 8px 10px; background: #f7f7f7; border-bottom: 1px solid #eee; font-weight: 600; }
    img { width: 100%; display: block; background: #000; }
    .controls { margin-top: 12px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; }
    .row { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
    .row label { width: 120px; }
    input[type="range"] { width: 320px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
  </style>
</head>
<body>
  <h2 style="margin: 0 0 10px 0;">Debug Views</h2>

  <div class="grid">
    <div class="panel">
      <div class="hdr">AVP Raw (UxPlay)</div>
      <img id="raw" src="/mjpeg?view=raw" />
    </div>
    <div class="panel">
      <div class="hdr">AVP ArUco Overlay</div>
      <img id="overlay" src="/mjpeg?view=overlay" />
    </div>
    <div class="panel">
      <div class="hdr">AVP Mask (HSV)</div>
      <img id="mask" src="/mjpeg?view=mask" />
    </div>
    <div class="panel">
      <div class="hdr">RS RGB</div>
      <img id="rsrgb" src="/mjpeg?view=rs_rgb" />
    </div>
    <div class="panel">
      <div class="hdr">RS Depth</div>
      <img id="rsdepth" src="/mjpeg?view=rs_depth" />
    </div>
    <div class="panel">
      <div class="hdr">RS ArUco Overlay</div>
      <img id="rsaruco" src="/mjpeg?view=rs_aruco" />
    </div>
    <div class="panel">
      <div class="hdr">AVP + RS Pose Overlay</div>
      <img id="avprs" src="/mjpeg?view=avp_rs" />
    </div>
    <div class="panel">
      <div class="hdr">RS Depth → AVP</div>
      <img id="rsdepthavp" src="/mjpeg?view=avp_depth" />
    </div>
  </div>

  <div class="controls">
    <div class="row">
      <label>Enabled</label>
      <input id="enabled" type="checkbox" />
    </div>

    <div class="row">
      <label>Mean color</label>
      <input id="color" type="color" value="#00ffff" />
      <span class="mono" id="meanOut"></span>
    </div>

    <div class="row">
      <label>Std H</label>
      <input id="stdH" type="range" min="0" max="90" value="10" />
      <span class="mono" id="stdHOut"></span>
    </div>
    <div class="row">
      <label>Std S</label>
      <input id="stdS" type="range" min="0" max="127" value="40" />
      <span class="mono" id="stdSOut"></span>
    </div>
    <div class="row">
      <label>Std V</label>
      <input id="stdV" type="range" min="0" max="127" value="40" />
      <span class="mono" id="stdVOut"></span>
    </div>
  </div>

<script>
function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return m ? { r: parseInt(m[1],16), g: parseInt(m[2],16), b: parseInt(m[3],16) } : {r:0,g:0,b:0};
}

// Convert RGB [0..255] to HSV like OpenCV: H [0..179], S,V [0..255]
function rgbToOpenCvHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r,g,b), mn = Math.min(r,g,b);
  const d = mx - mn;

  let h = 0;
  if (d === 0) h = 0;
  else if (mx === r) h = ((g - b) / d) % 6;
  else if (mx === g) h = ((b - r) / d) + 2;
  else h = ((r - g) / d) + 4;

  h = Math.round((h * 60 + 360) % 360); // 0..360
  const s = mx === 0 ? 0 : d / mx;      // 0..1
  const v = mx;                         // 0..1

  const H = Math.round(h / 2);          // 0..179
  const S = Math.round(s * 255);
  const V = Math.round(v * 255);
  return {H, S, V};
}

async function postCfg(payload) {
  await fetch("/hsv_config", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
}

async function loadCfg() {
  const r = await fetch("/hsv_config");
  const j = await r.json();

  document.getElementById("enabled").checked = !!j.enabled;
  document.getElementById("stdH").value = j.std_h;
  document.getElementById("stdS").value = j.std_s;
  document.getElementById("stdV").value = j.std_v;

  document.getElementById("meanOut").textContent = `mean HSV: ${j.mean_h}, ${j.mean_s}, ${j.mean_v}`;
  document.getElementById("stdHOut").textContent = j.std_h;
  document.getElementById("stdSOut").textContent = j.std_s;
  document.getElementById("stdVOut").textContent = j.std_v;
}

function wire() {
  const enabled = document.getElementById("enabled");
  const color = document.getElementById("color");
  const stdH = document.getElementById("stdH");
  const stdS = document.getElementById("stdS");
  const stdV = document.getElementById("stdV");

  enabled.addEventListener("change", () => postCfg({enabled: enabled.checked}));

  color.addEventListener("input", () => {
    const {r,g,b} = hexToRgb(color.value);
    const hsv = rgbToOpenCvHsv(r,g,b);
    document.getElementById("meanOut").textContent = `mean HSV: ${hsv.H}, ${hsv.S}, ${hsv.V}`;
    postCfg({mean_h: hsv.H, mean_s: hsv.S, mean_v: hsv.V});
  });

  function sliderChanged() {
    document.getElementById("stdHOut").textContent = stdH.value;
    document.getElementById("stdSOut").textContent = stdS.value;
    document.getElementById("stdVOut").textContent = stdV.value;
    postCfg({std_h: stdH.value, std_s: stdS.value, std_v: stdV.value});
  }

  stdH.addEventListener("input", sliderChanged);
  stdS.addEventListener("input", sliderChanged);
  stdV.addEventListener("input", sliderChanged);
}

loadCfg().then(wire);
</script>
</body>
</html>
"""
        return Response(html, mimetype="text/html")

    @app.route("/mjpeg", methods=["GET"])
    def mjpeg():
        """
        MJPEG stream with multiple views:
          /mjpeg?view=raw       -> raw UxPlay feed
          /mjpeg?view=overlay   -> AVP ArUco overlay
          /mjpeg?view=mask      -> HSV mean±std binary mask view
          /mjpeg?view=rs_rgb    -> RealSense RGB
          /mjpeg?view=rs_depth  -> RealSense depth colormap
          /mjpeg?view=rs_aruco  -> RealSense ArUco overlay
          /mjpeg?view=avp_rs    -> AVP view with RS pose overlay
          /mjpeg?view=avp_depth -> RS depth transformed to AVP view
        """
        view = (request.args.get("view", "overlay") or "overlay").lower().strip()
        if view not in ("raw", "overlay", "mask", "rs_rgb", "rs_depth", "rs_aruco", "avp_rs", "avp_depth"):
            view = "overlay"

        K = processor.K.copy()
        dist = processor.dist.copy()

        def gen():
            while True:
                out = None

                if view in ("raw", "overlay", "mask", "avp_rs"):
                    frame_bgr, _ts = capture.get_latest()
                    if frame_bgr is None:
                        time.sleep(0.02)
                        continue
                    out = frame_bgr.copy()

                if view == "overlay":
                    pose = processor.get_pose() or {}
                    if pose.get("detected") and pose.get("rvec") and pose.get("tvec"):
                        rvec = np.array(pose["rvec"], dtype=np.float32).reshape(3, 1)
                        tvec = np.array(pose["tvec"], dtype=np.float32).reshape(3, 1)
                        out = _draw_axes(out, rvec, tvec, K, dist, processor.cfg.axis_length_m, label="Aruco")
                        txt = f"t (m): x={tvec[0][0]:.3f} y={tvec[1][0]:.3f} z={tvec[2][0]:.3f}"
                        cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(out, "Aruco: not detected", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                elif view == "mask":
                    with hsv_lock:
                        cfg = HsvFilterConfig(**vars(hsv_cfg))  # copy
                    if cfg.enabled:
                        mask = _compute_hsv_mask(out, cfg)
                        out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        hud = f"mean HSV=({cfg.mean_h},{cfg.mean_s},{cfg.mean_v}) std=({cfg.std_h},{cfg.std_s},{cfg.std_v})"
                        cv2.putText(out, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        out[:] = 0
                        cv2.putText(out, "mask disabled", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                elif view == "rs_rgb":
                    if rs_capture is None:
                        time.sleep(0.05)
                        continue
                    latest = rs_capture.get_latest()
                    out = latest.get("rgb")
                    if out is None:
                        time.sleep(0.02)
                        continue
                    out = out.copy()
                    cv2.putText(out, "RS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

                elif view == "rs_depth":
                    if rs_capture is None:
                        time.sleep(0.05)
                        continue
                    latest = rs_capture.get_latest()
                    depth = latest.get("depth")
                    if depth is None:
                        time.sleep(0.02)
                        continue
                    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    out = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                    cv2.putText(out, "RS Depth", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                elif view == "rs_aruco":
                    with rs_aruco_lock:
                        overlay = rs_aruco_latest.get("overlay")
                    if overlay is None:
                        time.sleep(0.02)
                        continue
                    out = overlay.copy()

                elif view == "avp_rs":
                    if out is None:
                        time.sleep(0.02)
                        continue
                    with avp_pose_lock:
                        avp_pose = avp_pose_latest.get("pose_matrix")
                    if avp_pose is not None:
                        rvec, tvec = _T_to_rvec_tvec(avp_pose)
                        out = _draw_axes(out, rvec, tvec, K, dist, processor.cfg.axis_length_m, label="Aruco")
                    T_avp_rs = _get_T_avp_rs()
                    if T_avp_rs is not None:
                        rvec, tvec = _T_to_rvec_tvec(T_avp_rs)
                        out = _draw_axes(out, rvec, tvec, K, dist, processor.cfg.axis_length_m, label="RS")

                elif view == "avp_depth":
                    if rs_capture is None:
                        time.sleep(0.05)
                        continue
                    latest = rs_capture.get_latest()
                    depth = latest.get("depth")
                    K_rs = latest.get("K")
                    if depth is None or K_rs is None:
                        time.sleep(0.02)
                        continue
                    T_avp_rs = _get_T_avp_rs()
                    if T_avp_rs is None:
                        time.sleep(0.05)
                        continue
                    depth_avp = _transform_depth_rs_to_avp(depth, K_rs, T_avp_rs, K, (capture.height, capture.width))
                    depth_norm = cv2.normalize(depth_avp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    out = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                    cv2.putText(out, "RS Depth -> AVP", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                if out is None:
                    time.sleep(0.02)
                    continue

                ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    time.sleep(0.02)
                    continue
                jpg = buf.tobytes()

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                )
                time.sleep(0.03)

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
    parser.add_argument("--rs-width", type=int, default=CONFIG["realsense"]["resolution_width"])
    parser.add_argument("--rs-height", type=int, default=CONFIG["realsense"]["resolution_height"])
    parser.add_argument("--rs-fps", type=int, default=CONFIG["realsense"]["fps"])

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

    global rs_capture, coordinate_manager
    rs_capture = RealSenseCapture(width=args.rs_width, height=args.rs_height, fps=args.rs_fps)
    rs_capture.start()
    _start_rs_aruco_worker()

    try:
        calib_path = CONFIG["paths"]["calibration_file"]
        T_world_rs = load_calibration(calib_path)
        if T_world_rs is None:
            logger.warning("No T_world_rs calibration found at %s", calib_path)
        else:
            coordinate_manager = CoordinateManager(T_world_rs=T_world_rs)
            logger.info("Loaded T_world_rs calibration")
    except Exception as e:
        logger.warning("Failed to initialize CoordinateManager: %s", e)

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
        if rs_capture is not None:
            rs_capture.stop()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("API running on http://%s:%d", args.host, args.port)
    logger.info("Endpoints: /health  /rgb  /aruco  /get_rgbd_frame  /get_rs_aruco_frame  /get_avp_latest_frame  /get_avp_aruco_frame  /get_avp_rs_overlay  /get_transformed_depth  /get_rs_pose_in_avp  /mjpeg  /debug  /hsv_config")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
