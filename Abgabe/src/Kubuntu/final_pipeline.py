#!/usr/bin/env python3
"""
basic_main_api_with_uxplay_rs.py (DROP-IN REPLACEMENT)

What changed vs your original:
- REMOVED HSV-based ROI mask for AVP (UxPlay). Replaced with a normalized circular ROI config.
  - New endpoint: /avp_roi_config  (GET/POST)
  - /get_avp_mask_frame now shows the ROI mask (not HSV).
  - /mjpeg?view=mask now shows the ROI mask (not HSV).

- FIXED RS->AVP depth reprojection:
  - Uses distortion properly:
    * RS: undistortPoints with RS K/dist from RealSense SDK color stream intrinsics
    * AVP: projectPoints with AVP K/dist from intrinsics.json
  - Correct z-buffer: keep nearest depth per AVP pixel (min Z)

- FIXED relative transform direction:
  - Assumes ArUco pose matrices are T_cam<-board (OpenCV convention) for BOTH AVP and RS
  - Computes: T_avp<-rs = T_avp<-board @ inv(T_rs<-board)

- ADDED "is depth aligned" check:
  - RealSenseClient already aligns depth to color; we verify shape match and report in /health

Everything else:
- Keeps your endpoints, debug UI, workers, and flow as close as possible.
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
from foundationpose_client import estimate_pose

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
    "dist": None,
    "timestamp": None,
    "aligned_ok": None,
}

rs_aruco_lock = threading.Lock()
rs_aruco_latest = {
    "overlay": None,
    "pose_matrix": None,   # expected T_rs<-board (OpenCV)
    "rvec": None,
    "tvec": None,
    "marker_ids": [],
    "markers_detected": 0,
    "timestamp": None,
}

avp_pose_lock = threading.Lock()
avp_pose_latest = {
    "pose_matrix": None,   # expected T_avp<-board (OpenCV)
    "rvec": None,
    "tvec": None,
    "timestamp": None,
}

coord_lock = threading.Lock()
coordinate_manager: Optional[CoordinateManager] = None

foundationpose_lock = threading.Lock()
foundationpose_latest = {
    "pose_matrix": None,
    "timestamp": None,
    "message": "FoundationPose not started",
    "success": False,
}

foundationpose_save_lock = threading.Lock()
foundationpose_save_pair = {
    "armed": False,
    "capture_id": None,
    "need_avp": False,
    "need_rs": False,
    "armed_time": None,
}

def _rs_depth_alignment_ok(latest: Dict[str, Any]) -> Tuple[bool, str]:
    rgb = latest.get("rgb")
    depth = latest.get("depth")
    if rgb is None or depth is None:
        return False, "missing rgb/depth"
    if depth.shape[:2] != rgb.shape[:2]:
        return False, f"shape mismatch depth={depth.shape} rgb={rgb.shape}"
    return True, "ok"


def _arm_save_pair() -> Dict[str, Any]:
    with foundationpose_save_lock:
        cid = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%100000:05d}"
        foundationpose_save_pair.update({
            "armed": True,
            "capture_id": cid,
            "need_avp": True,
            "need_rs": True,
            "armed_time": time.time(),
        })
        return dict(foundationpose_save_pair)

def _consume_save_avp() -> Optional[str]:
    with foundationpose_save_lock:
        if foundationpose_save_pair["armed"] and foundationpose_save_pair["need_avp"]:
            foundationpose_save_pair["need_avp"] = False
            cid = foundationpose_save_pair["capture_id"]
            if not foundationpose_save_pair["need_rs"]:
                foundationpose_save_pair["armed"] = False
            return cid
        return None

def _consume_save_rs() -> Optional[str]:
    with foundationpose_save_lock:
        if foundationpose_save_pair["armed"] and foundationpose_save_pair["need_rs"]:
            foundationpose_save_pair["need_rs"] = False
            cid = foundationpose_save_pair["capture_id"]
            if not foundationpose_save_pair["need_avp"]:
                foundationpose_save_pair["armed"] = False
            return cid
        return None

def _get_save_pair_state() -> Dict[str, Any]:
    with foundationpose_save_lock:
        return dict(foundationpose_save_pair)




processing_stride_lock = threading.Lock()
processing_stride = 1

rs_roi_lock = threading.Lock()
rs_roi_config = {
    "x_center": CONFIG["realsense"]["resolution_width"] // 2,
    "y_center": CONFIG["realsense"]["resolution_height"] // 2,
    "radius": 120,
}

foundationpose_rs_lock = threading.Lock()
foundationpose_rs_latest = {
    "pose_matrix": None,
    "timestamp": None,
    "message": "FoundationPose RS not started",
    "success": False,
}

# -----------------------------
# AVP normalized ROI (NEW, replaces HSV)
# -----------------------------
@dataclass
class AvpRoiConfig:
    enabled: bool = True
    # normalized center in [0..1]
    cx_n: float = 0.5
    cy_n: float = 0.5
    # normalized radius relative to min(width,height) in [0..1]
    r_n: float = 0.18

avp_roi_lock = threading.Lock()
avp_roi_cfg = AvpRoiConfig()


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
            "dist": None,
            "timestamp": None,
            "aligned_ok": None,
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

            rgb = frame.get("rgb")
            depth = frame.get("depth")
            aligned_ok = False
            if rgb is not None and depth is not None:
                aligned_ok = (rgb.shape[0] == depth.shape[0]) and (rgb.shape[1] == depth.shape[1])

            with self._lock:
                self.latest = {
                    "rgb": rgb,
                    "depth": depth,
                    "K": frame.get("K"),
                    "dist": frame.get("dist"),
                    "timestamp": frame.get("timestamp", time.time()),
                    "aligned_ok": aligned_ok,
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
    with open(path, "r") as f:
        d = json.load(f)
    K = np.array(d["K"], dtype=np.float32)
    dist = np.array(d.get("dist", [0, 0, 0, 0, 0]), dtype=np.float32).reshape(-1, 1)
    return K, dist


def _rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    # OpenCV pose is typically: object->camera (T_cam<-obj)
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = tvec.reshape(3).astype(np.float32)
    return T


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
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


# -----------------------------
# AVP ROI mask (NEW)
# -----------------------------
def _clamp_float(v: float, lo: float, hi: float) -> float:
    try:
        v = float(v)
    except Exception:
        v = lo
    return max(lo, min(hi, v))


def _compute_avp_roi_mask(h: int, w: int) -> np.ndarray:
    """
    Returns uint8 mask (0/255) for AVP image size, based on normalized ROI circle.
    Radius is relative to min(w,h).
    """
    with avp_roi_lock:
        cfg = AvpRoiConfig(**vars(avp_roi_cfg))

    if not cfg.enabled:
        return np.full((h, w), 255, dtype=np.uint8)

    cx = int(_clamp_float(cfg.cx_n, 0.0, 1.0) * (w - 1))
    cy = int(_clamp_float(cfg.cy_n, 0.0, 1.0) * (h - 1))
    r_pix = int(_clamp_float(cfg.r_n, 0.0, 1.0) * float(min(w, h)))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), max(1, r_pix), 255, -1)
    return mask


# -----------------------------
# Depth transform (FIXED + z-buffer + distortion aware)
# -----------------------------
def _transform_depth_rs_to_avp(depth_rs: np.ndarray,
                               K_rs: np.ndarray,
                               dist_rs: Optional[np.ndarray],
                               T_avp_rs: np.ndarray,
                               K_avp: np.ndarray,
                               dist_avp: Optional[np.ndarray],
                               target_size: Tuple[int, int]) -> np.ndarray:
    """
    Reproject RS aligned depth (aligned to RS color) into AVP image plane.

    - RS pixels are undistorted using RS K/dist (RealSense SDK intrinsics from color stream).
    - RS 3D points are transformed into AVP camera using T_avp<-rs.
    - AVP projection uses cv2.projectPoints with AVP K/dist.
    - Uses z-buffer (nearest Z wins) per AVP pixel.
    """
    h_rs, w_rs = depth_rs.shape
    h_avp, w_avp = target_size

    if K_rs is None or K_avp is None or T_avp_rs is None:
        return np.zeros((h_avp, w_avp), dtype=np.float32)

    # gather valid RS pixels
    z = depth_rs.reshape(-1).astype(np.float32)
    valid = z > 0.01
    if not np.any(valid):
        return np.zeros((h_avp, w_avp), dtype=np.float32)

    u = np.tile(np.arange(w_rs, dtype=np.float32), h_rs).reshape(-1)[valid]
    v = np.repeat(np.arange(h_rs, dtype=np.float32), w_rs).reshape(-1)[valid]
    z = z[valid]

    # undistort RS pixels -> normalized coordinates
    use_rs_dist = dist_rs is not None and np.any(np.abs(dist_rs) > 1e-9)
    pts = np.stack([u, v], axis=1).reshape(-1, 1, 2).astype(np.float32)
    if use_rs_dist:
        undist = cv2.undistortPoints(pts, K_rs, dist_rs)  # Nx1x2 normalized
    else:
        undist = cv2.undistortPoints(pts, K_rs, None)

    x = undist[:, 0, 0]
    y = undist[:, 0, 1]

    # 3D in RS camera
    X_rs = x * z
    Y_rs = y * z
    ones = np.ones_like(z, dtype=np.float32)
    pts_rs_h = np.stack([X_rs, Y_rs, z, ones], axis=0)  # 4xN

    # transform to AVP camera
    pts_avp_h = (T_avp_rs.astype(np.float32) @ pts_rs_h).astype(np.float32)
    pts_avp = pts_avp_h[:3, :].T  # Nx3

    # keep points in front of AVP cam
    valid2 = pts_avp[:, 2] > 0.01
    pts_avp = pts_avp[valid2]
    if pts_avp.size == 0:
        return np.zeros((h_avp, w_avp), dtype=np.float32)

    # project into AVP with distortion
    use_avp_dist = dist_avp is not None and np.any(np.abs(dist_avp) > 1e-9)
    dist_use = dist_avp if use_avp_dist else None

    img_pts, _ = cv2.projectPoints(
        pts_avp.astype(np.float32),
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        K_avp.astype(np.float32),
        None if dist_use is None else dist_use.astype(np.float32),
    )
    img_pts = img_pts.reshape(-1, 2)
    u_avp = img_pts[:, 0].astype(np.int32)
    v_avp = img_pts[:, 1].astype(np.int32)
    Z_avp = pts_avp[:, 2].astype(np.float32)

    inside = (u_avp >= 0) & (u_avp < w_avp) & (v_avp >= 0) & (v_avp < h_avp)
    if not np.any(inside):
        return np.zeros((h_avp, w_avp), dtype=np.float32)

    u_avp = u_avp[inside]
    v_avp = v_avp[inside]
    Z_avp = Z_avp[inside]

    # z-buffer
    depth_avp = np.full((h_avp, w_avp), np.inf, dtype=np.float32)
    # Use numpy "minimum.at" to do z-buffering efficiently
    np.minimum.at(depth_avp, (v_avp, u_avp), Z_avp)
    depth_avp[~np.isfinite(depth_avp)] = 0.0
    return depth_avp


def _encode_jpeg_b64(frame_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf).decode("utf-8")


def _decode_jpeg_b64(data_url: Optional[str]) -> Optional[np.ndarray]:
    if not data_url:
        return None
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        data = base64.b64decode(data_url)
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


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
        ids = ids.flatten().astype(int)

        if hasattr(cv2.aruco, "estimatePoseBoard"):
            try:
                # returns: (retval, rvec, tvec)
                return cv2.aruco.estimatePoseBoard(
                    corners, ids, self.board, self.K, self.dist, None, None
                )
            except Exception:
                pass

        # fallback: per-marker estimatePoseSingleMarkers
        try:
            if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.cfg.marker_size_m, self.K, self.dist
                )
                if rvecs is not None and len(rvecs) > 0:
                    return True, rvecs[0].reshape(3, 1), tvecs[0].reshape(3, 1)
        except Exception:
            pass

        return False, None, None

    def _loop(self):
        logger.info("Aruco processor thread started (fps=%.2f)", self.process_fps)
        period = 1.0 / max(0.1, self.process_fps)
        frame_index = 0

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

            frame_index += 1
            stride = _get_processing_stride()
            annotated = frame.copy()
            pose_payload = {
                "detected": False,
                "marker_ids": [],
                "board_pose_camera_T_4x4": None,  # T_avp<-board
                "rvec": None,
                "tvec": None,
                "quaternion_xyzw": None,
                "num_markers": 0,
                "K": self.K.tolist(),
                "dist": self.dist.reshape(-1).tolist(),
            }

            if stride > 1 and (frame_index % stride) != 0:
                with self._lock:
                    self.latest_rgb_jpeg_b64 = rgb_b64
                    self.latest_rgb_ts = ts
                elapsed = time.time() - t0
                sleep_s = period - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue

            try:
                corners, ids = self._detect(frame)
                if ids is not None and len(ids) > 0:
                    pose_payload["marker_ids"] = ids.flatten().astype(int).tolist()
                    pose_payload["num_markers"] = int(len(ids))

                    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

                    retval, rvec, tvec = self._estimate_board_pose(corners, ids, frame_bgr=annotated)
                    if rvec is not None and tvec is not None:
                        T = _rvec_tvec_to_T(rvec, tvec)  # T_avp<-board
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

                        # Update coordinate_manager with T_world_avp (not T_avp_board!)
                        # T_world_avp = T_world_rs @ T_rs_board @ inv(T_avp_board)
                        with coord_lock:
                            if coordinate_manager is not None:
                                try:
                                    # Get T_rs_board from RS ArUco detection
                                    with rs_aruco_lock:
                                        T_rs_board = rs_aruco_latest.get("pose_matrix")

                                    if T_rs_board is not None:
                                        # Get current head pose for reference tracking
                                        with head_pose_lock:
                                            head_pos = head_pose_latest.get("position") if head_pose_latest else None
                                            head_quat = head_pose_latest.get("quaternion") if head_pose_latest else None

                                        # Construct T_world_head if head pose available
                                        T_world_head = None
                                        if head_pos is not None and head_quat is not None:
                                            try:
                                                rot = Rotation.from_quat(head_quat).as_matrix()
                                                T_world_head = np.eye(4, dtype=np.float64)
                                                T_world_head[:3, :3] = rot
                                                T_world_head[:3, 3] = head_pos
                                                logger.debug("Captured reference head pose for continuous tracking")
                                            except Exception as e:
                                                logger.warning("Failed to construct T_world_head: %s", e)
                                                T_world_head = None

                                        # Compute T_world_avp correctly
                                        T_world_rs = coordinate_manager.get_T_world_rs()
                                        T_avp_board = T  # Current detection
                                        T_world_avp = T_world_rs @ T_rs_board @ np.linalg.inv(T_avp_board)

                                        # Set calibration with reference head pose for continuous tracking
                                        coordinate_manager.set_avp_calibration(T_world_avp, T_world_head)
                                        logger.debug("Updated coordinate_manager with T_world_avp from dual ArUco detection")
                                    else:
                                        logger.debug("Cannot update coordinate_manager: RS ArUco not available")
                                except Exception as e:
                                    logger.warning("Failed to update coordinate_manager: %s", e)

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
        frame_index = 0
        while True:
            if rs_capture is None:
                time.sleep(0.05)
                continue
            latest = rs_capture.get_latest()
            rgb = latest.get("rgb")
            K = latest.get("K")
            dist = latest.get("dist")
            ts = latest.get("timestamp") or time.time()
            if rgb is None or K is None:
                time.sleep(0.02)
                continue

            frame_index += 1
            stride = _get_processing_stride()
            if stride > 1 and (frame_index % stride) != 0:
                time.sleep(0.05)
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
                    dist_coeffs = dist if dist is not None else np.zeros((5, 1), dtype=np.float32)
                    pose = detector.estimate_board_pose(corners, ids, K, dist_coeffs)
                    if pose is not None:
                        rvec = pose[0].reshape(3, 1)
                        tvec = pose[1].reshape(3, 1)
                        pose_matrix = ArucoDetector.pose_to_transformation_matrix(rvec.reshape(3), tvec.reshape(3)).astype(np.float32)
                        overlay = _draw_axes(overlay, rvec, tvec, K, dist_coeffs,
                                             length=rs_cfg["marker_size_m"] * 2.0,
                                             label="RS Aruco")
                cv2.putText(overlay, "RS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
            except Exception as e:
                logger.warning("RS ArUco detection failed: %s", e)

            with rs_aruco_lock:
                rs_aruco_latest.update(
                    {
                        "overlay": overlay,
                        "pose_matrix": pose_matrix,  # T_rs<-board
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
    """
    Returns T_avp<-rs.

    Priority:
    1) coordinate_manager if calibrated (assumed correct)
    2) fallback from simultaneous ArUco poses:
       T_avp<-rs = T_avp<-board @ inv(T_rs<-board)
    """
    with coord_lock:
        if coordinate_manager is not None and coordinate_manager.is_calibrated():
            try:
                return coordinate_manager.get_T_avp_rs()
            except Exception:
                pass

    with avp_pose_lock:
        T_avp_board = avp_pose_latest.get("pose_matrix")  # T_avp<-board
    with rs_aruco_lock:
        T_rs_board = rs_aruco_latest.get("pose_matrix")   # T_rs<-board

    if T_avp_board is None or T_rs_board is None:
        return None

    try:
        return (T_avp_board @ np.linalg.inv(T_rs_board)).astype(np.float32)
    except Exception:
        return None


def _get_selected_model() -> str:
    if selected_model:
        return selected_model
    return CONFIG["processing"]["default_model"]


# def _consume_save_next_foundationpose() -> bool:
#     global foundationpose_save_next
#     with foundationpose_save_lock:
#         if foundationpose_save_next:
#             foundationpose_save_next = False
#             return True
#     return False


def _get_processing_stride() -> int:
    with processing_stride_lock:
        return max(1, int(processing_stride))


def _save_foundationpose_request(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    pose: Optional[np.ndarray],
    meta: Dict[str, Any],
) -> None:
    try:
        base_dir = os.path.join(CONFIG["uxplay"]["frame_dir"], "foundationpose")
        os.makedirs(base_dir, exist_ok=True)

        cid = meta.get("capture_id") or time.strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(base_dir, f"fp_{cid}")

        # rgb is RGB; cv2.imwrite expects BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(prefix + "_rgb.jpg", bgr)

        cv2.imwrite(prefix + "_mask.png", mask)
        np.save(prefix + "_depth.npy", depth.astype(np.float32))
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(prefix + "_depth.png", depth_norm)

        if pose is not None:
            np.save(prefix + "_pose.npy", pose.astype(np.float32))

        meta_path = prefix + "_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved FoundationPose AVP request: %s", prefix)
    except Exception as e:
        logger.warning("Failed to save FoundationPose request: %s", e)


def _save_foundationpose_rs_request(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    pose: Optional[np.ndarray],
    meta: Dict[str, Any],
) -> None:
    try:
        base_dir = os.path.join(CONFIG["uxplay"]["frame_dir"], "foundationpose_rs")
        os.makedirs(base_dir, exist_ok=True)

        cid = meta.get("capture_id") or time.strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(base_dir, f"fp_rs_{cid}")

        # rgb is RGB; cv2.imwrite expects BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(prefix + "_rgb.jpg", bgr)

        cv2.imwrite(prefix + "_mask.png", mask)
        np.save(prefix + "_depth.npy", depth.astype(np.float32))
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(prefix + "_depth.png", depth_norm)

        if pose is not None:
            np.save(prefix + "_pose.npy", pose.astype(np.float32))

        meta_path = prefix + "_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved FoundationPose RS request: %s", prefix)
    except Exception as e:
        logger.warning("Failed to save FoundationPose RS request: %s", e)


def _start_foundationpose_worker(capture: UxPlayCapture, processor: "ArucoProcessor") -> threading.Thread:
    """
    FoundationPose in AVP camera coordinates:
    - RGB comes from UxPlay AVP frame
    - Depth comes from RS aligned depth projected into AVP using T_avp<-rs and AVP intrinsics
    - Mask comes from normalized AVP ROI (no HSV)
    """
    def loop():
        frame_index = 0
        while True:
            frame, ts = capture.get_latest()
            if frame is None:
                with foundationpose_lock:
                    foundationpose_latest.update(
                        {"pose_matrix": None, "timestamp": None, "message": "No AVP frame yet", "success": False}
                    )
                time.sleep(0.2)
                continue

            frame_index += 1
            stride = _get_processing_stride()
            if stride > 1 and (frame_index % stride) != 0:
                time.sleep(0.2)
                continue

            if rs_capture is None:
                with foundationpose_lock:
                    foundationpose_latest.update(
                        {"pose_matrix": None, "timestamp": None, "message": "RealSense not connected", "success": False}
                    )
                time.sleep(0.5)
                continue

            latest = rs_capture.get_latest()
            depth_rs = latest.get("depth")
            K_rs = latest.get("K")
            dist_rs = latest.get("dist")
            aligned_ok = latest.get("aligned_ok")
            if depth_rs is None or K_rs is None:
                with foundationpose_lock:
                    foundationpose_latest.update(
                        {"pose_matrix": None, "timestamp": ts, "message": "No RealSense depth yet", "success": False}
                    )
                time.sleep(0.2)
                continue

            T_avp_rs = _get_T_avp_rs()
            if T_avp_rs is None:
                with foundationpose_lock:
                    foundationpose_latest.update(
                        {"pose_matrix": None, "timestamp": ts, "message": "Missing RS->AVP calibration", "success": False}
                    )
                time.sleep(0.5)
                continue

            try:
                depth_avp = _transform_depth_rs_to_avp(
                    depth_rs,
                    K_rs,
                    dist_rs,
                    T_avp_rs,
                    processor.K,
                    processor.dist,
                    (capture.height, capture.width),
                )
            except Exception as e:
                with foundationpose_lock:
                    foundationpose_latest.update(
                        {"pose_matrix": None, "timestamp": ts, "message": f"Depth transform failed: {e}", "success": False}
                    )
                time.sleep(0.5)
                continue

            # AVP ROI mask (normalized circle)
            mask = _compute_avp_roi_mask(capture.height, capture.width)

            model_name = _get_selected_model()
            mesh_path = model_name
            if not os.path.isabs(mesh_path):
                mesh_path = os.path.join(CONFIG["paths"]["models_dir"], mesh_path)
            if not os.path.exists(mesh_path):
                with foundationpose_lock:
                    foundationpose_latest.update(
                        {"pose_matrix": None, "timestamp": ts, "message": f"Missing mesh: {model_name}", "success": False}
                    )
                time.sleep(1.0)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = estimate_pose(
                rgb=rgb,
                depth=depth_avp,
                mask=mask,
                K=processor.K,
                mesh_path=mesh_path,
                api_url=CONFIG["network"]["foundationpose_url"],
            )

            # if _consume_save_next_foundationpose():
            #     _save_foundationpose_request(
            #         rgb=rgb,
            #         depth=depth_avp,
            #         mask=mask,
            #         pose=pose,
            #         meta={
            #             "timestamp": ts,
            #             "model": model_name,
            #             "foundationpose_url": CONFIG["network"]["foundationpose_url"],
            #             "rs_depth_aligned_to_color": bool(aligned_ok),
            #             "T_avp_rs": None if T_avp_rs is None else T_avp_rs.tolist(),
            #             "avp_roi": vars(avp_roi_cfg),
            #         },
            #     )

            capture_id = _consume_save_avp()
            if capture_id is not None:
                # include both directions
                T_rs_avp = None
                try:
                    T_rs_avp = np.linalg.inv(T_avp_rs).astype(np.float32).tolist()
                except Exception:
                    pass

                with avp_pose_lock:
                    T_avp_board = avp_pose_latest.get("pose_matrix")
                with rs_aruco_lock:
                    T_rs_board = rs_aruco_latest.get("pose_matrix")

                _save_foundationpose_request(
                    rgb=rgb,
                    depth=depth_avp,
                    mask=mask,
                    pose=pose,
                    meta={
                        "capture_id": capture_id,
                        "source": "avp",
                        "timestamp": ts,
                        "model": model_name,
                        "foundationpose_url": CONFIG["network"]["foundationpose_url"],

                        "rs_depth_aligned_to_color": bool(aligned_ok),

                        # key transforms for later evaluation
                        "T_avp_rs": T_avp_rs.astype(np.float32).tolist() if T_avp_rs is not None else None,
                        "T_rs_avp": T_rs_avp,

                        # optional but very useful debugging context
                        "T_avp_board": None if T_avp_board is None else T_avp_board.astype(np.float32).tolist(),
                        "T_rs_board": None if T_rs_board is None else T_rs_board.astype(np.float32).tolist(),

                        "avp_roi": vars(avp_roi_cfg),
                    },
                )


            if pose is None:
                with foundationpose_lock:
                    foundationpose_latest.update(
                        {"pose_matrix": None, "timestamp": ts, "message": "FoundationPose returned no pose", "success": False}
                    )
                time.sleep(1.0)
                continue

            with foundationpose_lock:
                foundationpose_latest.update(
                    {"pose_matrix": pose, "timestamp": ts, "message": "ok", "success": True}
                )

            time.sleep(0.8)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def _start_foundationpose_rs_worker() -> threading.Thread:
    def loop():
        frame_index = 0
        while True:
            if rs_capture is None:
                with foundationpose_rs_lock:
                    foundationpose_rs_latest.update(
                        {"pose_matrix": None, "timestamp": None, "message": "RealSense not connected", "success": False}
                    )
                time.sleep(0.5)
                continue

            latest = rs_capture.get_latest()
            rgb_bgr = latest.get("rgb")
            depth = latest.get("depth")
            K = latest.get("K")
            ts = latest.get("timestamp") or time.time()
            if rgb_bgr is None or depth is None or K is None:
                with foundationpose_rs_lock:
                    foundationpose_rs_latest.update(
                        {"pose_matrix": None, "timestamp": ts, "message": "Missing RS frame", "success": False}
                    )
                time.sleep(0.2)
                continue

            frame_index += 1
            stride = _get_processing_stride()
            if stride > 1 and (frame_index % stride) != 0:
                time.sleep(0.2)
                continue

            with rs_roi_lock:
                cx = int(rs_roi_config["x_center"])
                cy = int(rs_roi_config["y_center"])
                radius = int(rs_roi_config["radius"])

            mask = np.zeros(rgb_bgr.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)

            model_name = _get_selected_model()
            mesh_path = model_name
            if not os.path.isabs(mesh_path):
                mesh_path = os.path.join(CONFIG["paths"]["models_dir"], mesh_path)
            if not os.path.exists(mesh_path):
                with foundationpose_rs_lock:
                    foundationpose_rs_latest.update(
                        {"pose_matrix": None, "timestamp": ts, "message": f"Missing mesh: {model_name}", "success": False}
                    )
                time.sleep(1.0)
                continue

            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            pose = estimate_pose(
                rgb=rgb,
                depth=depth,
                mask=mask,
                K=K,
                mesh_path=mesh_path,
                api_url=CONFIG["network"]["foundationpose_url"],
            )

            # if _consume_save_next_foundationpose():
            #     _save_foundationpose_rs_request(
            #         rgb=rgb,
            #         depth=depth,
            #         mask=mask,
            #         pose=pose,
            #         meta={
            #             "timestamp": ts,
            #             "model": model_name,
            #             "foundationpose_url": CONFIG["network"]["foundationpose_url"],
            #             "roi": {"x_center": cx, "y_center": cy, "radius": radius},
            #         },
            #     )

            capture_id = _consume_save_rs()
            if capture_id is not None:
                # grab transform at save time (best effort)
                T_avp_rs_now = _get_T_avp_rs()
                T_rs_avp = None
                try:
                    if T_avp_rs_now is not None:
                        T_rs_avp = np.linalg.inv(T_avp_rs_now).astype(np.float32).tolist()
                except Exception:
                    pass

                with avp_pose_lock:
                    T_avp_board = avp_pose_latest.get("pose_matrix")
                with rs_aruco_lock:
                    T_rs_board = rs_aruco_latest.get("pose_matrix")

                _save_foundationpose_rs_request(
                    rgb=rgb,
                    depth=depth,
                    mask=mask,
                    pose=pose,
                    meta={
                        "capture_id": capture_id,
                        "source": "rs",
                        "timestamp": ts,
                        "model": model_name,
                        "foundationpose_url": CONFIG["network"]["foundationpose_url"],

                        # key transforms for later evaluation
                        "T_avp_rs": None if T_avp_rs_now is None else T_avp_rs_now.astype(np.float32).tolist(),
                        "T_rs_avp": T_rs_avp,

                        # optional debug
                        "T_avp_board": None if T_avp_board is None else T_avp_board.astype(np.float32).tolist(),
                        "T_rs_board": None if T_rs_board is None else T_rs_board.astype(np.float32).tolist(),

                        "roi": {"x_center": cx, "y_center": cy, "radius": radius},
                    },
                )


            if pose is None:
                with foundationpose_rs_lock:
                    foundationpose_rs_latest.update(
                        {"pose_matrix": None, "timestamp": ts, "message": "FoundationPose returned no pose", "success": False}
                    )
                logger.warning("FoundationPose RS returned no pose")
                time.sleep(1.0)
                continue

            with foundationpose_rs_lock:
                foundationpose_rs_latest.update(
                    {"pose_matrix": pose, "timestamp": ts, "message": "ok", "success": True}
                )

            time.sleep(0.8)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# -----------------------------
# Flask API
# -----------------------------
def create_app(capture: UxPlayCapture, processor: ArucoProcessor) -> Flask:
    app = Flask(__name__)
    MJPEG_FPS = 10
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        with coord_lock:
            calibrated = coordinate_manager is not None and coordinate_manager.is_calibrated()

        rs_connected = rs_capture is not None and rs_capture.running
        rs_aligned = None
        if rs_capture is not None:
            rs_aligned = rs_capture.get_latest().get("aligned_ok")

        with avp_roi_lock:
            roi_state = vars(avp_roi_cfg).copy()

        return jsonify(
            {
                "status": "ok",
                "uxplay_running": capture.running,
                "rs_connected": rs_connected,
                "rs_depth_aligned_to_color": rs_aligned,
                "calibrated": calibrated,
                "frames_received": capture.frames_received,
                "processed_frames": processor.processed_frames,
                "detected_frames": processor.detected_frames,
                "resolution": f"{capture.width}x{capture.height}",
                "process_fps": processor.process_fps,
                "avp_roi": roi_state,
            }
        ), 200

    # @app.route("/foundationpose_save_next", methods=["GET", "POST"])
    # def foundationpose_save_next_route():
    #     global foundationpose_save_next
    #     if request.method == "GET":
    #         with foundationpose_save_lock:
    #             return jsonify({"enabled": foundationpose_save_next}), 200

    #     data = request.get_json(silent=True) or {}
    #     enabled = bool(data.get("enabled", False))
    #     with foundationpose_save_lock:
    #         foundationpose_save_next = enabled
    #     return jsonify({"enabled": foundationpose_save_next}), 200

    # @app.route("/foundationpose_save_next", methods=["GET", "POST"])
    # def foundationpose_save_next_route():
    #     if request.method == "GET":
    #         return jsonify(_get_save_pair_state()), 200

    #     # POST
    #     data = request.get_json(silent=True) or {}
    #     enabled = bool(data.get("enabled", False))
    #     if enabled:
    #         st = _arm_save_pair()
    #         return jsonify({"armed": True, **st}), 200
    #     else:
    #         # disarm
    #         with foundationpose_save_lock:
    #             foundationpose_save_pair.update({
    #                 "armed": False,
    #                 "capture_id": None,
    #                 "need_avp": False,
    #                 "need_rs": False,
    #                 "armed_time": None,
    #             })
    #         return jsonify(_get_save_pair_state()), 200

    @app.route("/foundationpose_save_next", methods=["GET", "POST"])
    def foundationpose_save_next_route():
        if request.method == "GET":
            return jsonify(_get_save_pair_state()), 200

        data = request.get_json(silent=True) or {}
        enabled = bool(data.get("enabled", False))

        if enabled:
            st = _arm_save_pair()
            return jsonify({"armed": True, **st}), 200

        # disarm
        with foundationpose_save_lock:
            foundationpose_save_pair.update({
                "armed": False,
                "capture_id": None,
                "need_avp": False,
                "need_rs": False,
                "armed_time": None,
            })
        return jsonify(_get_save_pair_state()), 200



    @app.route("/rs_roi_config", methods=["GET", "POST"])
    def rs_roi_config_route():
        if request.method == "GET":
            with rs_roi_lock:
                return jsonify(dict(rs_roi_config)), 200

        data = request.get_json(silent=True) or {}
        with rs_roi_lock:
            if "x_center" in data:
                rs_roi_config["x_center"] = int(data["x_center"])
            if "y_center" in data:
                rs_roi_config["y_center"] = int(data["y_center"])
            if "radius" in data:
                rs_roi_config["radius"] = int(data["radius"])
        return jsonify(dict(rs_roi_config)), 200

    # NEW: AVP normalized ROI config (replaces HSV)
    @app.route("/avp_roi_config", methods=["GET", "POST"])
    def avp_roi_config_route():
        """
        GET: current ROI config
        POST: update fields {enabled, cx_n, cy_n, r_n}
          - cx_n, cy_n in [0..1]
          - r_n in [0..1] relative to min(w,h)
        """
        if request.method == "GET":
            with avp_roi_lock:
                return jsonify(vars(avp_roi_cfg)), 200

        data = request.get_json(silent=True) or {}
        with avp_roi_lock:
            if "enabled" in data:
                avp_roi_cfg.enabled = bool(data["enabled"])
            if "cx_n" in data:
                avp_roi_cfg.cx_n = _clamp_float(data["cx_n"], 0.0, 1.0)
            if "cy_n" in data:
                avp_roi_cfg.cy_n = _clamp_float(data["cy_n"], 0.0, 1.0)
            if "r_n" in data:
                avp_roi_cfg.r_n = _clamp_float(data["r_n"], 0.0, 1.0)
            return jsonify(vars(avp_roi_cfg)), 200

    @app.route("/processing_stride", methods=["GET", "POST"])
    def processing_stride_route():
        global processing_stride
        if request.method == "GET":
            return jsonify({"stride": _get_processing_stride()}), 200

        data = request.get_json(silent=True) or {}
        stride = int(data.get("stride", 1))
        with processing_stride_lock:
            processing_stride = max(1, stride)
        return jsonify({"stride": _get_processing_stride()}), 200

    @app.route("/models", methods=["GET"])
    def models():
        try:
            # your config sometimes uses "models" sometimes "models_dir"
            models_dir = Path(CONFIG["paths"].get("models", CONFIG["paths"].get("models_dir", "")))
            if not models_dir or not models_dir.exists():
                return jsonify({"models": []}), 200
            model_files = sorted([f.name for f in models_dir.glob("*.ply")])
            return jsonify({"models": model_files}), 200
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/select_model", methods=["POST"])
    def select_model():
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
        return jsonify(
            {
                "K": processor.K.tolist(),
                "dist": processor.dist.reshape(-1).tolist(),
            }
        ), 200

    @app.route("/get_rgbd_frame", methods=["GET"])
    def get_rgbd_frame():
        if rs_capture is None:
            return jsonify({"error": "RealSense not connected"}), 503
        latest = rs_capture.get_latest()
        rgb = latest.get("rgb")
        depth = latest.get("depth")
        ts = latest.get("timestamp")
        aligned_ok = latest.get("aligned_ok")
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
                "depth_aligned_to_color": bool(aligned_ok),
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
        has_avp_overlay = False

        try:
            annotated = _decode_jpeg_b64(processor.get_aruco().get("rgb"))
            if annotated is not None:
                out = annotated
                has_avp_overlay = True
        except Exception:
            pass
        K_avp = processor.K
        dist = processor.dist

        with avp_pose_lock:
            avp_pose = avp_pose_latest.get("pose_matrix")

        if avp_pose is not None and not has_avp_overlay:
            rvec, tvec = _T_to_rvec_tvec(avp_pose)
            out = _draw_axes(out, rvec, tvec, K_avp, dist, processor.cfg.axis_length_m, label="Aruco")

        T_avp_rs = _get_T_avp_rs()
        if T_avp_rs is not None:
            rvec, tvec = _T_to_rvec_tvec(T_avp_rs)
            out = _draw_axes(out, rvec, tvec, K_avp, dist, processor.cfg.axis_length_m, label="RS")

        with foundationpose_lock:
            fp_pose = foundationpose_latest.get("pose_matrix")
        if fp_pose is not None:
            rvec, tvec = _T_to_rvec_tvec(fp_pose)
            out = _draw_axes(out, rvec, tvec, K_avp, dist, processor.cfg.axis_length_m, label="foundationpose")

        rgb_b64 = _encode_jpeg_b64(out, quality=85)
        return jsonify(
            {
                "rgb": f"data:image/jpeg;base64,{rgb_b64}",
                "timestamp": ts,
            }
        ), 200

    # now shows ROI mask (not HSV)
    @app.route("/get_avp_mask_frame", methods=["GET"])
    def get_avp_mask_frame():
        frame, ts = capture.get_latest()
        if frame is None:
            return jsonify({"error": "No AVP frame yet"}), 503
        mask = _compute_avp_roi_mask(capture.height, capture.width)
        out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
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
        dist_rs = latest.get("dist")
        aligned_ok = latest.get("aligned_ok")
        if depth is None or K_rs is None:
            return jsonify({"error": "No RealSense depth yet"}), 503
        T_avp_rs = _get_T_avp_rs()
        if T_avp_rs is None:
            return jsonify({"error": "Missing calibration for RS->AVP transform"}), 503

        K_avp = processor.K
        dist_avp = processor.dist
        depth_avp = _transform_depth_rs_to_avp(
            depth,
            K_rs,
            dist_rs,
            T_avp_rs,
            K_avp,
            dist_avp,
            (capture.height, capture.width),
        )
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
                "rs_depth_aligned_to_color": bool(aligned_ok),
            }
        ), 200

    @app.route("/get_intrinsics", methods=["GET"])
    def get_intrinsics():
        rs_K = None
        rs_dist = None
        rs_aligned = None
        if rs_capture is not None:
            latest = rs_capture.get_latest()
            K = latest.get("K")
            dist = latest.get("dist")
            rs_aligned = latest.get("aligned_ok")
            if K is not None:
                rs_K = K.tolist()
            if dist is not None:
                rs_dist = dist.reshape(-1).tolist()
        return jsonify(
            {
                "rs": {
                    "K": rs_K,
                    "dist": rs_dist,
                    "depth_aligned_to_color": rs_aligned,
                    "method": "realsense_sdk_color_intrinsics",
                    "timestamp": time.time(),
                },
                "avp": {
                    "K": processor.K.tolist(),
                    "dist": processor.dist.reshape(-1).tolist(),
                    "method": "intrinsics.json",
                    "timestamp": time.time(),
                },
            }
        ), 200

    @app.route("/get_transformation", methods=["GET"])
    def get_transformation():
        T_avp_rs = _get_T_avp_rs()
        with avp_pose_lock:
            T_avp_board = avp_pose_latest.get("pose_matrix")
        with rs_aruco_lock:
            T_rs_board = rs_aruco_latest.get("pose_matrix")

        calibrated = T_avp_rs is not None
        return jsonify(
            {
                "T_avp_rs": None if T_avp_rs is None else T_avp_rs.tolist(),
                "T_avp_board": None if T_avp_board is None else T_avp_board.tolist(),
                "T_rs_board": None if T_rs_board is None else T_rs_board.tolist(),
                "calibrated": calibrated,
                "message": "ok" if calibrated else "Missing calibration",
                "note": "fallback uses T_avp<-rs = T_avp<-board @ inv(T_rs<-board)",
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

    @app.route("/get_foundationpose_pose", methods=["GET"])
    def get_foundationpose_pose():
        with foundationpose_lock:
            pose = foundationpose_latest.get("pose_matrix")
            ts = foundationpose_latest.get("timestamp")
            msg = foundationpose_latest.get("message")
            ok = foundationpose_latest.get("success")
        return jsonify(
            {
                "pose_matrix": None if pose is None else pose.tolist(),
                "timestamp": ts,
                "success": bool(ok),
                "message": msg,
            }
        ), 200

    @app.route("/get_foundationpose_rs_pose", methods=["GET"])
    def get_foundationpose_rs_pose():
        with foundationpose_rs_lock:
            pose = foundationpose_rs_latest.get("pose_matrix")
            ts = foundationpose_rs_latest.get("timestamp")
            msg = foundationpose_rs_latest.get("message")
            ok = foundationpose_rs_latest.get("success")
        return jsonify(
            {
                "pose_matrix": None if pose is None else pose.tolist(),
                "timestamp": ts,
                "success": bool(ok),
                "message": msg,
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

    # keep endpoint name for compatibility; now returns ROI config instead of HSV
    @app.route("/hsv_config", methods=["GET", "POST"])
    def hsv_config_route():
        """
        Compatibility endpoint. HSV ROI is removed.
        Returns AVP ROI config.
        """
        if request.method == "GET":
            with avp_roi_lock:
                return jsonify({"deprecated": True, "avp_roi": vars(avp_roi_cfg)}), 200

        data = request.get_json(force=True, silent=True) or {}
        # accept same POST but map to ROI if user tries
        with avp_roi_lock:
            if "enabled" in data:
                avp_roi_cfg.enabled = bool(data["enabled"])
            if "cx_n" in data:
                avp_roi_cfg.cx_n = _clamp_float(data["cx_n"], 0.0, 1.0)
            if "cy_n" in data:
                avp_roi_cfg.cy_n = _clamp_float(data["cy_n"], 0.0, 1.0)
            if "r_n" in data:
                avp_roi_cfg.r_n = _clamp_float(data["r_n"], 0.0, 1.0)
        return jsonify({"deprecated": True, "avp_roi": vars(avp_roi_cfg)}), 200

    @app.route("/debug", methods=["GET"])
    def debug_page():
        rs_w = CONFIG["realsense"]["resolution_width"]
        rs_h = CONFIG["realsense"]["resolution_height"]
        html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Debug Views</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 14px; }}
    .panel {{ border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
    .panel.live {{ max-width: 980px; margin: 0 auto 12px auto; }}
    .hdr {{ padding: 8px 10px; background: #f7f7f7; border-bottom: 1px solid #eee; font-weight: 600; }}
    img {{ width: 100%; display: block; background: #000; max-height: 420px; object-fit: contain; }}
    .controls {{ margin-top: 12px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; max-width: 980px; margin-left:auto; margin-right:auto; }}
    .row {{ display: flex; align-items: center; gap: 10px; margin: 8px 0; }}
    .row label {{ width: 170px; }}
    input[type="range"] {{ width: 360px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
  </style>
</head>
<body>
  <h2 style="margin: 0 0 10px 0;">Debug Views</h2>

  <div class="panel live">
    <div class="hdr">Live View</div>
    <div style="padding: 10px; background: #fff;">
      <label for="viewSelect" style="font-weight: 600; margin-right: 8px;">View:</label>
      <select id="viewSelect">
        <option value="overlay">AVP ArUco Overlay</option>
        <option value="raw">AVP Raw (UxPlay)</option>
        <option value="mask">AVP ROI Mask (normalized)</option>
        <option value="rs_rgb">RS RGB</option>
        <option value="rs_depth">RS Depth</option>
        <option value="rs_aruco">RS ArUco Overlay</option>
        <option value="rs_roi">RS ROI</option>
        <option value="avp_rs">AVP + RS Pose Overlay</option>
        <option value="avp_depth">RS Depth → AVP</option>
        <option value="fp_avp_on_avp">FP (AVP request) → AVP view</option>
        <option value="fp_avp_on_rs">FP (AVP request) → RS view</option>
        <option value="fp_rs_on_avp">FP (RS request) → AVP view</option>
        <option value="fp_rs_on_rs">FP (RS request) → RS view</option>
      </select>
    </div>
    <img id="viewImg" src="/mjpeg?view=overlay" />
  </div>

  <div class="controls">
    <div style="font-weight:600; margin-bottom:6px;">RS ROI (pixels)</div>
    <div class="row">
      <label>ROI X Center</label>
      <input id="roiX" type="range" min="0" max="{rs_w}" value="{rs_w // 2}" />
      <span class="mono" id="roiXOut"></span>
    </div>
    <div class="row">
      <label>ROI Y Center</label>
      <input id="roiY" type="range" min="0" max="{rs_h}" value="{rs_h // 2}" />
      <span class="mono" id="roiYOut"></span>
    </div>
    <div class="row">
      <label>ROI Radius</label>
      <input id="roiR" type="range" min="10" max="{max(rs_w, rs_h) // 2}" value="120" />
      <span class="mono" id="roiROut"></span>
    </div>
    <div class="row">
      <label>Save next request</label>
      <button id="saveNext">Save next request</button>
      <span class="mono" id="saveState"></span>
    </div>
  </div>

<script>
async function postROI(payload) {{
  await fetch("/rs_roi_config", {{
    method: "POST",
    headers: {{"Content-Type":"application/json"}},
    body: JSON.stringify(payload)
  }});
}}

async function postSave(enabled) {{
  await fetch("/foundationpose_save_next", {{
    method: "POST",
    headers: {{"Content-Type":"application/json"}},
    body: JSON.stringify({{enabled}})
  }});
}}

async function loadROI() {{
  const r = await fetch("/rs_roi_config");
  return await r.json();
}}

async function wire() {{
  const viewSelect = document.getElementById("viewSelect");
  const viewImg = document.getElementById("viewImg");

  // RS ROI
  const roiX = document.getElementById("roiX");
  const roiY = document.getElementById("roiY");
  const roiR = document.getElementById("roiR");
  const roiXOut = document.getElementById("roiXOut");
  const roiYOut = document.getElementById("roiYOut");
  const roiROut = document.getElementById("roiROut");

  const saveBtn = document.getElementById("saveNext");
  const saveState = document.getElementById("saveState");

  viewSelect.addEventListener("change", () => {{
    const view = encodeURIComponent(viewSelect.value);
    viewImg.src = `/mjpeg?view=${{view}}&_ts=${{Date.now()}}`;
  }});

  // init RS ROI
  const cfg = await loadROI();
  roiX.value = cfg.x_center;
  roiY.value = cfg.y_center;
  roiR.value = cfg.radius;
  roiXOut.textContent = cfg.x_center;
  roiYOut.textContent = cfg.y_center;
  roiROut.textContent = cfg.radius;

  function updateROI() {{
    roiXOut.textContent = roiX.value;
    roiYOut.textContent = roiY.value;
    roiROut.textContent = roiR.value;
    postROI({{x_center: roiX.value, y_center: roiY.value, radius: roiR.value}});
  }}

  roiX.addEventListener("input", updateROI);
  roiY.addEventListener("input", updateROI);
  roiR.addEventListener("input", updateROI);

  saveBtn.addEventListener("click", async () => {{
    await postSave(true);
    saveState.textContent = "armed";
    setTimeout(() => {{ saveState.textContent = ""; }}, 2000);
  }});
}}

wire();
</script>

</body>
</html>
"""
        return Response(html, mimetype="text/html")

    @app.route("/mjpeg", methods=["GET"])
    def mjpeg():
        """
        MJPEG stream with multiple views:
          /mjpeg?view=raw          -> raw UxPlay feed
          /mjpeg?view=overlay      -> AVP ArUco overlay
          /mjpeg?view=mask         -> AVP ROI mask (normalized)
          /mjpeg?view=rs_rgb       -> RealSense RGB
          /mjpeg?view=rs_depth     -> RealSense depth colormap
          /mjpeg?view=rs_aruco     -> RealSense ArUco overlay
          /mjpeg?view=rs_roi       -> RealSense RGB with ROI + FoundationPose overlay
          /mjpeg?view=avp_rs       -> AVP view with RS pose overlay
          /mjpeg?view=avp_depth    -> RS depth transformed to AVP view
          /mjpeg?view=fp_avp_on_avp -> FoundationPose (from AVP request) overlayed on AVP view
          /mjpeg?view=fp_avp_on_rs  -> FoundationPose (from AVP request) overlayed on RS view
          /mjpeg?view=fp_rs_on_avp  -> FoundationPose (from RS request) overlayed on AVP view
          /mjpeg?view=fp_rs_on_rs   -> FoundationPose (from RS request) overlayed on RS view
        """
        view = (request.args.get("view", "overlay") or "overlay").lower().strip()
        valid_views = (
            "raw", "overlay", "mask", "rs_rgb", "rs_depth", "rs_aruco", "rs_roi",
            "avp_rs", "avp_depth", "fp_avp_on_avp", "fp_avp_on_rs", "fp_rs_on_avp", "fp_rs_on_rs"
        )
        if view not in valid_views:
            view = "overlay"
        frame_interval = 1.0 / MJPEG_FPS

        K = processor.K.copy()
        dist = processor.dist.copy()

        def gen():
            while True:
                out = None

                if view in ("raw", "overlay", "mask", "avp_rs", "fp_avp_on_avp", "fp_rs_on_avp"):
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
                    mask = _compute_avp_roi_mask(capture.height, capture.width)
                    out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    with avp_roi_lock:
                        cfg = AvpRoiConfig(**vars(avp_roi_cfg))
                    hud = f"ROI norm: cx={cfg.cx_n:.3f} cy={cfg.cy_n:.3f} r={cfg.r_n:.3f} enabled={cfg.enabled}"
                    cv2.putText(out, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

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
                    ok = latest.get("aligned_ok")
                    cv2.putText(out, f"RS (aligned={ok})", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

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
                    ok = latest.get("aligned_ok")
                    cv2.putText(out, f"RS Depth (aligned={ok})", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                elif view == "rs_aruco":
                    with rs_aruco_lock:
                        overlay = rs_aruco_latest.get("overlay")
                    if overlay is None:
                        time.sleep(0.02)
                        continue
                    out = overlay.copy()

                elif view == "rs_roi":
                    if rs_capture is None:
                        time.sleep(0.05)
                        continue
                    latest = rs_capture.get_latest()
                    out = latest.get("rgb")
                    K_rs = latest.get("K")
                    if out is None or K_rs is None:
                        time.sleep(0.02)
                        continue
                    out = out.copy()
                    with rs_roi_lock:
                        cx = int(rs_roi_config["x_center"])
                        cy = int(rs_roi_config["y_center"])
                        radius = int(rs_roi_config["radius"])
                    cv2.circle(out, (cx, cy), radius, (0, 255, 255), 2)
                    with foundationpose_rs_lock:
                        fp_rs = foundationpose_rs_latest.get("pose_matrix")
                    if fp_rs is not None:
                        rvec, tvec = _T_to_rvec_tvec(fp_rs)
                        out = _draw_axes(out, rvec, tvec, K_rs, np.zeros((5, 1), dtype=np.float32),
                                         processor.cfg.axis_length_m, label="foundationpose")

                elif view == "avp_rs":
                    if out is None:
                        time.sleep(0.02)
                        continue
                    annotated = _decode_jpeg_b64(processor.get_aruco().get("rgb"))
                    if annotated is not None:
                        out = annotated
                    with avp_pose_lock:
                        avp_pose = avp_pose_latest.get("pose_matrix")
                    if avp_pose is not None:
                        rvec, tvec = _T_to_rvec_tvec(avp_pose)
                        out = _draw_axes(out, rvec, tvec, K, dist, processor.cfg.axis_length_m, label="Aruco")
                    T_avp_rs = _get_T_avp_rs()
                    if T_avp_rs is not None:
                        rvec, tvec = _T_to_rvec_tvec(T_avp_rs)
                        out = _draw_axes(out, rvec, tvec, K, dist, processor.cfg.axis_length_m, label="RS")
                    with foundationpose_lock:
                        fp_pose = foundationpose_latest.get("pose_matrix")
                    if fp_pose is not None:
                        rvec, tvec = _T_to_rvec_tvec(fp_pose)
                        out = _draw_axes(out, rvec, tvec, K, dist, processor.cfg.axis_length_m, label="foundationpose")

                elif view == "avp_depth":
                    if rs_capture is None:
                        time.sleep(0.05)
                        continue
                    latest = rs_capture.get_latest()
                    depth = latest.get("depth")
                    K_rs = latest.get("K")
                    dist_rs = latest.get("dist")
                    if depth is None or K_rs is None:
                        time.sleep(0.02)
                        continue
                    T_avp_rs = _get_T_avp_rs()
                    if T_avp_rs is None:
                        time.sleep(0.05)
                        continue
                    depth_avp = _transform_depth_rs_to_avp(
                        depth,
                        K_rs,
                        dist_rs,
                        T_avp_rs,
                        K,
                        dist,
                        (capture.height, capture.width),
                    )
                    depth_norm = cv2.normalize(depth_avp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    out = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                    cv2.putText(out, "RS Depth -> AVP (z-buffer)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                elif view == "fp_avp_on_avp":
                    # FoundationPose (from AVP request) overlayed on AVP view
                    if out is None:
                        time.sleep(0.02)
                        continue
                    annotated = _decode_jpeg_b64(processor.get_aruco().get("rgb"))
                    if annotated is not None:
                        out = annotated
                    with foundationpose_lock:
                        fp_pose = foundationpose_latest.get("pose_matrix")
                    if fp_pose is not None:
                        rvec, tvec = _T_to_rvec_tvec(fp_pose)
                        out = _draw_axes(out, rvec, tvec, K, dist, processor.cfg.axis_length_m, label="FP(AVP)")
                        txt = f"FP(AVP): t={tvec[0][0]:.3f},{tvec[1][0]:.3f},{tvec[2][0]:.3f}"
                        cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(out, "FP(AVP): no pose", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                elif view == "fp_rs_on_avp":
                    # FoundationPose (from RS request) overlayed on AVP view
                    if out is None:
                        time.sleep(0.02)
                        continue
                    annotated = _decode_jpeg_b64(processor.get_aruco().get("rgb"))
                    if annotated is not None:
                        out = annotated
                    # Need to transform RS pose to AVP frame
                    with foundationpose_rs_lock:
                        fp_rs_pose = foundationpose_rs_latest.get("pose_matrix")
                    if fp_rs_pose is not None:
                        T_avp_rs = _get_T_avp_rs()
                        if T_avp_rs is not None:
                            # Transform pose from RS to AVP: T_avp_object = T_avp_rs @ T_rs_object
                            fp_avp_pose = T_avp_rs @ fp_rs_pose
                            rvec, tvec = _T_to_rvec_tvec(fp_avp_pose)
                            out = _draw_axes(out, rvec, tvec, K, dist, processor.cfg.axis_length_m, label="FP(RS)")
                            txt = f"FP(RS): t={tvec[0][0]:.3f},{tvec[1][0]:.3f},{tvec[2][0]:.3f}"
                            cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(out, "FP(RS): no T_avp_rs", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(out, "FP(RS): no pose", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                elif view == "fp_avp_on_rs":
                    # FoundationPose (from AVP request) overlayed on RS view
                    if rs_capture is None:
                        time.sleep(0.05)
                        continue
                    latest = rs_capture.get_latest()
                    with rs_aruco_lock:
                        rs_overlay = rs_aruco_latest.get("overlay")
                    out = rs_overlay if rs_overlay is not None else latest.get("rgb")
                    K_rs = latest.get("K")
                    if out is None or K_rs is None:
                        time.sleep(0.02)
                        continue
                    out = out.copy()
                    with foundationpose_lock:
                        fp_avp_pose = foundationpose_latest.get("pose_matrix")
                    if fp_avp_pose is not None:
                        T_avp_rs = _get_T_avp_rs()
                        if T_avp_rs is not None:
                            # Transform pose from AVP to RS: T_rs_object = inv(T_avp_rs) @ T_avp_object
                            try:
                                T_rs_avp = np.linalg.inv(T_avp_rs)
                                fp_rs_pose = T_rs_avp @ fp_avp_pose
                                rvec, tvec = _T_to_rvec_tvec(fp_rs_pose)
                                out = _draw_axes(out, rvec, tvec, K_rs, np.zeros((5, 1), dtype=np.float32),
                                               processor.cfg.axis_length_m, label="FP(AVP)")
                                txt = f"FP(AVP): t={tvec[0][0]:.3f},{tvec[1][0]:.3f},{tvec[2][0]:.3f}"
                                cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            except np.linalg.LinAlgError:
                                cv2.putText(out, "FP(AVP): cannot invert T_avp_rs", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(out, "FP(AVP): no T_avp_rs", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(out, "FP(AVP): no pose", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                elif view == "fp_rs_on_rs":
                    # FoundationPose (from RS request) overlayed on RS view
                    if rs_capture is None:
                        time.sleep(0.05)
                        continue
                    latest = rs_capture.get_latest()
                    with rs_aruco_lock:
                        rs_overlay = rs_aruco_latest.get("overlay")
                    out = rs_overlay if rs_overlay is not None else latest.get("rgb")
                    K_rs = latest.get("K")
                    if out is None or K_rs is None:
                        time.sleep(0.02)
                        continue
                    out = out.copy()
                    with foundationpose_rs_lock:
                        fp_rs_pose = foundationpose_rs_latest.get("pose_matrix")
                    if fp_rs_pose is not None:
                        rvec, tvec = _T_to_rvec_tvec(fp_rs_pose)
                        out = _draw_axes(out, rvec, tvec, K_rs, np.zeros((5, 1), dtype=np.float32),
                                       processor.cfg.axis_length_m, label="FP(RS)")
                        txt = f"FP(RS): t={tvec[0][0]:.3f},{tvec[1][0]:.3f},{tvec[2][0]:.3f}"
                        cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(out, "FP(RS): no pose", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

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
                time.sleep(frame_interval)

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
    _start_foundationpose_worker(capture, processor)
    _start_foundationpose_rs_worker()

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
    logger.info("Endpoints: /health /rgb /aruco /get_rgbd_frame /get_rs_aruco_frame /get_avp_latest_frame /get_avp_aruco_frame "
                "/get_avp_rs_overlay /get_transformed_depth /get_rs_pose_in_avp /mjpeg /debug /avp_roi_config /rs_roi_config /processing_stride")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
