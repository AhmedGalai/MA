#!/usr/bin/env python3
"""RealSense helper for disparity and coordinate transforms."""

from __future__ import annotations

import os
import time
import threading
from typing import Optional, Dict, Any

import numpy as np
import cv2 as cv
import requests

try:  # optional dependency
    import pyrealsense2 as rs  # type: ignore
except Exception:  # pragma: no cover - optional hardware dependency
    rs = None


def _safe_print(msg: str):
    try:
        print(msg)
    except Exception:
        pass


class RealSenseDisparityAdapter:
    """Grabs RGB/Depth from a RealSense camera and aligns it with AVP data."""

    def __init__(self, main_api_base: Optional[str] = None):
        self.main_api_base = (
            main_api_base
            or os.environ.get("MAIN_API_BASE")
            or "http://localhost:5000"
        )
        self.pipeline = None
        self.align = None
        self.profile = None
        self.available = False
        self._lock = threading.Lock()

        self._aruco_dict = None
        self._aruco_detector = None
        self._aruco_api = "old"
        self._color_K = None
        self._color_dist = None

        if rs is None:
            _safe_print("[RealSense] pyrealsense2 not available - skipping camera init")
            return

        try:
            self._prepare_aruco()
            self._start_pipeline()
            self.available = True
        except Exception as exc:  # pragma: no cover - hardware init
            _safe_print(f"[RealSense] Failed to initialize camera: {exc}")
            self.stop()

    # ------------------------------------------------------------------
    def _prepare_aruco(self):
        if not hasattr(cv, "aruco"):
            raise RuntimeError("OpenCV built without ArUco support")
        aruco = cv.aruco
        try:
            self._aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        except Exception:
            self._aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        try:
            params = aruco.DetectorParameters()
        except Exception:
            params = aruco.DetectorParameters_create()
        if hasattr(aruco, "ArucoDetector"):
            self._aruco_detector = aruco.ArucoDetector(self._aruco_dict, params)
            self._aruco_api = "new"
        else:
            self._aruco_detector = params
            self._aruco_api = "old"

    def _start_pipeline(self):  # pragma: no cover - hardware init path
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        intr = (
            self.profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self._color_K = np.array(
            [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
            dtype=np.float32,
        )
        coeffs = intr.coeffs[:] if intr.coeffs else [0, 0, 0, 0, 0]
        self._color_dist = np.array(coeffs[:5], dtype=np.float32).reshape(-1, 1)
        _safe_print("[RealSense] Camera initialized")

    # ------------------------------------------------------------------
    def stop(self):  # pragma: no cover - shutdown helper
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None
        self.align = None
        self.available = False

    # ------------------------------------------------------------------
    def capture_and_process(self) -> Optional[Dict[str, Any]]:
        if not self.available or self.pipeline is None or self.align is None:
            return None

        with self._lock:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1500)
            except Exception as exc:
                _safe_print(f"[RealSense] wait_for_frames failed: {exc}")
                return None

        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        disparity = self._depth_to_disparity(depth_image)
        pattern = self._detect_pattern(color_image)
        main_api = self._fetch_main_api_snapshot()
        transform = self._coordinate_transform(pattern, main_api)
        transformed = self._apply_transform(disparity, transform)

        return {
            "color_frame": color_image,
            "depth_map": depth_image,
            "disparity": disparity,
            "pattern_pose": pattern,
            "main_api": main_api,
            "transform": transform,
            "transformed_disparity": transformed,
            "pattern_view": self._pattern_overlay(color_image, pattern),
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    def _depth_to_disparity(self, depth: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            disparity = np.where(depth > 0, 1.0 / depth, 0.0)
        disparity = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
        return disparity.astype(np.uint8)

    def _detect_pattern(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
            if self._aruco_dict is None:
                return None
            if self._aruco_api == "new":
                detector = cv.aruco.ArucoDetector(self._aruco_dict, self._aruco_detector)
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv.aruco.detectMarkers(
                    gray, self._aruco_dict, parameters=self._aruco_detector
                )
            if ids is None or len(ids) == 0:
                return None
            pose = _solve_board_pose(corners, ids, self._color_K, self._color_dist)
            return pose
        except Exception as exc:
            _safe_print(f"[RealSense] Pattern detection failed: {exc}")
            return None

    def _pattern_overlay(self, frame: np.ndarray, pose) -> np.ndarray:
        viz = frame.copy()
        try:
            if pose is None:
                return viz
            rvec, tvec = pose
            cv.drawFrameAxes(viz, self._color_K, self._color_dist, rvec, tvec, 0.05)
        except Exception:
            pass
        return viz

    def _fetch_main_api_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "head_pose": None,
            "avp_pose": None,
            "intrinsics": None,
        }
        if not self.main_api_base:
            return snapshot

        try:
            r = requests.get(f"{self.main_api_base}/head_pose", timeout=0.5)
            if r.ok:
                snapshot["head_pose"] = r.json().get("head_pose")
        except Exception as exc:
            _safe_print(f"[RealSense] head_pose fetch failed: {exc}")

        try:
            r = requests.get(f"{self.main_api_base}/pose", timeout=0.5)
            if r.ok:
                snapshot["avp_pose"] = r.json()
        except Exception as exc:
            _safe_print(f"[RealSense] pose fetch failed: {exc}")

        try:
            r = requests.get(f"{self.main_api_base}/intrinsics", timeout=0.5)
            if r.ok:
                snapshot["intrinsics"] = r.json()
        except Exception as exc:
            _safe_print(f"[RealSense] intrinsics fetch failed: {exc}")

        return snapshot

    def _coordinate_transform(self, rs_pose, snapshot) -> Optional[np.ndarray]:
        try:
            if rs_pose is None:
                return None
            if not snapshot or not snapshot.get("avp_pose"):
                return None

            avp_pose = snapshot["avp_pose"]
            rvec = np.array(avp_pose.get("rvec", [])).reshape(-1, 1)
            tvec = np.array(avp_pose.get("tvec", [])).reshape(3)
            if rvec.size != 3 or tvec.size != 3:
                return None

            T_avp = _pose_to_matrix(rvec, tvec)
            T_rs = _pose_to_matrix(rs_pose[0], rs_pose[1])
            transform = T_avp @ np.linalg.inv(T_rs)
            return transform
        except Exception as exc:
            _safe_print(f"[RealSense] coord transform failed: {exc}")
            return None

    def _apply_transform(self, disparity: np.ndarray, transform) -> np.ndarray:
        if transform is None:
            return disparity
        try:
            tx, ty, _ = transform[:3, 3]
            shift_x = float(tx) * 200.0  # empirical scale to pixels
            shift_y = float(ty) * 200.0
            M = np.float32([[1, 0, shift_x], [0, 1, -shift_y]])
            warped = cv.warpAffine(
                disparity,
                M,
                (disparity.shape[1], disparity.shape[0]),
                flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_REFLECT
            )
            return warped
        except Exception as exc:
            _safe_print(f"[RealSense] transform application failed: {exc}")
            return disparity


# ----------------------------------------------------------------------
def _pose_to_matrix(rvec, tvec) -> np.ndarray:
    R, _ = cv.Rodrigues(np.asarray(rvec, dtype=np.float32))
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float32).reshape(3)
    return T


def _board_id_to_corners_m(marker_id: int):
    rows, cols = 3, 4
    marker_size = 0.030
    separation = 0.010
    if marker_id < 0 or marker_id >= rows * cols:
        return None
    row, col = divmod(marker_id, cols)
    x0 = col * (marker_size + separation)
    y0 = row * (marker_size + separation)
    return np.array(
        [
            [x0, marker_size * 0 + y0, 0],
            [x0 + marker_size, y0, 0],
            [x0 + marker_size, y0 + marker_size, 0],
            [x0, y0 + marker_size, 0],
        ],
        dtype=np.float32,
    )


def _solve_board_pose(corners, ids, K, dist):
    if ids is None or len(ids) == 0:
        return None
    obj_pts, img_pts = [], []
    for i, c in zip(ids.flatten().tolist(), corners):
        obj = _board_id_to_corners_m(i)
        if obj is None:
            continue
        pts = np.asarray(c, dtype=np.float32).reshape(-1, 2)
        obj_pts.append(obj)
        img_pts.append(pts)
    if not obj_pts:
        return None
    obj_pts = np.concatenate(obj_pts, axis=0)
    img_pts = np.concatenate(img_pts, axis=0)
    ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K, dist, flags=cv.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K, dist)
        if not ok:
            return None
    return rvec.reshape(3, 1), tvec.reshape(3, 1)
