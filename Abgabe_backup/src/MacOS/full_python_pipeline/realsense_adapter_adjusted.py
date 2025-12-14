#!/usr/bin/env python3
"""RealSense → AVP aligned disparity via 3D projection and Z‑buffering."""

from __future__ import annotations

import json
import os
import time
import threading
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2 as cv
import requests

try:  # optional hardware dependency
    import pyrealsense2 as rs  # type: ignore
except Exception:
    rs = None


def _safe_print(msg: str):
    try:
        print(msg)
    except Exception:
        pass


def _load_extrinsics(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        R = np.array(data.get("R", []), dtype=float).reshape(3, 3)
        t = np.array(data.get("t", []), dtype=float).reshape(3)
        if R.shape == (3, 3) and t.shape == (3,):
            return R, t
    except Exception as exc:
        _safe_print(f"[RS-Adj] No extrinsics at {path} or failed to load: {exc}")
    return None


def _pose_to_matrix_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv.Rodrigues(rvec.astype(np.float32).reshape(3, 1))
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.astype(float).reshape(3)
    return T


class RealSenseToAVPAligner:
    def __init__(self, main_api_base: Optional[str] = None, extrinsics_path: Optional[str] = None):
        self.main_api_base = main_api_base or os.environ.get("MAIN_API_BASE") or "http://localhost:5000"
        self.extrinsics_path = extrinsics_path or os.path.join(os.path.dirname(__file__), "extrinsics_avp_from_rs_color.json")

        self.pipeline = None
        self.align = None
        self.profile = None
        self.available = False
        self.lock = threading.Lock()

        # RealSense color intrinsics
        self.K_rs_c: Optional[np.ndarray] = None
        self.dist_rs_c: Optional[np.ndarray] = None

        # AVP intrinsics cache
        self.K_avp: Optional[np.ndarray] = None
        self.dist_avp: Optional[np.ndarray] = None

        # Extrinsics RS-color → AVP
        self.R_avp_rs_c: Optional[np.ndarray] = None
        self.t_avp_rs_c: Optional[np.ndarray] = None

        if rs is None:
            _safe_print("[RS-Adj] pyrealsense2 not available; running in no‑camera mode")
            return

        try:
            self._start_rs()
            # Try to load static extrinsics
            ex = _load_extrinsics(self.extrinsics_path)
            if ex:
                self.R_avp_rs_c, self.t_avp_rs_c = ex
            self.available = True
        except Exception as exc:
            _safe_print(f"[RS-Adj] Failed to start pipeline: {exc}")
            self.stop()

    # ----------------------------- RS init -----------------------------
    def _start_rs(self):
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
        self.K_rs_c = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=float)
        coeffs = intr.coeffs[:] if intr.coeffs else [0, 0, 0, 0, 0]
        self.dist_rs_c = np.array(coeffs[:5], dtype=float).reshape(-1, 1)
        _safe_print("[RS-Adj] RealSense started (depth aligned to color)")

    def stop(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None
        self.align = None
        self.available = False

    # ---------------------------- API fetch ----------------------------
    def _fetch_avp_intrinsics(self):
        if not self.main_api_base:
            return
        try:
            r = requests.get(f"{self.main_api_base}/intrinsics", timeout=0.8)
            if r.ok:
                j = r.json()
                self.K_avp = np.array(j.get("K", []), dtype=float)
                self.dist_avp = np.array(j.get("dist", [0, 0, 0, 0, 0]), dtype=float).reshape(-1, 1)
        except Exception as exc:
            _safe_print(f"[RS-Adj] intrinsics fetch failed: {exc}")

    def _fetch_poses_for_extrinsics(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Derive T_avp<-rs_c from simultaneous poses on a common board if available."""
        try:
            r = requests.get(f"{self.main_api_base}/pose", timeout=0.6)
            if not r.ok:
                return None
            avp = r.json()
            rvec_avp = np.array(avp.get("rvec", []), dtype=float).reshape(3, 1)
            tvec_avp = np.array(avp.get("tvec", []), dtype=float).reshape(3)
            if rvec_avp.size != 3 or tvec_avp.size != 3:
                return None
            T_avp_obj = _pose_to_matrix_rvec_tvec(rvec_avp, tvec_avp)
        except Exception:
            return None
        # Try to detect pose in RS color right now
        rs_pose = self._detect_pattern_latest()
        if rs_pose is None:
            return None
        T_rs_obj = _pose_to_matrix_rvec_tvec(rs_pose[0], rs_pose[1])
        T_avp_rs = T_avp_obj @ np.linalg.inv(T_rs_obj)
        R = T_avp_rs[:3, :3]
        t = T_avp_rs[:3, 3]
        return R, t

    # ------------------------- Frame processing ------------------------
    def _get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        if self.pipeline is None or self.align is None:
            return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1500)
            frames = self.align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                return None
            depth_units = float(depth_frame.get_units())  # meters per unit
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_units
            color = np.asanyarray(color_frame.get_data())
            return color, depth, depth_units
        except Exception as exc:
            _safe_print(f"[RS-Adj] get_frames failed: {exc}")
            return None

    def _detect_pattern_latest(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.K_rs_c is None or self.dist_rs_c is None:
            return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1)
            frames = self.align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            img = np.asanyarray(color_frame.get_data())
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            if not hasattr(cv, "aruco"):
                return None
            aruco = cv.aruco
            try:
                dct = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            except Exception:
                dct = aruco.Dictionary_get(aruco.DICT_4X4_50)
            try:
                params = aruco.DetectorParameters()
            except Exception:
                params = aruco.DetectorParameters_create()
            if hasattr(aruco, "ArucoDetector"):
                det = aruco.ArucoDetector(dct, params)
                corners, ids, _ = det.detectMarkers(gray)
            else:
                corners, ids, _ = aruco.detectMarkers(gray, dct, parameters=params)
            if ids is None or len(ids) == 0:
                return None
            # Simple 3x4 board mapping (match pipeline)
            def board_id_to_corners(i: int):
                rows, cols = 3, 4
                marker_size, sep = 0.030, 0.010
                if i < 0 or i >= rows * cols:
                    return None
                row, col = divmod(i, cols)
                x0 = col * (marker_size + sep)
                y0 = row * (marker_size + sep)
                return np.array([[x0, y0, 0], [x0+marker_size, y0, 0], [x0+marker_size, y0+marker_size, 0], [x0, y0+marker_size, 0]], dtype=np.float32)
            obj_pts, img_pts = [], []
            for i, c in zip(ids.flatten().tolist(), corners):
                obj = board_id_to_corners(i)
                if obj is None:
                    continue
                pts = np.asarray(c, dtype=np.float32).reshape(-1, 2)
                obj_pts.append(obj)
                img_pts.append(pts)
            if not obj_pts:
                return None
            obj_pts = np.concatenate(obj_pts, axis=0)
            img_pts = np.concatenate(img_pts, axis=0)
            ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, self.K_rs_c, self.dist_rs_c, flags=cv.SOLVEPNP_IPPE)
            if not ok:
                ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, self.K_rs_c, self.dist_rs_c)
                if not ok:
                    return None
            return rvec.reshape(3, 1), tvec.reshape(3)
        except Exception:
            return None

    # ------------------------- Reprojection warp -----------------------
    @staticmethod
    def _zbuffer_scatter(height: int, width: int, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Scatter (u,v,z) into an image with Z‑buffering (keep nearest)."""
        depth_out = np.full((height, width), np.inf, dtype=np.float32)
        # Flatten to 1D indices
        idx = v.astype(np.int64) * width + u.astype(np.int64)
        # Sort by ascending depth, keep first per pixel
        order = np.argsort(z)
        idx_sorted = idx[order]
        first = np.unique(idx_sorted, return_index=True)[1]
        chosen = order[first]
        depth_out_flat = depth_out.ravel()
        depth_out_flat[idx[chosen]] = z[chosen]
        depth_out = depth_out_flat.reshape(height, width)
        depth_out[np.isinf(depth_out)] = 0.0
        return depth_out

    def _warp_depth_to_avp(self, depth_rs_c: np.ndarray, K_rs_c: np.ndarray, K_avp: np.ndarray, R: np.ndarray, t: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
        H, W = depth_rs_c.shape[:2]
        out_h, out_w = out_hw
        # Build pixel grid
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        Z = depth_rs_c.astype(np.float32)
        valid = Z > 0
        uu = uu[valid]
        vv = vv[valid]
        Z = Z[valid]
        # Back‑project to RS color camera 3D
        fx, fy, cx, cy = K_rs_c[0, 0], K_rs_c[1, 1], K_rs_c[0, 2], K_rs_c[1, 2]
        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy
        pts = np.stack([X, Y, Z], axis=0)  # 3xN
        # Transform to AVP camera
        pts_avp = (R @ pts) + t.reshape(3, 1)
        Xa, Ya, Za = pts_avp[0], pts_avp[1], pts_avp[2]
        ok = Za > 1e-6
        Xa, Ya, Za = Xa[ok], Ya[ok], Za[ok]
        # Project to AVP pixels
        fx2, fy2, cx2, cy2 = K_avp[0, 0], K_avp[1, 1], K_avp[0, 2], K_avp[1, 2]
        u2 = fx2 * (Xa / Za) + cx2
        v2 = fy2 * (Ya / Za) + cy2
        u2i = np.round(u2).astype(np.int32)
        v2i = np.round(v2).astype(np.int32)
        inb = (u2i >= 0) & (u2i < out_w) & (v2i >= 0) & (v2i < out_h)
        if not np.any(inb):
            return np.zeros((out_h, out_w), dtype=np.float32)
        u2i = u2i[inb]
        v2i = v2i[inb]
        Za = Za[inb]
        # Z‑buffer splat
        depth_avp = self._zbuffer_scatter(out_h, out_w, u2i, v2i, Za)
        return depth_avp

    # -------------------- Reverse Transform (AVP -> RS) --------------------
    def transform_mask_avp_to_rs(self, mask_avp: np.ndarray, depth_avp: Optional[np.ndarray] = None, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """
        Transform mask from AVP view to RealSense view with proper occlusion handling.

        Args:
            mask_avp: Binary mask in AVP view (HxW uint8)
            depth_avp: Optional depth map in AVP view (HxW float32, in meters)
                       If provided, uses actual depth for accurate transformation
                       If None, uses constant depth assumption
            target_size: Optional (width, height) for RealSense output

        Returns:
            Binary mask in RealSense view or None
        """
        if self.R_avp_rs_c is None or self.t_avp_rs_c is None:
            _safe_print("[RS-Adj] Cannot transform mask: no extrinsics")
            return None

        if self.K_avp is None or self.K_rs_c is None:
            _safe_print("[RS-Adj] Cannot transform mask: missing intrinsics")
            return None

        try:
            # Compute inverse transformation (AVP -> RS)
            R_rs_avp = self.R_avp_rs_c.T  # Inverse rotation
            t_rs_avp = -R_rs_avp @ self.t_avp_rs_c  # Inverse translation

            H_avp, W_avp = mask_avp.shape[:2]
            H_rs, W_rs = target_size[::-1] if target_size else (480, 640)

            # Get mask pixels
            ys, xs = np.where(mask_avp > 0)
            if len(xs) == 0:
                return np.zeros((H_rs, W_rs), dtype=np.uint8)

            # Get depth values for mask pixels
            if depth_avp is not None and depth_avp.shape[:2] == mask_avp.shape[:2]:
                # Use actual depth from AVP (PROPER OCCLUSION HANDLING)
                Z_avp = depth_avp[ys, xs].astype(np.float32)

                # Filter out invalid depth values
                valid_depth = Z_avp > 1e-6
                if not np.any(valid_depth):
                    _safe_print("[RS-Adj] No valid depth values in mask region, using constant depth")
                    Z_avp = np.full(len(xs), 0.5, dtype=np.float32)
                else:
                    xs, ys = xs[valid_depth], ys[valid_depth]
                    Z_avp = Z_avp[valid_depth]
                    _safe_print(f"[RS-Adj] Using actual depth: {Z_avp.min():.3f}m to {Z_avp.max():.3f}m")
            else:
                # Fallback: constant depth assumption (NO OCCLUSION HANDLING)
                Z_avp = np.full(len(xs), 0.5, dtype=np.float32)
                _safe_print("[RS-Adj] Warning: Using constant depth assumption (may not handle occlusion correctly)")

            # Back-project AVP mask pixels to 3D using actual depths
            fx_avp, fy_avp = self.K_avp[0, 0], self.K_avp[1, 1]
            cx_avp, cy_avp = self.K_avp[0, 2], self.K_avp[1, 2]

            X_avp = (xs - cx_avp) * Z_avp / fx_avp
            Y_avp = (ys - cy_avp) * Z_avp / fy_avp

            pts_avp = np.stack([X_avp, Y_avp, Z_avp], axis=0)  # 3xN

            # Transform to RS camera
            pts_rs = (R_rs_avp @ pts_avp) + t_rs_avp.reshape(3, 1)
            X_rs, Y_rs, Z_rs = pts_rs[0], pts_rs[1], pts_rs[2]

            # Filter points behind camera
            valid = Z_rs > 1e-6
            X_rs, Y_rs, Z_rs = X_rs[valid], Y_rs[valid], Z_rs[valid]

            if len(X_rs) == 0:
                _safe_print("[RS-Adj] No valid points after transformation")
                return np.zeros((H_rs, W_rs), dtype=np.uint8)

            # Project to RS pixels
            fx_rs, fy_rs = self.K_rs_c[0, 0], self.K_rs_c[1, 1]
            cx_rs, cy_rs = self.K_rs_c[0, 2], self.K_rs_c[1, 2]

            u_rs = fx_rs * (X_rs / Z_rs) + cx_rs
            v_rs = fy_rs * (Y_rs / Z_rs) + cy_rs

            u_rs = np.round(u_rs).astype(np.int32)
            v_rs = np.round(v_rs).astype(np.int32)

            # Filter bounds
            inbounds = (u_rs >= 0) & (u_rs < W_rs) & (v_rs >= 0) & (v_rs < H_rs)
            u_rs = u_rs[inbounds]
            v_rs = v_rs[inbounds]
            Z_rs = Z_rs[inbounds]

            if len(u_rs) == 0:
                _safe_print("[RS-Adj] No points within RS view bounds")
                return np.zeros((H_rs, W_rs), dtype=np.uint8)

            # Create output mask with Z-buffering for proper occlusion handling
            # (similar to forward approach)
            if depth_avp is not None:
                # Use Z-buffer: keep nearest point per pixel
                mask_rs = self._zbuffer_scatter(H_rs, W_rs, u_rs, v_rs, Z_rs)
                # Convert depth image to binary mask
                mask_rs = (mask_rs > 0).astype(np.uint8) * 255
            else:
                # Fallback: simple scatter (no occlusion handling)
                mask_rs = np.zeros((H_rs, W_rs), dtype=np.uint8)
                mask_rs[v_rs, u_rs] = 255

            # Dilate to fill small gaps (conservative dilation)
            kernel = np.ones((3, 3), np.uint8)
            mask_rs = cv.dilate(mask_rs, kernel, iterations=1)

            return mask_rs

        except Exception as exc:
            _safe_print(f"[RS-Adj] Mask transformation failed: {exc}")
            import traceback
            traceback.print_back()
            return None

    def transform_pose_rs_to_avp(self, T_rs: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform 4x4 pose matrix from RealSense view to AVP view.

        Args:
            T_rs: 4x4 transformation matrix in RealSense camera frame

        Returns:
            4x4 transformation matrix in AVP camera frame or None
        """
        if self.R_avp_rs_c is None or self.t_avp_rs_c is None:
            _safe_print("[RS-Adj] Cannot transform pose: no extrinsics")
            return None

        try:
            # Build T_avp_rs (AVP from RS)
            T_avp_rs = np.eye(4, dtype=float)
            T_avp_rs[:3, :3] = self.R_avp_rs_c
            T_avp_rs[:3, 3] = self.t_avp_rs_c

            # Transform: T_avp_obj = T_avp_rs @ T_rs_obj
            T_avp = T_avp_rs @ T_rs

            return T_avp

        except Exception as exc:
            _safe_print(f"[RS-Adj] Pose transformation failed: {exc}")
            return None

    # ----------------------------- Public ------------------------------
    def capture_and_align(self, target_size: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        if self.K_avp is None:
            self._fetch_avp_intrinsics()
            if self.K_avp is None:
                return None
        frames = self._get_frames()
        if frames is None:
            return None
        color, depth_m, _ = frames
        Hc, Wc = color.shape[:2]
        out_h = target_size[1] if target_size else Hc
        out_w = target_size[0] if target_size else Wc

        # Acquire or derive extrinsics
        if self.R_avp_rs_c is None or self.t_avp_rs_c is None:
            ex = self._fetch_poses_for_extrinsics()
            if ex:
                self.R_avp_rs_c, self.t_avp_rs_c = ex
            else:
                return {
                    "color": color,
                    "depth_rs_color": depth_m,
                    "aligned_depth": None,
                    "aligned_disparity": None,
                    "reason": "No extrinsics available"
                }

        depth_avp = self._warp_depth_to_avp(
            depth_m,
            self.K_rs_c,
            self.K_avp,
            self.R_avp_rs_c,
            self.t_avp_rs_c,
            (out_w, out_h),
        )
        # Pseudo‑disparity for visualization
        disp = depth_avp.copy()
        if disp.size:
            disp = cv.normalize(np.where(disp > 0, 1.0 / disp, 0.0), None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        return {
            "color": color,
            "depth_rs_color": depth_m,
            "aligned_depth": depth_avp,
            "aligned_disparity": disp,
            "timestamp": time.time(),
        }

    def capture_rs_native(self) -> Optional[Dict[str, Any]]:
        """
        Capture RealSense data in native RS view (for reverse approach).

        Returns:
            Dict with 'color', 'depth', 'disparity', 'timestamp'
        """
        if not self.available:
            return None

        frames = self._get_frames()
        if frames is None:
            return None

        color, depth_m, _ = frames

        # Create disparity for visualization
        disp = depth_m.copy()
        if disp.size:
            disp = cv.normalize(np.where(disp > 0, 1.0 / disp, 0.0), None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        return {
            "color": color,
            "depth": depth_m,
            "disparity": disp,
            "timestamp": time.time(),
        }


def encode_img_to_base64(img: np.ndarray) -> str:
    if img.dtype != np.uint8:
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    else:
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    import base64, io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

