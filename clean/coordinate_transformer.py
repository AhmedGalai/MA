#!/usr/bin/env python3
"""
Coordinate transformation between AVP and RealSense camera frames.
"""

import numpy as np
import cv2 as cv
import json
from typing import Optional, Tuple
import config


class CoordinateTransformer:
    """Handles coordinate transformations between AVP and RealSense frames."""

    def __init__(self):
        """Initialize coordinate transformer."""
        self.R_avp_from_rs = None  # Rotation from RS to AVP
        self.t_avp_from_rs = None  # Translation from RS to AVP
        self.T_avp_from_rs = None  # 4x4 transformation matrix from RS to AVP
        self.T_rs_from_avp = None  # 4x4 transformation matrix from AVP to RS

        # Try to load extrinsics from file
        self.load_extrinsics(config.EXTRINSICS_FILE)

    def load_extrinsics(self, filepath: str) -> bool:
        """
        Load extrinsic calibration from JSON file.

        Args:
            filepath: Path to extrinsics JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.R_avp_from_rs = np.array(data['R'], dtype=np.float64).reshape(3, 3)
            self.t_avp_from_rs = np.array(data['t'], dtype=np.float64).reshape(3)

            # Build transformation matrices
            self.T_avp_from_rs = np.eye(4, dtype=np.float64)
            self.T_avp_from_rs[:3, :3] = self.R_avp_from_rs
            self.T_avp_from_rs[:3, 3] = self.t_avp_from_rs

            # Compute inverse transformation
            self.T_rs_from_avp = np.linalg.inv(self.T_avp_from_rs)

            print(f"[Transformer] Loaded extrinsics from {filepath}")
            return True

        except Exception as e:
            print(f"[Transformer] Failed to load extrinsics: {e}")
            return False

    def save_extrinsics(self, filepath: str) -> bool:
        """
        Save extrinsic calibration to JSON file.

        Args:
            filepath: Path to save extrinsics JSON file

        Returns:
            True if successful, False otherwise
        """
        if self.R_avp_from_rs is None or self.t_avp_from_rs is None:
            print("[Transformer] No extrinsics to save")
            return False

        try:
            data = {
                'R': self.R_avp_from_rs.tolist(),
                't': self.t_avp_from_rs.tolist()
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"[Transformer] Saved extrinsics to {filepath}")
            return True

        except Exception as e:
            print(f"[Transformer] Failed to save extrinsics: {e}")
            return False

    def calibrate_from_poses(
        self,
        avp_rvec: np.ndarray,
        avp_tvec: np.ndarray,
        rs_rvec: np.ndarray,
        rs_tvec: np.ndarray
    ) -> bool:
        """
        Calibrate transformation from simultaneous board detections.

        Both cameras should see the same ArUco board simultaneously.
        This computes the transformation: T_avp_from_rs

        Args:
            avp_rvec: AVP board pose rotation vector (3,)
            avp_tvec: AVP board pose translation vector (3,)
            rs_rvec: RS board pose rotation vector (3,)
            rs_tvec: RS board pose translation vector (3,)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert poses to transformation matrices
            R_avp, _ = cv.Rodrigues(avp_rvec.reshape(3, 1))
            T_avp_board = np.eye(4, dtype=np.float64)
            T_avp_board[:3, :3] = R_avp
            T_avp_board[:3, 3] = avp_tvec.reshape(3)

            R_rs, _ = cv.Rodrigues(rs_rvec.reshape(3, 1))
            T_rs_board = np.eye(4, dtype=np.float64)
            T_rs_board[:3, :3] = R_rs
            T_rs_board[:3, 3] = rs_tvec.reshape(3)

            # Compute transformation: T_avp_from_rs = T_avp_board * inv(T_rs_board)
            self.T_avp_from_rs = T_avp_board @ np.linalg.inv(T_rs_board)
            self.T_rs_from_avp = np.linalg.inv(self.T_avp_from_rs)

            # Extract R and t
            self.R_avp_from_rs = self.T_avp_from_rs[:3, :3]
            self.t_avp_from_rs = self.T_avp_from_rs[:3, 3]

            print("[Transformer] Calibration successful")
            return True

        except Exception as e:
            print(f"[Transformer] Calibration failed: {e}")
            return False

    def transform_pose_rs_to_avp(self, T_rs: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform pose from RealSense frame to AVP frame.

        Args:
            T_rs: 4x4 transformation matrix in RS frame

        Returns:
            4x4 transformation matrix in AVP frame or None if not calibrated
        """
        if self.T_avp_from_rs is None:
            print("[Transformer] Not calibrated")
            return None

        T_avp = self.T_avp_from_rs @ T_rs
        return T_avp

    def transform_pose_avp_to_rs(self, T_avp: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform pose from AVP frame to RealSense frame.

        Args:
            T_avp: 4x4 transformation matrix in AVP frame

        Returns:
            4x4 transformation matrix in RS frame or None if not calibrated
        """
        if self.T_rs_from_avp is None:
            print("[Transformer] Not calibrated")
            return None

        T_rs = self.T_rs_from_avp @ T_avp
        return T_rs

    def transform_mask_avp_to_rs(
        self,
        mask_avp: np.ndarray,
        K_avp: np.ndarray,
        K_rs: np.ndarray,
        depth_avp: Optional[np.ndarray] = None,
        target_size: Tuple[int, int] = (640, 480)
    ) -> Optional[np.ndarray]:
        """
        Transform mask from AVP view to RealSense view.

        Uses depth information for accurate projection. If depth is not available,
        uses a constant depth approximation.

        Args:
            mask_avp: Binary mask in AVP view (H_avp, W_avp)
            K_avp: AVP camera intrinsic matrix (3, 3)
            K_rs: RealSense camera intrinsic matrix (3, 3)
            depth_avp: Depth map in AVP view (H_avp, W_avp) in meters (optional)
            target_size: Target size (width, height) for RS mask

        Returns:
            Binary mask in RS view (H_rs, W_rs) or None if transformation fails
        """
        if self.T_rs_from_avp is None:
            print("[Transformer] Not calibrated")
            return None

        try:
            h_avp, w_avp = mask_avp.shape
            w_rs, h_rs = target_size

            # Find mask pixels
            ys, xs = np.where(mask_avp > 0)
            if len(xs) == 0:
                print("[Transformer] Empty mask")
                return np.zeros((h_rs, w_rs), dtype=np.uint8)

            # Use depth if available, otherwise use constant depth assumption
            if depth_avp is not None:
                depths = depth_avp[ys, xs]
                # Filter out invalid depths
                valid = depths > 0
                xs, ys, depths = xs[valid], ys[valid], depths[valid]
            else:
                # Assume constant depth of 0.5 meters
                depths = np.full(len(xs), 0.5, dtype=np.float32)
                print("[Transformer] Using constant depth assumption (0.5m)")

            if len(xs) == 0:
                return np.zeros((h_rs, w_rs), dtype=np.uint8)

            # Unproject AVP pixels to 3D points in AVP frame
            fx_avp, fy_avp = K_avp[0, 0], K_avp[1, 1]
            cx_avp, cy_avp = K_avp[0, 2], K_avp[1, 2]

            X_avp = (xs - cx_avp) * depths / fx_avp
            Y_avp = (ys - cy_avp) * depths / fy_avp
            Z_avp = depths

            # Create homogeneous coordinates
            points_avp = np.vstack([X_avp, Y_avp, Z_avp, np.ones_like(X_avp)])

            # Transform to RS frame
            points_rs = self.T_rs_from_avp @ points_avp

            # Project to RS image plane
            fx_rs, fy_rs = K_rs[0, 0], K_rs[1, 1]
            cx_rs, cy_rs = K_rs[0, 2], K_rs[1, 2]

            X_rs, Y_rs, Z_rs = points_rs[0], points_rs[1], points_rs[2]

            # Filter points behind camera
            valid = Z_rs > 0.01
            X_rs, Y_rs, Z_rs = X_rs[valid], Y_rs[valid], Z_rs[valid]

            if len(X_rs) == 0:
                return np.zeros((h_rs, w_rs), dtype=np.uint8)

            # Project to image coordinates
            u_rs = (X_rs * fx_rs / Z_rs + cx_rs).astype(np.int32)
            v_rs = (Y_rs * fy_rs / Z_rs + cy_rs).astype(np.int32)

            # Filter points outside image bounds
            valid = (u_rs >= 0) & (u_rs < w_rs) & (v_rs >= 0) & (v_rs < h_rs)
            u_rs, v_rs = u_rs[valid], v_rs[valid]

            # Create mask in RS view
            mask_rs = np.zeros((h_rs, w_rs), dtype=np.uint8)
            mask_rs[v_rs, u_rs] = 255

            # Dilate to fill small gaps
            kernel = np.ones((3, 3), np.uint8)
            mask_rs = cv.dilate(mask_rs, kernel, iterations=2)

            print(f"[Transformer] Transformed {len(xs)} -> {len(u_rs)} mask pixels")
            return mask_rs

        except Exception as e:
            print(f"[Transformer] Mask transformation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def is_calibrated(self) -> bool:
        """Check if transformer is calibrated."""
        return self.T_avp_from_rs is not None
