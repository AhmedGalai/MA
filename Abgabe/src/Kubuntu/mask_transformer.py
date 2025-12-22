"""
Transform binary masks from AVP view to RealSense view using camera parameters and extrinsics.

This module provides functionality to project masks between different camera views using
camera intrinsics and transformation matrices.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


def transform_mask_avp_to_rs(
    mask_avp: np.ndarray,
    K_avp: np.ndarray,
    K_rs: np.ndarray,
    T_rs_avp: np.ndarray,
    depth_avp: Optional[np.ndarray] = None,
    target_size: Tuple[int, int] = (640, 480)
) -> np.ndarray:
    """
    Transform a binary mask from AVP view to RealSense view.

    Transforms mask pixels from AVP camera frame to RealSense camera frame using
    camera intrinsics and the transformation matrix between the two cameras.

    Parameters
    ----------
    mask_avp : np.ndarray
        Binary mask from AVP camera (HxW uint8, values 0 or 255).
    K_avp : np.ndarray
        AVP camera intrinsic matrix (3x3).
    K_rs : np.ndarray
        RealSense camera intrinsic matrix (3x3).
    T_rs_avp : np.ndarray
        4x4 transformation matrix from AVP frame to RealSense frame.
    depth_avp : np.ndarray, optional
        Depth map for AVP camera (HxW float32 in meters).
        If None, a constant depth of 1.0m is used for all pixels.
    target_size : Tuple[int, int], optional
        Target output size as (width, height). Default is (640, 480).

    Returns
    -------
    np.ndarray
        Binary mask in RealSense view (height x width uint8, values 0 or 255).

    Raises
    ------
    ValueError
        If input arrays have invalid shapes or types.
    TypeError
        If inputs are not numpy arrays.
    """
    # Input validation
    if not isinstance(mask_avp, np.ndarray):
        raise TypeError("mask_avp must be a numpy array")
    if not isinstance(K_avp, np.ndarray):
        raise TypeError("K_avp must be a numpy array")
    if not isinstance(K_rs, np.ndarray):
        raise TypeError("K_rs must be a numpy array")
    if not isinstance(T_rs_avp, np.ndarray):
        raise TypeError("T_rs_avp must be a numpy array")

    if mask_avp.ndim != 2:
        raise ValueError(f"mask_avp must be 2D, got shape {mask_avp.shape}")
    if K_avp.shape != (3, 3):
        raise ValueError(f"K_avp must be 3x3, got shape {K_avp.shape}")
    if K_rs.shape != (3, 3):
        raise ValueError(f"K_rs must be 3x3, got shape {K_rs.shape}")
    if T_rs_avp.shape != (4, 4):
        raise ValueError(f"T_rs_avp must be 4x4, got shape {T_rs_avp.shape}")

    if depth_avp is not None:
        if not isinstance(depth_avp, np.ndarray):
            raise TypeError("depth_avp must be a numpy array or None")
        if depth_avp.shape != mask_avp.shape:
            raise ValueError(
                f"depth_avp shape {depth_avp.shape} must match mask_avp shape {mask_avp.shape}"
            )

    h_avp, w_avp = mask_avp.shape
    target_w, target_h = target_size

    # Find all non-zero pixels in mask_avp
    # Returns (row_indices, col_indices) where mask is non-zero
    y_avp, x_avp = np.where(mask_avp != 0)

    if len(y_avp) == 0:
        # Empty mask, return empty mask in target size
        return np.zeros((target_h, target_w), dtype=np.uint8)

    # Get depth values for each pixel
    if depth_avp is not None:
        depths = depth_avp[y_avp, x_avp].astype(np.float32)
    else:
        depths = np.ones(len(y_avp), dtype=np.float32)

    # Back-project to 3D using AVP camera intrinsics
    # pts_2d_homogeneous = [x, y, 1]
    # pts_3d = inv(K_avp) @ pts_2d_homogeneous * depth
    K_avp_inv = np.linalg.inv(K_avp)

    # Create homogeneous pixel coordinates [x, y, 1]
    pts_2d_homogeneous = np.stack([x_avp, y_avp, np.ones_like(x_avp)], axis=0).astype(np.float32)

    # Back-project: pts_3d = inv(K) @ pts_2d * depth
    # Shape: (3, N)
    pts_3d_avp = K_avp_inv @ pts_2d_homogeneous * depths[np.newaxis, :]

    # Convert to homogeneous coordinates [X, Y, Z, 1]
    # Shape: (4, N)
    pts_3d_homogeneous_avp = np.vstack([pts_3d_avp, np.ones(pts_3d_avp.shape[1], dtype=np.float32)])

    # Transform to RealSense frame
    # pts_3d_rs = T_rs_avp @ pts_3d_homogeneous_avp
    # Shape: (4, N)
    pts_3d_homogeneous_rs = T_rs_avp @ pts_3d_homogeneous_avp

    # Extract 3D coordinates in RealSense frame
    pts_3d_rs = pts_3d_homogeneous_rs[:3, :]

    # Project to RealSense image plane
    # pts_2d_rs = K_rs @ pts_3d_rs / pts_3d_rs[2]
    z_rs = pts_3d_rs[2, :]

    # Handle points behind camera (z <= 0)
    valid_depth = z_rs > 0

    # Initialize with invalid values
    pts_2d_rs = np.full((2, len(y_avp)), -1.0, dtype=np.float32)

    if np.any(valid_depth):
        # Project only valid points
        pts_2d_projected = K_rs @ (pts_3d_rs[:, valid_depth] / z_rs[valid_depth])
        pts_2d_rs[:, valid_depth] = pts_2d_projected[:2, :]

    # Clip projected coordinates to image bounds
    x_rs = pts_2d_rs[0, :]
    y_rs = pts_2d_rs[1, :]

    # Filter points within valid bounds and with valid depth
    in_bounds = (
        (x_rs >= 0) & (x_rs < target_w) &
        (y_rs >= 0) & (y_rs < target_h) &
        valid_depth
    )

    # Create output mask
    mask_rs = np.zeros((target_h, target_w), dtype=np.uint8)

    if np.any(in_bounds):
        # Round to nearest pixel
        x_rs_int = np.round(x_rs[in_bounds]).astype(np.int32)
        y_rs_int = np.round(y_rs[in_bounds]).astype(np.int32)

        # Set valid pixels to 255
        mask_rs[y_rs_int, x_rs_int] = 255

    # Apply morphological dilation to fill small gaps
    # 3x3 kernel, 1 iteration
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_rs = cv2.dilate(mask_rs, kernel, iterations=1)

    return mask_rs
