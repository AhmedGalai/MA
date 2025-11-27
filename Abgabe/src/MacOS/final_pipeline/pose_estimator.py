"""
6D Pose Estimator
Estimates object pose using depth map + mask in RealSense view
"""

import numpy as np
import cv2 as cv
from typing import Optional, Dict, Tuple

from .config import POSE_ESTIMATION_CONFIG


class PoseEstimator:
    """
    6D pose estimator using depth + mask
    Combines depth information with object mask for robust pose estimation
    """

    def __init__(self):
        """Initialize pose estimator"""
        self.min_points = POSE_ESTIMATION_CONFIG["min_points"]
        self.ransac_threshold = POSE_ESTIMATION_CONFIG["ransac_threshold"]
        self.ransac_iterations = POSE_ESTIMATION_CONFIG["ransac_iterations"]
        self.confidence_threshold = POSE_ESTIMATION_CONFIG["confidence_threshold"]

    def estimate_pose_from_depth_and_mask(self, depth_map: np.ndarray, mask: np.ndarray,
                                          K: np.ndarray, dist_coeffs: np.ndarray,
                                          object_model: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Estimate 6D pose from depth map and mask

        Args:
            depth_map: Depth map (H, W) in millimeters
            mask: Binary mask (H, W) indicating object pixels
            K: Camera intrinsics matrix (3, 3)
            dist_coeffs: Distortion coefficients
            object_model: Optional 3D object model points for PnP

        Returns:
            Dictionary with pose estimate and confidence, or None
        """
        # Get 3D points from depth + mask
        points_3d = self._extract_3d_points_from_depth(depth_map, mask, K)

        if points_3d is None or len(points_3d) < self.min_points:
            print(f"[PoseEst] Insufficient points: {len(points_3d) if points_3d is not None else 0}")
            return None

        # If object model is provided, use PnP
        if object_model is not None:
            return self._estimate_pose_pnp(points_3d, object_model, K, dist_coeffs)

        # Otherwise, use centroid-based pose
        return self._estimate_pose_from_pointcloud(points_3d, mask)

    def _extract_3d_points_from_depth(self, depth_map: np.ndarray, mask: np.ndarray,
                                      K: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 3D points from depth map using mask

        Args:
            depth_map: Depth in millimeters (H, W)
            mask: Binary mask (H, W)
            K: Camera intrinsics (3, 3)

        Returns:
            3D points (N, 3) in camera frame, or None
        """
        h, w = depth_map.shape

        # Get valid pixels (mask > 0 and depth > 0)
        valid_mask = (mask > 0) & (depth_map > 0)

        if not valid_mask.any():
            return None

        # Get pixel coordinates
        v_coords, u_coords = np.where(valid_mask)
        depths = depth_map[valid_mask] / 1000.0  # Convert to meters

        # Deproject to 3D
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x = (u_coords - cx) * depths / fx
        y = (v_coords - cy) * depths / fy
        z = depths

        points_3d = np.stack([x, y, z], axis=1)

        return points_3d

    def _estimate_pose_from_pointcloud(self, points_3d: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Estimate pose from point cloud using centroid and PCA

        Args:
            points_3d: 3D points (N, 3)
            mask: Binary mask

        Returns:
            Dictionary with pose estimate
        """
        # Compute centroid (position)
        centroid = np.mean(points_3d, axis=0)

        # Compute principal axes using PCA (orientation)
        centered_points = points_3d - centroid
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1

        # Convert rotation matrix to rotation vector
        rvec, _ = cv.Rodrigues(eigenvectors)

        # Compute confidence based on point distribution
        spread = np.std(centered_points, axis=0)
        confidence = 1.0 / (1.0 + np.mean(spread))  # Higher spread = lower confidence

        return {
            "rvec": rvec.flatten(),
            "tvec": centroid,
            "confidence": float(confidence),
            "num_points": len(points_3d),
            "method": "pca"
        }

    def _estimate_pose_pnp(self, points_3d: np.ndarray, object_model: np.ndarray,
                          K: np.ndarray, dist_coeffs: np.ndarray) -> Optional[Dict]:
        """
        Estimate pose using PnP with object model

        Args:
            points_3d: Observed 3D points (N, 3)
            object_model: 3D object model points (M, 3)
            K: Camera intrinsics
            dist_coeffs: Distortion coefficients

        Returns:
            Dictionary with pose estimate, or None
        """
        # Match observed points to model points (simplified - use nearest neighbor)
        # In practice, you'd use feature matching or ICP

        if len(object_model) < self.min_points:
            print("[PoseEst] Object model too small")
            return None

        # For now, use centroid-based alignment
        # TODO: Implement proper point correspondence

        # Compute centroids
        model_centroid = np.mean(object_model, axis=0)
        obs_centroid = np.mean(points_3d, axis=0)

        # Translation is difference in centroids
        tvec = obs_centroid - model_centroid

        # Estimate rotation using Kabsch algorithm
        centered_model = object_model - model_centroid
        centered_obs = points_3d[:len(object_model)] - obs_centroid

        H = centered_model.T @ centered_obs
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure right-handed
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        rvec, _ = cv.Rodrigues(R)

        # Compute confidence (based on residual error)
        transformed_model = (R @ centered_model.T).T + obs_centroid
        residuals = np.linalg.norm(points_3d[:len(object_model)] - transformed_model, axis=1)
        mean_error = np.mean(residuals)
        confidence = 1.0 / (1.0 + mean_error)

        return {
            "rvec": rvec.flatten(),
            "tvec": tvec,
            "confidence": float(confidence),
            "num_points": len(points_3d),
            "mean_error": float(mean_error),
            "method": "pnp"
        }

    def refine_pose_with_icp(self, pose: Dict, source_points: np.ndarray,
                            target_points: np.ndarray, max_iterations: int = 50) -> Dict:
        """
        Refine pose estimate using Iterative Closest Point (ICP)

        Args:
            pose: Initial pose estimate
            source_points: Source point cloud (N, 3)
            target_points: Target point cloud (M, 3)
            max_iterations: Maximum ICP iterations

        Returns:
            Refined pose
        """
        # TODO: Implement ICP refinement
        # For now, just return original pose
        return pose

    def compute_pose_confidence(self, depth_map: np.ndarray, mask: np.ndarray,
                                pose: Dict, K: np.ndarray) -> float:
        """
        Compute confidence score for pose estimate

        Args:
            depth_map: Depth map
            mask: Object mask
            pose: Pose estimate
            K: Camera intrinsics

        Returns:
            Confidence score (0-1)
        """
        # Factors affecting confidence:
        # 1. Number of points
        # 2. Depth consistency
        # 3. Mask coverage

        num_points = pose.get("num_points", 0)
        if num_points < self.min_points:
            return 0.0

        # Point count score
        point_score = min(num_points / 100.0, 1.0)

        # Existing confidence from pose estimation
        method_confidence = pose.get("confidence", 0.5)

        # Combined confidence
        confidence = (point_score + method_confidence) / 2.0

        return float(np.clip(confidence, 0.0, 1.0))


if __name__ == "__main__":
    print("Testing Pose Estimator...")

    estimator = PoseEstimator()

    # Create synthetic depth and mask
    depth_map = np.ones((480, 640), dtype=np.uint16) * 1000  # 1m depth
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[200:300, 300:400] = 255  # Object region

    # Camera intrinsics
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)

    # Estimate pose
    result = estimator.estimate_pose_from_depth_and_mask(depth_map, mask, K, dist)

    if result:
        print(f"Pose estimated successfully!")
        print(f"  Position: {result['tvec']}")
        print(f"  Rotation: {result['rvec']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Points: {result['num_points']}")
    else:
        print("Pose estimation failed")

    print("Test complete!")
