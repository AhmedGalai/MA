"""
Coordinate Transformation Pipeline with Probabilistic Pose Correction
Handles transformations between AVP and RealSense coordinate frames
with Kalman filtering for smooth, accurate pose estimates
"""

import numpy as np
import cv2 as cv
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation

from .config import KALMAN_CONFIG


class KalmanPoseFilter:
    """
    Kalman filter for 6D pose (position + rotation)
    Smooths noisy pose measurements over time
    """

    def __init__(self):
        """Initialize Kalman filter for 6D pose"""
        # State: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        # Position (3) + Quaternion (4) + Linear velocity (3) + Angular velocity (3)
        self.state_dim = 13
        self.measurement_dim = 7  # Position (3) + Quaternion (4)

        # State vector
        self.x = np.zeros(self.state_dim)
        self.x[3] = 1.0  # Initialize quaternion to identity

        # State covariance
        self.P = np.eye(self.state_dim) * KALMAN_CONFIG["initial_uncertainty"]

        # Process noise
        self.Q = np.eye(self.state_dim) * KALMAN_CONFIG["process_noise"]

        # Measurement noise
        self.R = np.eye(self.measurement_dim) * KALMAN_CONFIG["measurement_noise"]

        # Time of last update
        self.last_update_time = None

    def predict(self, dt: float):
        """
        Prediction step

        Args:
            dt: Time step in seconds
        """
        # State transition matrix (constant velocity model)
        F = np.eye(self.state_dim)
        F[0:3, 7:10] = np.eye(3) * dt  # Position += velocity * dt
        # Rotation prediction is more complex, simplified here

        # Predict state
        self.x = F @ self.x

        # Normalize quaternion
        q = self.x[3:7]
        self.x[3:7] = q / np.linalg.norm(q)

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: np.ndarray):
        """
        Update step

        Args:
            measurement: [x, y, z, qw, qx, qy, qz]
        """
        # Measurement matrix (observe position and rotation only)
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[0:7, 0:7] = np.eye(7)

        # Innovation
        y = measurement - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Normalize quaternion
        q = self.x[3:7]
        self.x[3:7] = q / np.linalg.norm(q)

        # Update covariance
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current pose estimate

        Returns:
            (position [x, y, z], quaternion [qw, qx, qy, qz])
        """
        return self.x[0:3].copy(), self.x[3:7].copy()


class CoordinateTransformer:
    """
    Handles coordinate transformations between AVP and RealSense frames
    with probabilistic pose correction using Kalman filtering
    """

    def __init__(self, pose_manager):
        """
        Initialize coordinate transformer

        Args:
            pose_manager: PoseManager instance with calibration data
        """
        self.pose_manager = pose_manager

        # Kalman filter for pose smoothing
        self.kalman_filter = KalmanPoseFilter()

        # Last update time for prediction
        self.last_update_time = None

    def transform_point_avp_to_realsense(self, point_avp: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform 3D point from AVP frame to RealSense frame

        Args:
            point_avp: Point in AVP frame [x, y, z]

        Returns:
            Point in RealSense frame, or None if not calibrated
        """
        T = self.pose_manager.get_transform_avp_to_realsense()
        if T is None:
            print("[ERROR] Not calibrated - cannot transform")
            return None

        # Convert to homogeneous coordinates
        point_homo = np.append(point_avp, 1.0)

        # Transform
        point_rs_homo = T @ point_homo

        return point_rs_homo[:3]

    def transform_mask_avp_to_realsense(self, mask_avp: np.ndarray,
                                       K_avp: np.ndarray, K_rs: np.ndarray,
                                       size_rs: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Transform 2D mask from AVP view to RealSense view using homography

        Args:
            mask_avp: Binary mask in AVP view (H_avp, W_avp)
            K_avp: AVP camera intrinsics
            K_rs: RealSense camera intrinsics
            size_rs: RealSense image size (width, height)

        Returns:
            Transformed mask in RealSense view, or None
        """
        T = self.pose_manager.get_transform_avp_to_realsense()
        if T is None:
            return None

        # Extract rotation and translation
        R = T[:3, :3]
        t = T[:3, 3]

        # Compute homography assuming planar scene at distance d
        # H = K_rs * (R - t*n^T/d) * K_avp^{-1}
        # Simplified: assume d is large, so H ≈ K_rs * R * K_avp^{-1}

        H = K_rs @ R @ np.linalg.inv(K_avp)

        # Warp mask
        h_rs, w_rs = size_rs[1], size_rs[0]
        mask_rs = cv.warpPerspective(mask_avp, H, (w_rs, h_rs))

        return mask_rs

    def transform_depth_rs_to_avp(self, depth_rs: np.ndarray, K_rs: np.ndarray, K_avp: np.ndarray, size_avp: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Transform depth map from RealSense view to AVP view.

        Args:
            depth_rs: Depth map in RealSense view (H_rs, W_rs)
            K_rs: RealSense camera intrinsics
            K_avp: AVP camera intrinsics
            size_avp: AVP image size (width, height)

        Returns:
            Transformed depth map in AVP view, or None
        """
        T_rs_avp = self.pose_manager.get_transform_avp_to_realsense()
        if T_rs_avp is None:
            return None

        # Invert the transformation to get from RealSense to AVP
        T_avp_rs = self.invert_transformation(T_rs_avp)
        R_avp_rs = T_avp_rs[:3, :3]

        # Compute homography
        H = K_avp @ R_avp_rs @ np.linalg.inv(K_rs)

        # Warp depth image
        w_avp, h_avp = size_avp
        depth_avp = cv.warpPerspective(depth_rs, H, (w_avp, h_avp), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0)

        return depth_avp

    def update_pose_with_correction(self, measured_pose: dict, dt: float) -> dict:
        """
        Update pose estimate with probabilistic correction using Kalman filter

        Args:
            measured_pose: Raw measured pose with 'position' and 'rotation'
            dt: Time since last update

        Returns:
            Corrected pose estimate
        """
        # Convert rotation to quaternion if needed
        position = np.array(measured_pose["position"])

        if "quaternion" in measured_pose:
            quaternion = np.array(measured_pose["quaternion"])
        else:
            # Convert from rotation vector or Euler angles
            rotation = np.array(measured_pose["rotation"])
            r = Rotation.from_rotvec(rotation)
            quaternion = r.as_quat()  # [qx, qy, qz, qw]
            # Convert to [qw, qx, qy, qz]
            quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

        # Prediction step
        if dt > 0:
            self.kalman_filter.predict(dt)

        # Measurement update
        measurement = np.concatenate([position, quaternion])
        self.kalman_filter.update(measurement)

        # Get corrected pose
        corrected_pos, corrected_quat = self.kalman_filter.get_pose()

        # Convert quaternion back to rotation vector
        r = Rotation.from_quat([corrected_quat[1], corrected_quat[2], corrected_quat[3], corrected_quat[0]])
        corrected_rot = r.as_rotvec()

        return {
            "position": corrected_pos.tolist(),
            "rotation": corrected_rot.tolist(),
            "quaternion": corrected_quat.tolist()
        }

    def project_3d_to_2d(self, points_3d: np.ndarray, K: np.ndarray,
                        dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates

        Args:
            points_3d: 3D points (N, 3)
            K: Camera intrinsics matrix
            dist_coeffs: Distortion coefficients (optional)

        Returns:
            2D points (N, 2)
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5)

        # Use OpenCV projectPoints
        rvec = np.zeros(3)  # Identity rotation
        tvec = np.zeros(3)  # No translation

        points_2d, _ = cv.projectPoints(
            points_3d.reshape(-1, 1, 3),
            rvec, tvec, K, dist_coeffs
        )

        return points_2d.reshape(-1, 2)

    def compute_transformation_matrix(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Compute 4x4 transformation matrix from rvec and tvec

        Args:
            rvec: Rotation vector (3,)
            tvec: Translation vector (3,)

        Returns:
            4x4 transformation matrix
        """
        R, _ = cv.Rodrigues(rvec.reshape(3, 1))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    def invert_transformation(self, T: np.ndarray) -> np.ndarray:
        """
        Invert 4x4 transformation matrix

        Args:
            T: 4x4 transformation matrix

        Returns:
            Inverted transformation matrix
        """
        R = T[:3, :3]
        t = T[:3, 3]

        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t

        return T_inv


if __name__ == "__main__":
    print("Testing Coordinate Transformer...")

    # Mock pose manager
    class MockPoseManager:
        def get_transform_avp_to_realsense(self):
            # Identity transform for testing
            return np.eye(4)

    pose_manager = MockPoseManager()
    transformer = CoordinateTransformer(pose_manager)

    # Test point transformation
    point_avp = np.array([1.0, 2.0, 3.0])
    point_rs = transformer.transform_point_avp_to_realsense(point_avp)
    print(f"Point AVP: {point_avp}")
    print(f"Point RS: {point_rs}")

    # Test pose correction
    measured_pose = {
        "position": [0.1, 0.2, 0.3],
        "rotation": [0.0, 0.0, 0.1]
    }
    corrected_pose = transformer.update_pose_with_correction(measured_pose, dt=0.033)
    print(f"Measured: {measured_pose['position']}")
    print(f"Corrected: {corrected_pose['position']}")

    print("Test complete!")
