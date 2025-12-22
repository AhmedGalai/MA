"""
Coordinate Manager Module

Manages all coordinate transformations between AVP, RealSense, and World frames.
Handles calibration from ArUco detection and head pose updates from streamed data.

Author: MA Project
Date: 2025-12-14
"""

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation
import time
from typing import Optional, List, Tuple


class CoordinateManager:
    """
    Manages coordinate transformations between AVP, RealSense, and World frames.

    This class maintains transformation matrices between three coordinate frames:
    - World: Global reference frame
    - RealSense (RS): Camera coordinate frame
    - AVP: Augmented Vision Platform coordinate frame

    It also handles dynamic head pose corrections that update the AVP-to-World
    transformation based on streamed head pose data.

    Attributes:
        T_world_rs (ndarray): 4x4 transformation matrix from RealSense to World
        T_world_avp (ndarray): 4x4 transformation matrix from AVP to World
        head_pose_correction (ndarray): 4x4 correction matrix from head pose data
        last_head_pose_time (float): Timestamp of last head pose update
    """

    def __init__(self, T_world_rs: ndarray):
        """
        Initialize the CoordinateManager with RealSense calibration.

        Args:
            T_world_rs (ndarray): 4x4 transformation matrix from RealSense to World
                                  (from calibration data)

        Raises:
            ValueError: If T_world_rs is not a 4x4 matrix
            TypeError: If T_world_rs is not a numpy array
        """
        if not isinstance(T_world_rs, ndarray):
            raise TypeError("T_world_rs must be a numpy ndarray")

        if T_world_rs.shape != (4, 4):
            raise ValueError(f"T_world_rs must be 4x4, got shape {T_world_rs.shape}")

        self.T_world_rs = T_world_rs.astype(np.float64)
        self.T_world_avp: Optional[ndarray] = None
        self.T_world_avp_ref: Optional[ndarray] = None  # Reference when ArUco last seen
        self.T_world_head_ref: Optional[ndarray] = None  # Head pose when ArUco last seen
        self.T_world_head_current = np.eye(4, dtype=np.float64)  # Current head pose
        self.last_head_pose_time: Optional[float] = None
        self.head_pose_staleness_warning_threshold = 5.0  # seconds

    def set_avp_calibration(self, T_world_avp: ndarray, T_world_head: Optional[ndarray] = None) -> None:
        """
        Store AVP calibration from ArUco detection with optional reference head pose.

        This transformation represents the initial/reference pose of the AVP
        coordinate frame relative to the World coordinate frame, typically
        obtained from ArUco marker detection. If a reference head pose is provided,
        it enables continuous tracking when ArUco is not visible.

        Args:
            T_world_avp (ndarray): 4x4 transformation matrix from AVP to World
            T_world_head (ndarray, optional): 4x4 head pose at calibration time

        Raises:
            ValueError: If T_world_avp is not a 4x4 matrix
            TypeError: If T_world_avp is not a numpy array
        """
        if not isinstance(T_world_avp, ndarray):
            raise TypeError("T_world_avp must be a numpy ndarray")

        if T_world_avp.shape != (4, 4):
            raise ValueError(f"T_world_avp must be 4x4, got shape {T_world_avp.shape}")

        # Store reference transform
        self.T_world_avp_ref = T_world_avp.astype(np.float64)
        self.T_world_avp = self.T_world_avp_ref.copy()

        # Store reference head pose if provided
        if T_world_head is not None:
            if not isinstance(T_world_head, ndarray):
                raise TypeError("T_world_head must be a numpy ndarray")
            if T_world_head.shape != (4, 4):
                raise ValueError(f"T_world_head must be 4x4, got shape {T_world_head.shape}")
            self.T_world_head_ref = T_world_head.astype(np.float64)
        else:
            # No head pose provided, use current or identity
            self.T_world_head_ref = self.T_world_head_current.copy()

    def update_head_pose(self, position: List[float], quaternion: List[float],
                        timestamp: float) -> None:
        """
        Update head pose from streamed data and recompute T_world_avp for continuous tracking.

        Converts position and quaternion data to a 4x4 homogeneous transformation
        matrix. If a reference head pose exists, computes the relative head motion
        and updates T_world_avp to maintain tracking when ArUco is not visible.

        Args:
            position (List[float]): [x, y, z] position in meters (in ARKit world frame)
            quaternion (List[float]): [x, y, z, w] quaternion (scalar last)
            timestamp (float): Timestamp of head pose measurement

        Raises:
            ValueError: If position is not length 3 or quaternion is not length 4
            TypeError: If position or quaternion contain non-numeric values
        """
        try:
            position = np.array(position, dtype=np.float64)
            quaternion = np.array(quaternion, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Position and quaternion must be numeric: {e}")

        if position.shape != (3,):
            raise ValueError(f"Position must be length 3, got {position.shape}")

        if quaternion.shape != (4,):
            raise ValueError(f"Quaternion must be length 4, got {quaternion.shape}")

        # Create 4x4 transformation matrix from position and quaternion
        # Quaternion is [x, y, z, w] format (scipy expects this)
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

        self.T_world_head_current = np.eye(4, dtype=np.float64)
        self.T_world_head_current[:3, :3] = rotation_matrix
        self.T_world_head_current[:3, 3] = position

        self.last_head_pose_time = timestamp

        # If we have a reference calibration, update T_world_avp based on head motion
        if self.T_world_avp_ref is not None and self.T_world_head_ref is not None:
            # Compute head motion: T_head_delta = inv(T_world_head_ref) @ T_world_head_current
            # This represents how the head has moved since calibration
            try:
                T_head_delta = np.linalg.inv(self.T_world_head_ref) @ self.T_world_head_current

                # Update AVP transform: T_world_avp = T_world_avp_ref @ T_head_delta
                # This assumes AVP camera moves rigidly with the head
                self.T_world_avp = self.T_world_avp_ref @ T_head_delta
            except np.linalg.LinAlgError:
                # If inversion fails, keep previous T_world_avp
                pass

    def get_T_rs_avp(self) -> ndarray:
        """
        Compute transformation from AVP to RealSense frame.

        Computes: T_rs_avp = inv(T_world_rs) @ T_world_avp

        T_world_avp is automatically updated when head pose changes (if reference exists),
        so this method returns the current transform accounting for head motion.

        Returns:
            ndarray: 4x4 transformation matrix from AVP to RealSense

        Raises:
            RuntimeError: If T_world_avp is not yet calibrated
            np.linalg.LinAlgError: If T_world_rs is not invertible
        """
        if self.T_world_avp is None:
            raise RuntimeError("AVP calibration not set. Call set_avp_calibration() first.")

        # Check head pose staleness if we're using continuous tracking
        if (self.T_world_avp_ref is not None and
            self.T_world_head_ref is not None and
            self.last_head_pose_time is not None):
            age = time.time() - self.last_head_pose_time
            if age > self.head_pose_staleness_warning_threshold:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Head pose data is stale (%.1f seconds old). "
                    "Continuous tracking may be inaccurate.",
                    age
                )

        try:
            T_rs_world = np.linalg.inv(self.T_world_rs)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Cannot invert T_world_rs: {e}")

        # T_rs_avp = inv(T_world_rs) @ T_world_avp
        # Note: T_world_avp already includes head motion if reference exists
        T_rs_avp = T_rs_world @ self.T_world_avp

        return T_rs_avp

    def get_T_avp_rs(self) -> ndarray:
        """
        Get inverse transformation from RealSense to AVP frame.

        Returns the inverse of get_T_rs_avp(), representing the transformation
        of coordinates from the RealSense camera frame to the AVP frame.

        Returns:
            ndarray: 4x4 transformation matrix from RealSense to AVP

        Raises:
            RuntimeError: If T_world_avp is not yet calibrated
            np.linalg.LinAlgError: If the computed T_rs_avp is not invertible
        """
        try:
            T_rs_avp = self.get_T_rs_avp()
            T_avp_rs = np.linalg.inv(T_rs_avp)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Cannot invert T_rs_avp: {e}")

        return T_avp_rs

    def transform_pose_rs_to_avp(self, T_rs_object: ndarray) -> ndarray:
        """
        Transform object pose from RealSense frame to AVP frame.

        Transforms a pose/transformation matrix from the RealSense coordinate
        frame to the AVP coordinate frame using the current calibration and
        head pose correction.

        Args:
            T_rs_object (ndarray): 4x4 transformation matrix of object in RS frame

        Returns:
            ndarray: 4x4 transformation matrix of object in AVP frame

        Raises:
            ValueError: If T_rs_object is not a 4x4 matrix
            RuntimeError: If T_world_avp is not yet calibrated
        """
        if not isinstance(T_rs_object, ndarray):
            raise TypeError("T_rs_object must be a numpy ndarray")

        if T_rs_object.shape != (4, 4):
            raise ValueError(f"T_rs_object must be 4x4, got shape {T_rs_object.shape}")

        T_avp_rs = self.get_T_avp_rs()
        T_avp_object = T_avp_rs @ T_rs_object

        return T_avp_object

    def get_rs_pose_in_avp(self) -> ndarray:
        """
        Get RealSense camera pose in AVP frame.

        Returns the transformation that represents the RealSense camera's
        position and orientation relative to the AVP coordinate frame.
        This is equivalent to the camera-to-camera transformation T_avp_rs.

        Returns:
            ndarray: 4x4 transformation matrix of RealSense camera in AVP frame

        Raises:
            RuntimeError: If T_world_avp is not yet calibrated
        """
        return self.get_T_avp_rs()

    def is_calibrated(self) -> bool:
        """
        Check if both required calibrations are set.

        Returns True only if both RealSense (from initialization) and AVP
        (from ArUco detection) calibrations are available.

        Returns:
            bool: True if fully calibrated, False otherwise
        """
        return self.T_world_avp is not None

    # Utility methods for debugging and inspection

    def get_T_world_rs(self) -> ndarray:
        """
        Get the RealSense to World transformation matrix.

        Returns:
            ndarray: 4x4 transformation matrix
        """
        return self.T_world_rs.copy()

    def get_T_world_avp(self) -> Optional[ndarray]:
        """
        Get the AVP to World transformation matrix.

        Returns:
            ndarray or None: 4x4 transformation matrix or None if not calibrated
        """
        return self.T_world_avp.copy() if self.T_world_avp is not None else None

    def get_head_pose_correction(self) -> ndarray:
        """
        Get the current head pose correction matrix.

        Returns:
            ndarray: 4x4 head pose transformation matrix
        """
        return self.head_pose_correction.copy()

    def reset_head_pose(self) -> None:
        """
        Reset head pose correction to identity matrix.

        Useful for resetting head tracking or when switching to a different
        tracking source.
        """
        self.head_pose_correction = np.eye(4, dtype=np.float64)
        self.last_head_pose_time = None

    def __str__(self) -> str:
        """Return string representation of calibration status."""
        status = "CoordinateManager Status:\n"
        status += f"  RealSense Calibrated: Yes\n"
        status += f"  AVP Calibrated: {'Yes' if self.is_calibrated() else 'No'}\n"
        status += f"  Head Pose Last Update: {self.last_head_pose_time}\n"
        return status
