"""
Final Pipeline Core
Integrates all components for complete 6D pose estimation pipeline
"""

import numpy as np
import cv2 as cv
import time
from typing import Optional, Dict
import base64
import io
from PIL import Image
import os
import json
from datetime import datetime

from .realsense_depth import RealSenseDepth
from .pose_manager import PoseManager
from .coordinate_transformer import CoordinateTransformer
from .pose_estimator import PoseEstimator


class FinalPipeline:
    """
    Complete pipeline for 6D pose estimation:
    1. Capture RealSense depth (fixed camera)
    2. Get headset pose (streaming)
    3. Apply probabilistic pose correction
    4. Transform mask AVP → RealSense
    5. Estimate pose in RealSense view
    6. Transform pose back to AVP view
    """

    def __init__(self):
        """Initialize pipeline components"""
        print("[Pipeline] Initializing Final Pipeline...")

        # Initialize components
        self.realsense = RealSenseDepth()
        self.pose_manager = PoseManager()
        self.transformer = CoordinateTransformer(self.pose_manager)
        self.pose_estimator = PoseEstimator()

        # State
        self.last_process_time = None
        self.stats = {
            "frames_processed": 0,
            "successful_poses": 0,
            "failed_poses": 0,
            "avg_processing_time_ms": 0.0
        }

        # Check if ready
        if not self.realsense.available:
            print("[WARNING] RealSense not available - pipeline will not work")

        if not self.pose_manager.is_calibrated():
            print("[WARNING] Pipeline not calibrated - run calibrate_with_aruco()")

        # Create directory for saving pose requests
        self.pose_request_save_dir = "pose_estimation_io"
        os.makedirs(self.pose_request_save_dir, exist_ok=True)

        print("[Pipeline] Initialization complete")

    def calibrate_with_aruco(self, headset_image: np.ndarray,
                            headset_K: np.ndarray, headset_dist: np.ndarray) -> bool:
        """
        Perform one-time ArUco calibration

        Args:
            headset_image: Image from headset with ArUco marker visible
            headset_K: Headset camera intrinsics
            headset_dist: Headset distortion coefficients

        Returns:
            True if calibration successful
        """
        return self.pose_manager.calibrate_with_aruco(
            self.realsense,
            headset_image,
            headset_K,
            headset_dist
        )

    def process_frame(self, avp_rgb: Optional[np.ndarray] = None,
                     avp_mask: Optional[np.ndarray] = None,
                     headset_pose: Optional[Dict] = None,
                     save_pose_request_data: bool = False) -> Dict:
        """
        Process frame through complete pipeline

        Args:
            avp_rgb: RGB frame from AVP (optional, for visualization)
            avp_mask: Object mask from AVP view (H, W) binary
            headset_pose: Current headset pose dict with 'position' and 'rotation'
            save_pose_request_data: If True, saves inputs and outputs of pose estimation

        Returns:
            Dictionary with results:
                - pose_avp_view: Final pose in AVP coordinates
                - pose_rs_view: Pose in RealSense coordinates
                - confidence: Pose confidence score
                - processing_time_ms: Processing time
                - success: Boolean success flag
        """
        start_time = time.time()
        result = {
            "success": False,
            "error": None,
            "processing_time_ms": 0.0
        }

        try:
            # Validate inputs
            if avp_mask is None:
                result["error"] = "No mask provided"
                return result

            if not self.realsense.available:
                result["error"] = "RealSense not available"
                return result

            if not self.pose_manager.is_calibrated():
                result["error"] = "Pipeline not calibrated"
                return result

            # Step 1: Capture RealSense depth
            print("[Pipeline] Step 1: Capturing RealSense depth...")
            rs_data = self.realsense.capture_frame()
            if rs_data is None:
                result["error"] = "Failed to capture RealSense frame"
                return result

            depth_map = rs_data["depth"]
            rs_rgb = rs_data["rgb"]
            K_rs = rs_data["intrinsics"]["K"]
            dist_rs = rs_data["intrinsics"]["dist"]

            # Step 2: Update headset pose with correction
            if headset_pose is not None:
                print("[Pipeline] Step 2: Updating headset pose...")
                self.pose_manager.update_headset_pose(headset_pose)

                # Calculate time delta
                dt = 0.033  # Default 30fps
                if self.last_process_time is not None:
                    dt = time.time() - self.last_process_time

                # Apply probabilistic correction
                corrected_pose = self.transformer.update_pose_with_correction(headset_pose, dt)
                print(f"[Pipeline] Pose corrected: {corrected_pose['position']}")
            else:
                print("[Pipeline] No headset pose provided")

            # Step 3: Transform mask from AVP → RealSense view
            print("[Pipeline] Step 3: Transforming mask AVP → RealSense...")

            # Need AVP camera intrinsics (estimate from mask size)
            h_avp, w_avp = avp_mask.shape
            K_avp = self._estimate_camera_matrix(w_avp, h_avp)

            h_rs, w_rs = depth_map.shape
            mask_rs = self.transformer.transform_mask_avp_to_realsense(
                avp_mask, K_avp, K_rs, (w_rs, h_rs)
            )

            if mask_rs is None:
                result["error"] = "Mask transformation failed"
                return result

            # Step 4: Estimate 6D pose in RealSense view
            print("[Pipeline] Step 4: Estimating pose in RealSense view...")
            pose_rs = self.pose_estimator.estimate_pose_from_depth_and_mask(
                depth_map, mask_rs, K_rs, dist_rs
            )

            if pose_rs is None:
                result["error"] = "Pose estimation failed"
                self.stats["failed_poses"] += 1
                if save_pose_request_data:
                    self._save_pose_request_data(depth_map, mask_rs, K_rs, dist_rs, None)
                return result

            # Step 5: Transform pose back to AVP view
            print("[Pipeline] Step 5: Transforming pose RealSense → AVP...")
            pose_avp = self._transform_pose_rs_to_avp(pose_rs)

            if pose_avp is None:
                result["error"] = "Pose transformation to AVP failed"
                return result
            
            if save_pose_request_data:
                self._save_pose_request_data(depth_map, mask_rs, K_rs, dist_rs, pose_rs)

            # Success!
            result["success"] = True
            result["pose_rs_view"] = {
                "rvec": pose_rs["rvec"].tolist(),
                "tvec": pose_rs["tvec"].tolist(),
                "confidence": pose_rs["confidence"]
            }
            result["pose_avp_view"] = {
                "rvec": pose_avp["rvec"].tolist(),
                "tvec": pose_avp["tvec"].tolist(),
                "confidence": pose_avp["confidence"]
            }
            result["confidence"] = pose_rs["confidence"]
            result["num_points"] = pose_rs["num_points"]

            # Optional: Add visualizations
            if avp_rgb is not None:
                result["visualization"] = self._create_visualization(
                    avp_rgb, avp_mask, pose_avp, K_avp
                )

            self.stats["successful_poses"] += 1

        except Exception as e:
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
            self.stats["failed_poses"] += 1

        finally:
            # Update stats
            processing_time = (time.time() - start_time) * 1000  # ms
            result["processing_time_ms"] = processing_time

            self.stats["frames_processed"] += 1
            alpha = 0.9  # Exponential moving average
            self.stats["avg_processing_time_ms"] = (
                alpha * self.stats.get("avg_processing_time_ms", 0) +
                (1 - alpha) * processing_time
            )

            self.last_process_time = time.time()

            print(f"[Pipeline] Processing time: {processing_time:.1f}ms")
            print(f"[Pipeline] Success: {result['success']}")

        return result

    def _save_pose_request_data(self, depth_map, mask, K, dist, pose_result):
        """Saves the inputs and output of a pose estimation request."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_dir = os.path.join(self.pose_request_save_dir, timestamp)
            os.makedirs(save_dir, exist_ok=True)

            # Save depth map
            np.save(os.path.join(save_dir, "depth_map.npy"), depth_map)

            # Save mask
            cv.imwrite(os.path.join(save_dir, "mask.png"), mask)

            # Save intrinsics and distortion
            intrinsics_data = {
                "K": K.tolist(),
                "dist": dist.tolist()
            }
            with open(os.path.join(save_dir, "intrinsics.json"), 'w') as f:
                json.dump(intrinsics_data, f, indent=4)

            # Save pose result
            if pose_result:
                pose_data_to_save = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in pose_result.items()}
                with open(os.path.join(save_dir, "pose_result.json"), 'w') as f:
                    json.dump(pose_data_to_save, f, indent=4)
            else:
                with open(os.path.join(save_dir, "pose_result.json"), 'w') as f:
                    json.dump({"error": "Pose estimation failed"}, f, indent=4)

            print(f"[Pipeline] Saved pose request data to {save_dir}")
        except Exception as e:
            print(f"[ERROR] Could not save pose request data: {e}")

    def _transform_pose_rs_to_avp(self, pose_rs: Dict) -> Optional[Dict]:
        """
        Transform pose from RealSense to AVP coordinate frame

        Args:
            pose_rs: Pose in RealSense frame

        Returns:
            Pose in AVP frame, or None
        """
        # Get transformation matrix
        T_rs_avp = self.pose_manager.get_transform_avp_to_realsense()
        if T_rs_avp is None:
            return None

        # Invert to get T_avp_rs
        T_avp_rs = self.transformer.invert_transformation(T_rs_avp)

        # Get pose in RealSense frame
        T_rs_obj = self.transformer.compute_transformation_matrix(
            pose_rs["rvec"], pose_rs["tvec"]
        )

        # Transform to AVP frame: T_avp_obj = T_avp_rs * T_rs_obj
        T_avp_obj = T_avp_rs @ T_rs_obj

        # Extract rvec, tvec
        R_avp = T_avp_obj[:3, :3]
        tvec_avp = T_avp_obj[:3, 3]
        rvec_avp, _ = cv.Rodrigues(R_avp)

        return {
            "rvec": rvec_avp.flatten(),
            "tvec": tvec_avp,
            "confidence": pose_rs["confidence"]
        }

    def _estimate_camera_matrix(self, w: int, h: int) -> np.ndarray:
        """Estimate camera intrinsics from image size"""
        f = 0.8 * max(w, h)
        cx, cy = w / 2.0, h / 2.0
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

    def _create_visualization(self, rgb: np.ndarray, mask: np.ndarray,
                             pose: Dict, K: np.ndarray) -> str:
        """
        Create visualization image with pose overlay

        Args:
            rgb: RGB image
            mask: Binary mask
            pose: Pose dictionary
            K: Camera intrinsics

        Returns:
            Base64 encoded JPEG string
        """
        vis = rgb.copy()

        # Draw mask overlay
        mask_overlay = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        mask_overlay[:, :, 1] = 255  # Green mask
        vis = cv.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

        # Draw coordinate axes
        axis_length = 0.1  # 10cm axes
        axis_3d = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ])

        # Project axes
        imgpts, _ = cv.projectPoints(
            axis_3d, pose["rvec"], pose["tvec"], K, np.zeros(5)
        )
        imgpts = imgpts.reshape(-1, 2).astype(int)

        # Draw axes (X=red, Y=green, Z=blue)
        origin = tuple(imgpts[0])
        vis = cv.line(vis, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X - red
        vis = cv.line(vis, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y - green
        vis = cv.line(vis, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z - blue

        # Encode to base64
        vis_rgb = cv.cvtColor(vis, cv.COLOR_BGR2RGB)
        pil_img = Image.fromarray(vis_rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=90)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/jpeg;base64,{img_str}"

    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return self.stats.copy()

    def shutdown(self):
        """Shutdown pipeline and cleanup resources"""
        print("[Pipeline] Shutting down...")
        self.realsense.stop()
        print("[Pipeline] Shutdown complete")


# Test functionality
if __name__ == "__main__":
    print("Testing Final Pipeline...")

    pipeline = FinalPipeline()

    if not pipeline.realsense.available:
        print("RealSense not available - cannot test pipeline")
        exit(1)

    # Test frame processing
    print("\nProcessing test frame...")
    test_mask = np.zeros((480, 640), dtype=np.uint8)
    test_mask[200:300, 300:400] = 255

    test_pose = {
        "position": [0.0, 0.0, 0.5],
        "rotation": [0.0, 0.0, 0.0]
    }

    result = pipeline.process_frame(
        avp_mask=test_mask,
        headset_pose=test_pose
    )

    print(f"\nResult: {result['success']}")
    if result['success']:
        print(f"Pose (AVP): {result['pose_avp_view']}")
        print(f"Confidence: {result['confidence']:.3f}")
    else:
        print(f"Error: {result['error']}")

    print(f"\nStats: {pipeline.get_stats()}")

    pipeline.shutdown()
    print("\nTest complete!")
