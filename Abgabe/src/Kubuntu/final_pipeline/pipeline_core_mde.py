"""
Final Pipeline Core - Monocular Depth Estimation (MDE) Version
Integrates MDE with other components for a 6D pose estimation pipeline that does not require a hardware depth sensor.
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

from .depth_anything import DepthAnythingV2
from .pose_estimator import PoseEstimator
from .pose_manager import PoseManager
from .coordinate_transformer import CoordinateTransformer

class FinalPipeline:
    """
    Complete pipeline for 6D pose estimation using Monocular Depth Estimation:
    1. Get AVP RGB frame and mask.
    2. Estimate depth from the AVP RGB frame using DepthAnythingV2.
    3. Estimate pose directly in AVP view using the estimated depth and mask.
    """

    def __init__(self):
        """Initialize pipeline components"""
        print("[Pipeline] Initializing MDE Pipeline...")

        # Initialize components
        self.depth_estimator = DepthAnythingV2()
        self.pose_estimator = PoseEstimator()
        self.pose_manager = PoseManager()
        self.transformer = CoordinateTransformer(self.pose_manager)


        # State
        self.last_process_time = None
        self.stats = {
            "frames_processed": 0,
            "successful_poses": 0,
            "failed_poses": 0,
            "avg_processing_time_ms": 0.0
        }

        # Create directory for saving pose requests
        self.pose_request_save_dir = "pose_estimation_io"
        os.makedirs(self.pose_request_save_dir, exist_ok=True)

        print("[Pipeline] MDE Pipeline Initialization complete")

    def calibrate_with_aruco(self, *args, **kwargs):
        """Calibration is not required for the MDE pipeline."""
        print("[WARNING] Calibration is not used in the MDE pipeline.")
        return True

    def process_frame(self, avp_rgb: Optional[np.ndarray] = None,
                     avp_mask: Optional[np.ndarray] = None,
                     headset_pose: Optional[Dict] = None,
                     save_pose_request_data: bool = False) -> Dict:
        """
        Process frame through the MDE pipeline.

        Args:
            avp_rgb: RGB frame from AVP.
            avp_mask: Object mask from AVP view (H, W) binary.
            headset_pose: Current headset pose (not used in this pipeline but kept for API compatibility).
            save_pose_request_data: If True, saves inputs and outputs of pose estimation.

        Returns:
            Dictionary with results.
        """
        start_time = time.time()
        result = {
            "success": False,
            "error": None,
            "processing_time_ms": 0.0
        }

        try:
            # Validate inputs
            if avp_rgb is None:
                result["error"] = "No RGB image provided"
                return result
            if avp_mask is None:
                result["error"] = "No mask provided"
                return result

            # Step 1: Estimate depth from AVP RGB
            print("[Pipeline] Step 1: Estimating depth from AVP RGB...")
            depth_map_avp = self.depth_estimator.estimate_depth(avp_rgb)
            # The depth map is normalized to 0-1, so we need to scale it to a metric scale.
            # This is a key challenge in MDE. For now, we'll use a fixed scale factor.
            depth_map_avp = (depth_map_avp * 1000).astype(np.uint16) # Scale to millimeters

            # Step 2: Estimate 6D pose in AVP view
            print("[Pipeline] Step 2: Estimating pose in AVP view...")
            h_avp, w_avp = avp_mask.shape
            K_avp = self._estimate_camera_matrix(w_avp, h_avp)
            dist_avp = np.zeros(5)

            pose_avp = self.pose_estimator.estimate_pose_from_depth_and_mask(
                depth_map_avp, avp_mask, K_avp, dist_avp
            )

            if pose_avp is None:
                result["error"] = "Pose estimation failed"
                self.stats["failed_poses"] += 1
                if save_pose_request_data:
                    self._save_pose_request_data(depth_map_avp, avp_mask, K_avp, dist_avp, None)
                return result
            
            if save_pose_request_data:
                self._save_pose_request_data(depth_map_avp, avp_mask, K_avp, dist_avp, pose_avp)

            # Success!
            result["success"] = True
            result["pose_avp_view"] = {
                "rvec": pose_avp["rvec"].tolist(),
                "tvec": pose_avp["tvec"].tolist(),
                "confidence": pose_avp["confidence"]
            }
            result["pose_rs_view"] = None # Not applicable
            result["confidence"] = pose_avp["confidence"]
            result["num_points"] = pose_avp["num_points"]

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
            processing_time = (time.time() - start_time) * 1000
            result["processing_time_ms"] = processing_time
            self.stats["frames_processed"] += 1
            alpha = 0.9
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

    def _estimate_camera_matrix(self, w: int, h: int) -> np.ndarray:
        """Estimate camera intrinsics from image size"""
        f = 0.8 * max(w, h)
        cx, cy = w / 2.0, h / 2.0
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

    def _create_visualization(self, rgb: np.ndarray, mask: np.ndarray,
                             pose: Dict, K: np.ndarray) -> str:
        """
        Create visualization image with pose overlay
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
        # No realsense to stop
        print("[Pipeline] Shutdown complete")


# Test functionality
if __name__ == "__main__":

    print("Testing MDE Pipeline...")

    pipeline = FinalPipeline()

    # Test frame processing
    print("\nProcessing test frame...")
    test_avp_rgb = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
    test_mask = np.zeros((480, 640), dtype=np.uint8)
    test_mask[200:300, 300:400] = 255

    result = pipeline.process_frame(
        avp_rgb=test_avp_rgb,
        avp_mask=test_mask
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
