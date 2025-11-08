#!/usr/bin/env python3
"""
Computer Vision Pipeline
Optimized pipeline with GPU support for:
- ArUco marker detection
- Pose estimation
- ROI mask extraction
- Monocular Depth Estimation (MDE) using transformer model

Runs independently and communicates with main API.
"""

import numpy as np
import cv2 as cv
import torch
from transformers import pipeline as hf_pipeline
from PIL import Image
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import io
import base64

# ------------------ GPU Detection ------------------
def detect_gpu():
    """Detect available GPU and configure device"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[GPU] CUDA available: {gpu_name}")
        print(f"[GPU] CUDA version: {torch.version.cuda}")
        return device
    else:
        print("[GPU] CUDA not available, using CPU")
        return "cpu"

# Global device configuration
DEVICE = detect_gpu()

# ------------------ ArUco Configuration ------------------
def get_aruco_handles():
    """Get ArUco detection handles with version compatibility"""
    if not hasattr(cv, "aruco"):
        print("[ERROR] cv2.aruco not available")
        return None, None, None

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
        api = "new"
    else:
        det = params
        api = "old"

    return dct, det, api

ROWS, COLS = 3, 4
MARKER_SIZE_M = 0.030
SEPARATION_M = 0.010

def board_id_to_corners_m(marker_id: int):
    """Convert marker ID to 3D corner positions on the board"""
    if marker_id < 0 or marker_id >= ROWS * COLS:
        return None
    row, col = divmod(marker_id, COLS)
    x0 = col * (MARKER_SIZE_M + SEPARATION_M)
    y0 = row * (MARKER_SIZE_M + SEPARATION_M)
    return np.array([
        [x0,               y0,               0],
        [x0+MARKER_SIZE_M, y0,               0],
        [x0+MARKER_SIZE_M, y0+MARKER_SIZE_M, 0],
        [x0,               y0+MARKER_SIZE_M, 0]
    ], dtype=np.float32)

def solve_board_pose(corners, ids, K, dist):
    """Solve for board pose using detected markers"""
    if ids is None or len(ids) == 0:
        return None

    obj_pts, img_pts = [], []
    for i, c in zip(ids.flatten().tolist(), corners):
        obj = board_id_to_corners_m(i)
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

    return rvec.reshape(3), tvec.reshape(3)

def default_K_for_size(w, h):
    """Generate default camera intrinsics for given image size"""
    f = 0.8 * max(w, h)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((5, 1), np.float32)
    return K, dist

# ------------------ ROI Mask Creation ------------------
def create_roi_mask_gpu(frame_bgr, hsv_center, h_tol=12, s_tol=50, v_tol=50):
    """
    Create ROI mask based on HSV color range
    Uses GPU acceleration if available for color conversion
    """
    h, s, v = hsv_center

    # Convert tolerances to HSV ranges
    dh = int(round((h_tol / 360.0) * 179.0))
    ds = int(round((s_tol / 100.0) * 255.0))
    dv = int(round((v_tol / 100.0) * 255.0))

    lo = np.array([max(0, h - dh), max(0, s - ds), max(0, v - dv)], dtype=np.uint8)
    hi = np.array([min(179, h + dh), min(255, s + ds), min(255, v + dv)], dtype=np.uint8)

    # Color conversion (GPU accelerated in OpenCV if CUDA build)
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lo, hi)

    return mask

# ------------------ Depth Estimation ------------------
class DepthEstimator:
    """Monocular Depth Estimation using transformer model with GPU support"""

    def __init__(self, model_name="depth-anything/Depth-Anything-V2-base-hf"):
        self.device = DEVICE
        print(f"[MDE] Loading depth model: {model_name}")
        print(f"[MDE] Using device: {self.device}")

        start_time = time.time()
        try:
            self.pipe = hf_pipeline(
                "depth-estimation",
                model=model_name,
                device=self.device
            )
            load_time = time.time() - start_time
            print(f"[MDE] Model loaded successfully in {load_time:.2f}s")
        except Exception as e:
            print(f"[ERROR] Failed to load depth model: {e}")
            self.pipe = None

    def estimate_depth(self, frame_bgr):
        """
        Estimate depth from RGB frame
        Returns: depth map as numpy array (grayscale)
        """
        if self.pipe is None:
            print("[ERROR] Depth model not available")
            return None

        try:
            # Convert BGR to RGB PIL image
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Run inference (GPU accelerated)
            start_time = time.time()
            result = self.pipe(pil_img)
            inference_time = time.time() - start_time

            # Extract depth map
            depth_pil = result["depth"]
            depth_np = np.array(depth_pil)

            print(f"[MDE] Inference time: {inference_time*1000:.1f}ms")

            return depth_np

        except Exception as e:
            print(f"[ERROR] Depth estimation failed: {e}")
            return None

    def depth_to_disparity(self, depth_map):
        """Convert depth map to disparity map"""
        if depth_map is None:
            return None

        # Normalize to 0-255 range
        depth_normalized = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX)

        # Invert (closer = brighter in disparity)
        disparity = 255 - depth_normalized

        return disparity.astype(np.uint8)

# ------------------ Pipeline State ------------------
@dataclass
class PipelineConfig:
    """Configuration for the CV pipeline"""
    hsv_center: list = None
    h_tol: int = 12
    s_tol: int = 50
    v_tol: int = 50

    def __post_init__(self):
        if self.hsv_center is None:
            self.hsv_center = [90, 128, 128]

class PipelineState:
    """Thread-safe state for CV pipeline"""

    def __init__(self):
        self.lock = threading.Lock()

        # Configuration
        self.config = PipelineConfig()

        # ArUco detection
        self.aruco_dict, self.detector, self.api = get_aruco_handles()

        # Results
        self.last_frame = None
        self.last_intrinsics = None
        self.last_pose = None
        self.last_mask = None
        self.last_depth = None
        self.last_disparity = None

        # Statistics
        self.frames_processed = 0
        self.aruco_detections = 0
        self.pose_successes = 0

        # Depth estimator (lazy load)
        self._depth_estimator = None

    @property
    def depth_estimator(self):
        """Lazy load depth estimator"""
        if self._depth_estimator is None:
            self._depth_estimator = DepthEstimator()
        return self._depth_estimator

# Global state
state = PipelineState()

# ------------------ Main Processing Function ------------------
def process_frame(frame_bgr: np.ndarray, estimate_depth: bool = False) -> Dict[str, Any]:
    """
    Process a single frame through the complete CV pipeline

    Args:
        frame_bgr: Input frame in BGR format
        estimate_depth: Whether to run depth estimation (expensive)

    Returns:
        Dictionary with processing results
    """
    try:
        results = {}

        # Get current config
        with state.lock:
            config = PipelineConfig(
                hsv_center=state.config.hsv_center.copy(),
                h_tol=state.config.h_tol,
                s_tol=state.config.s_tol,
                v_tol=state.config.v_tol
            )

        # Get frame dimensions and compute intrinsics
        h, w = frame_bgr.shape[:2]
        K, dist = default_K_for_size(w, h)
        results['intrinsics'] = {"K": K.tolist(), "dist": dist.tolist()}

        # ArUco detection
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        corners, ids, rejected = None, None, None

        if state.aruco_dict is not None:
            if state.api == "new":
                corners, ids, rejected = state.detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv.aruco.detectMarkers(
                    gray, state.aruco_dict, parameters=state.detector
                )

            if ids is not None and len(ids) > 0:
                with state.lock:
                    state.aruco_detections += 1

        # Pose estimation
        pose = None
        if ids is not None and len(ids) > 0:
            pose_result = solve_board_pose(corners, ids, K, dist)
            if pose_result is not None:
                rvec, tvec = pose_result
                pose = {
                    "rvec": rvec.tolist(),
                    "tvec": tvec.tolist(),
                    "markers_detected": len(ids)
                }
                with state.lock:
                    state.pose_successes += 1

        results['pose'] = pose

        # ROI Mask extraction
        mask = create_roi_mask_gpu(
            frame_bgr,
            config.hsv_center,
            config.h_tol,
            config.s_tol,
            config.v_tol
        )
        results['mask'] = mask

        # Depth estimation (optional, expensive)
        depth_map = None
        disparity = None
        if estimate_depth:
            depth_map = state.depth_estimator.estimate_depth(frame_bgr)
            if depth_map is not None:
                disparity = state.depth_estimator.depth_to_disparity(depth_map)

        results['depth'] = depth_map
        results['disparity'] = disparity

        # Update state
        with state.lock:
            state.last_frame = frame_bgr
            state.last_intrinsics = results['intrinsics']
            state.last_pose = pose
            state.last_mask = mask
            state.last_depth = depth_map
            state.last_disparity = disparity
            state.frames_processed += 1

        results['success'] = True
        results['frame_count'] = state.frames_processed

        return results

    except Exception as e:
        print(f"[ERROR] process_frame: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# ------------------ Configuration Update ------------------
def update_config(hsv_center=None, h_tol=None, s_tol=None, v_tol=None):
    """Update pipeline configuration"""
    with state.lock:
        if hsv_center is not None:
            state.config.hsv_center = hsv_center
        if h_tol is not None:
            state.config.h_tol = h_tol
        if s_tol is not None:
            state.config.s_tol = s_tol
        if v_tol is not None:
            state.config.v_tol = v_tol

def get_config():
    """Get current configuration"""
    with state.lock:
        return {
            "hsv_center": state.config.hsv_center,
            "h_tol": state.config.h_tol,
            "s_tol": state.config.s_tol,
            "v_tol": state.config.v_tol
        }

def get_stats():
    """Get pipeline statistics"""
    with state.lock:
        return {
            "frames_processed": state.frames_processed,
            "aruco_detections": state.aruco_detections,
            "pose_successes": state.pose_successes,
            "has_intrinsics": state.last_intrinsics is not None,
            "has_pose": state.last_pose is not None,
            "has_mask": state.last_mask is not None,
            "has_depth": state.last_depth is not None,
            "device": DEVICE
        }

# ------------------ Helper Functions ------------------
def encode_image_to_base64(img):
    """Encode numpy array to base64 string"""
    if img is None:
        return None

    # Handle grayscale images
    if len(img.shape) == 2:
        img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    else:
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def decode_base64_image(base64_str):
    """Decode base64 string to numpy array (BGR)"""
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    img_rgb = np.array(img)
    img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
    return img_bgr

# ------------------ Main (for testing) ------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Computer Vision Pipeline - GPU Accelerated")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"ArUco available: {state.aruco_dict is not None}")
    print("=" * 60)

    # Test depth estimator initialization
    print("\nInitializing depth estimator...")
    estimator = state.depth_estimator
    print("Ready!")
