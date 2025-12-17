"""
Flask REST API server for pose estimation pipeline.

Provides endpoints for head pose updates, RealSense and AVP calibration,
pose estimation, and system health monitoring.
"""

import json
import logging
import os
import threading
import base64
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from config import CONFIG
from aruco_calibration import detect_aruco_board, save_calibration, load_calibration
from realsense_client import RealSenseClient
from coordinate_manager import CoordinateManager
from mask_transformer import transform_mask_avp_to_rs
from foundationpose_client import estimate_pose

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__)
CORS(app)

# Global state
realsense_client = None
coordinate_manager = None
state_lock = threading.Lock()

# AVP (Apple Vision Pro) frame storage
last_avp_frame = None
last_avp_frame_timestamp = None
last_avp_frame_metadata = {
    'width': None,
    'height': None,
    'receive_time': None
}

# Head pose storage (from VisionOS device)
last_head_pose = None
last_head_pose_metadata = {
    'position': None,
    'quaternion': None,
    'timestamp': None,
    'receive_time': None,
    'reception_count': 0,
    'last_reception_time': None
}

# Intrinsics storage for both cameras
rs_intrinsics = {
    'K': None,
    'calculated': False,
    'method': None,
    'timestamp': None
}

avp_intrinsics = {
    'K': None,
    'calculated': False,
    'method': None,
    'timestamp': None
}

# Cache for latest RS frames to avoid hard failures when capture is briefly unavailable
last_rs_frame = None
last_rs_depth = None
last_rs_timestamp = None
last_rs_K = None
selected_model = None
last_avp_frames = {}
last_avp_board_pose = None

def ensure_coordinate_manager():
    """Ensure coordinate_manager is initialized with a safe default."""
    global coordinate_manager
    if coordinate_manager is None:
        try:
            coordinate_manager = CoordinateManager(T_world_rs=np.eye(4))
            logger.warning("CoordinateManager initialized with identity T_world_rs (no calibration file found)")
        except Exception as e:
            logger.error(f"Failed to initialize CoordinateManager fallback: {e}")

def get_latest_rs_frame(allow_cache=True):
    """
    Try to capture an RS frame; optionally fall back to cached frame.
    Returns (frame_data, using_cache: bool) or (None, False) if unavailable.
    """
    global last_rs_frame, last_rs_depth, last_rs_timestamp, last_rs_K

    frame_data = None
    using_cache = False

    rs_ready = realsense_client is not None and realsense_client.is_running
    if rs_ready:
        frame_data = realsense_client.capture()

    if frame_data is None and allow_cache and last_rs_frame is not None and last_rs_depth is not None:
        frame_data = {
            'rgb': last_rs_frame.copy(),
            'depth': last_rs_depth.copy(),
            'K': last_rs_K.copy() if last_rs_K is not None else None,
            'timestamp': last_rs_timestamp
        }
        using_cache = True

    if frame_data is not None and not using_cache:
        last_rs_frame = frame_data['rgb'].copy()
        if 'depth' in frame_data and frame_data['depth'] is not None:
            last_rs_depth = frame_data['depth'].copy()
        if frame_data.get('K') is not None:
            last_rs_K = frame_data['K'].copy()
        last_rs_timestamp = frame_data.get('timestamp', time.time())

    return frame_data, using_cache


def store_avp_frame(frame_np, timestamp, purpose):
    global last_avp_frame, last_avp_frame_timestamp, last_avp_frame_metadata, last_avp_frames
    last_avp_frame = frame_np
    last_avp_frame_timestamp = timestamp
    last_avp_frame_metadata['width'] = frame_np.shape[1]
    last_avp_frame_metadata['height'] = frame_np.shape[0]
    last_avp_frame_metadata['receive_time'] = time.time()
    if purpose not in last_avp_frames:
        last_avp_frames[purpose] = {}
    last_avp_frames[purpose] = {
        'frame': frame_np,
        'timestamp': timestamp,
        'meta': last_avp_frame_metadata.copy()
    }


def get_avp_frame_for_purpose(purpose: str):
    """Return (frame, timestamp, meta) for a given purpose with fallback."""
    global last_avp_frames, last_avp_frame, last_avp_frame_timestamp, last_avp_frame_metadata
    # Prefer specific purpose
    if purpose in last_avp_frames:
        entry = last_avp_frames[purpose]
        return entry.get('frame'), entry.get('timestamp'), entry.get('meta')
    # Fallback to last general
    if last_avp_frame is not None:
        return last_avp_frame, last_avp_frame_timestamp, last_avp_frame_metadata
    return None, None, None

# Calibration buffers for intrinsics calculation
class IntrinsicsCalibBuffer:
    """Buffer for collecting ArUco detections to calculate intrinsics."""
    def __init__(self, max_frames=50):
        self.objpoints = []
        self.imgpoints = []
        self.img_size = None
        self.max_frames = max_frames

    def add(self, corners, ids, img_shape, marker_size_m, separation_m, rows, cols):
        """Add detected markers to the buffer."""
        self.img_size = (img_shape[1], img_shape[0])
        if ids is None or len(ids) == 0:
            return

        obj_rows, img_rows = [], []
        for corner, marker_id in zip(corners, ids.flatten()):
            obj = self._marker_obj_corners(marker_id, marker_size_m, separation_m, rows, cols)
            if obj is None:
                continue
            img = np.asarray(corner, dtype=np.float32).reshape(-1, 2)
            obj_rows.append(obj[:, :2])  # XY only (Z=0)
            img_rows.append(img)

        if obj_rows:
            self.objpoints.append(np.vstack(obj_rows).astype(np.float32))
            self.imgpoints.append(np.vstack(img_rows).astype(np.float32))
            # Keep only last max_frames
            self.objpoints = self.objpoints[-self.max_frames:]
            self.imgpoints = self.imgpoints[-self.max_frames:]

    def _marker_obj_corners(self, marker_id, marker_size_m, separation_m, rows, cols):
        """Get 3D corner positions for a marker in the board."""
        if marker_id < 0 or marker_id >= rows * cols:
            return None
        row, col = divmod(int(marker_id), cols)
        x0 = col * (marker_size_m + separation_m)
        y0 = row * (marker_size_m + separation_m)
        return np.array([
            [x0,                y0,                0.0],
            [x0 + marker_size_m, y0,                0.0],
            [x0 + marker_size_m, y0 + marker_size_m, 0.0],
            [x0,                y0 + marker_size_m, 0.0]
        ], dtype=np.float32)

    def ready(self, min_samples=12):
        """Check if enough samples collected."""
        return len(self.imgpoints) >= min_samples

    def calibrate(self):
        """Calculate camera intrinsics from collected samples."""
        if not self.ready():
            return None

        # Convert 2D object points to 3D
        obj3d = [np.hstack([op, np.zeros((op.shape[0], 1), np.float32)])
                 for op in self.objpoints]

        flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
        ret, K, dist, _, _ = cv2.calibrateCamera(
            objectPoints=obj3d,
            imagePoints=self.imgpoints,
            imageSize=self.img_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=flags
        )

        if ret:
            return K.astype(np.float32)
        return None

    def clear(self):
        """Clear the buffer."""
        self.objpoints.clear()
        self.imgpoints.clear()
        self.img_size = None

rs_calib_buffer = IntrinsicsCalibBuffer()
avp_calib_buffer = IntrinsicsCalibBuffer()


def initialize_api():
    """
    Initialize the API server.

    - Loads T_world_rs from extrinsics/ if exists
    - Initializes RealSenseClient and starts it
    - Initializes CoordinateManager with T_world_rs
    """
    global realsense_client, coordinate_manager

    logger.info("Initializing API server...")

    # Load existing calibration if available
    T_world_rs = None
    calibration_file = Path(CONFIG["paths"]["calibration_file"])
    if calibration_file.exists():
        try:
            T_world_rs = load_calibration(str(calibration_file))
            logger.info(f"Loaded T_world_rs from {calibration_file}")
        except Exception as e:
            logger.warning(f"Failed to load T_world_rs: {e}")

    # Create extrinsics directory if it doesn't exist
    extrinsics_dir = Path(CONFIG["paths"]["extrinsics_dir"])
    extrinsics_dir.mkdir(parents=True, exist_ok=True)

    # Initialize RealSenseClient
    try:
        realsense_client = RealSenseClient()
        realsense_client.start()
        logger.info("RealSenseClient initialized and started")
    except Exception as e:
        logger.error(f"Failed to initialize RealSenseClient: {e}")
        realsense_client = None

    # Initialize CoordinateManager
    try:
        if T_world_rs is None:
            T_world_rs = np.eye(4, dtype=np.float64)
            logger.warning("No T_world_rs calibration found. Using identity until calibrated.")
        coordinate_manager = CoordinateManager(T_world_rs=T_world_rs)
        logger.info("CoordinateManager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize CoordinateManager: {e}")
        coordinate_manager = None

    logger.info("API initialization complete")


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.

    Returns:
        JSON with status, RS connection, and calibration state
    """
    with state_lock:
        rs_connected = realsense_client is not None and realsense_client.is_running
        ensure_coordinate_manager()
        calibrated = coordinate_manager is not None and coordinate_manager.is_calibrated()

    return jsonify({
        'status': 'ok',
        'rs_connected': rs_connected,
        'calibrated': calibrated
    }), 200


@app.route('/get_rgbd_frame', methods=['GET'])
def get_rgbd_frame():
    """
    Get current RGBD frame from RealSense camera.

    Returns:
        JSON with base64-encoded RGB and depth images:
        {
            "rgb": "data:image/jpeg;base64,...",
            "depth": "data:image/jpeg;base64,...",
            "timestamp": float
        }
    """
    try:
        frame_data, using_cache = get_latest_rs_frame(allow_cache=True)
        if frame_data is None:
            return jsonify({'error': 'RealSense not connected or no frames available'}), 503

        rgb = frame_data['rgb']
        depth = frame_data['depth']
        K = frame_data['K']

        # Convert BGR to JPEG (RealSense returns BGR format)
        ok_rgb, rgb_buffer = cv2.imencode('.jpg', rgb,
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok_rgb:
            raise RuntimeError("Failed to encode RGB frame")
        rgb_b64 = base64.b64encode(rgb_buffer).decode('utf-8')

        # Convert depth to colormap for visualization
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        ok_depth, depth_buffer = cv2.imencode('.jpg', depth_colormap,
                                              [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok_depth:
            raise RuntimeError("Failed to encode depth frame")
        depth_b64 = base64.b64encode(depth_buffer).decode('utf-8')

        return jsonify({
            'rgb': f'data:image/jpeg;base64,{rgb_b64}',
            'depth': f'data:image/jpeg;base64,{depth_b64}',
            'timestamp': frame_data.get('timestamp', time.time()),
            'stale': using_cache,
            'age_seconds': None if frame_data.get('timestamp') is None else max(0.0, time.time() - frame_data['timestamp'])
        }), 200

    except Exception as e:
        logger.error(f"Error getting RGBD frame: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/rgb_frame', methods=['GET'])
def get_rgb_frame():
    """
    Return only the RGB frame (base64 JPEG) for compatibility with clients expecting /rgb_frame.
    """
    try:
        frame_data, using_cache = get_latest_rs_frame(allow_cache=True)
        if frame_data is None:
            return jsonify({'error': 'RealSense not connected or no frames available'}), 503

        rgb = frame_data['rgb']
        ok_rgb, rgb_buffer = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok_rgb:
            raise RuntimeError("Failed to encode RGB frame")
        rgb_b64 = base64.b64encode(rgb_buffer).decode('utf-8')

        return jsonify({
            'frame': f'data:image/jpeg;base64,{rgb_b64}',
            'timestamp': frame_data.get('timestamp', time.time()),
            'stale': using_cache
        }), 200
    except Exception as e:
        logger.error(f\"Error getting RGB frame: {e}\", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/get_aruco_frame', methods=['GET'])
def get_aruco_frame():
    """
    Get RGB frame with detected ArUco markers highlighted.

    Returns:
        JSON with base64-encoded RGB image with ArUco markers drawn:
        {
            "rgb": "data:image/jpeg;base64,...",
            "markers_detected": int,
            "marker_ids": [id1, id2, ...],
            "timestamp": float
        }
    """
    try:
        frame_data, using_cache = get_latest_rs_frame(allow_cache=True)
        if frame_data is None:
            return jsonify({'error': 'RealSense not connected or no frames available'}), 503

        rgb = frame_data['rgb'].copy()  # BGR format from RealSense
        K = frame_data.get('K')

        # Import ArUco detector
        from aruco_detector import ArucoDetector

        # Detect ArUco markers
        detector = ArucoDetector()
        corners, ids = detector.detect_markers(rgb)

        # Draw detected markers on the image
        if corners is not None and ids is not None:
            # Draw markers
            cv2.aruco.drawDetectedMarkers(rgb, corners, ids)

            # Draw IDs as text for better visibility
            for corner, marker_id in zip(corners, ids):
                # Get center of marker
                center = corner[0].mean(axis=0).astype(int)
                # Draw marker ID
                cv2.putText(rgb, f"ID:{marker_id[0]}",
                           tuple(center),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)

            markers_detected = len(ids)
            marker_ids = ids.flatten().tolist()

            # Add to RS calibration buffer
            rs_calib_buffer.add(
                corners, ids, rgb.shape,
                CONFIG["aruco"]["marker_size_m"],
                CONFIG["aruco"]["separation_m"],
                CONFIG["aruco"]["rows"],
                CONFIG["aruco"]["cols"]
            )

            # Try to calculate RS intrinsics
            if rs_calib_buffer.ready() and not rs_intrinsics['calculated']:
                K_calc = rs_calib_buffer.calibrate()
                if K_calc is not None:
                    rs_intrinsics['K'] = K_calc
                    rs_intrinsics['calculated'] = True
                    rs_intrinsics['method'] = 'aruco_calibration'
                    rs_intrinsics['timestamp'] = time.time()
                    logger.info(f"RS intrinsics calculated: fx={K_calc[0,0]:.1f}, fy={K_calc[1,1]:.1f}")

                    # Draw notification
                    cv2.putText(rgb, "RS Intrinsics Calculated!",
                               (10, rgb.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (0, 255, 0), 2)
        else:
            markers_detected = 0
            marker_ids = []

        # Convert BGR to JPEG
        ok_rgb, rgb_buffer = cv2.imencode('.jpg', rgb,
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok_rgb:
            raise RuntimeError("Failed to encode ArUco frame")
        rgb_b64 = base64.b64encode(rgb_buffer).decode('utf-8')

        return jsonify({
            'rgb': f'data:image/jpeg;base64,{rgb_b64}',
            'markers_detected': markers_detected,
            'marker_ids': marker_ids,
            'timestamp': time.time(),
            'intrinsics_calculated': rs_intrinsics['calculated'],
            'K': rs_intrinsics['K'].tolist() if rs_intrinsics['K'] is not None else None,
            'samples_collected': len(rs_calib_buffer.imgpoints),
            'stale': using_cache
        }), 200

    except Exception as e:
        logger.error(f"Error getting ArUco frame: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/models', methods=['GET'])
def models():
    """
    List available 3D models.

    Returns:
        JSON with list of .ply files in models/ directory
    """
    try:
        models_dir = Path(CONFIG["paths"]["models"])
        if not models_dir.exists():
            return jsonify({'models': []}), 200

        model_files = sorted([f.name for f in models_dir.glob('*.ply')])
        return jsonify({'models': model_files}), 200

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/select_model', methods=['POST'])
def select_model():
    """
    Set the active model (compat endpoint for VisionOS client).
    """
    global selected_model
    try:
        data = request.get_json() or {}
        name = data.get('model_name') or data.get('name')
        if not name:
            return jsonify({'error': 'model_name is required'}), 400
        selected_model = str(name)
        logger.info(f\"Selected model set to {selected_model}\")
        return jsonify({'success': True, 'model_name': selected_model}), 200
    except Exception as e:
        logger.error(f\"Error selecting model: {e}\", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/head_pose', methods=['POST'])
def head_pose():
    """
    Update head pose.

    Expected JSON:
        {
            "position": [x, y, z],
            "quaternion": [x, y, z, w],
            "timestamp": float
        }

    Returns:
        JSON with success status
    """
    try:
        data = request.get_json()

        # Validate input
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        if 'position' not in data or 'quaternion' not in data:
            return jsonify({'error': 'Missing position or quaternion'}), 400

        position = data['position']
        quaternion = data['quaternion']
        timestamp = data.get('timestamp', None)

        # Validate data types and dimensions
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            return jsonify({'error': 'Position must be [x, y, z]'}), 400

        if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
            return jsonify({'error': 'Quaternion must be [x, y, z, w]'}), 400

        position = np.array(position, dtype=np.float32)
        quaternion = np.array(quaternion, dtype=np.float32)

        with state_lock:
            ensure_coordinate_manager()
            # Store head pose data globally
            global last_head_pose, last_head_pose_metadata
            last_head_pose = {
                'position': position.tolist(),
                'quaternion': quaternion.tolist(),
                'timestamp': timestamp
            }

            current_time = time.time()
            last_head_pose_metadata['position'] = position.tolist()
            last_head_pose_metadata['quaternion'] = quaternion.tolist()
            last_head_pose_metadata['timestamp'] = timestamp
            last_head_pose_metadata['receive_time'] = current_time
            last_head_pose_metadata['reception_count'] += 1

            # Calculate reception rate
            if last_head_pose_metadata['last_reception_time'] is not None:
                delta_t = current_time - last_head_pose_metadata['last_reception_time']
                if delta_t > 0:
                    reception_rate = 1.0 / delta_t
                    last_head_pose_metadata['reception_rate'] = reception_rate

            last_head_pose_metadata['last_reception_time'] = current_time

            # Update CoordinateManager if available
            if coordinate_manager is not None:
                coordinate_manager.update_head_pose(position, quaternion, timestamp)

        logger.debug(f"Updated head pose - position: {position}, timestamp: {timestamp}")
        return jsonify({'success': True}), 200

    except Exception as e:
        logger.error(f"Error updating head pose: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/get_head_pose', methods=['GET'])
def get_head_pose():
    """
    Get the latest head pose data from VisionOS device.

    Returns:
        JSON with head pose information:
        {
            "position": [x, y, z],
            "quaternion": [x, y, z, w],
            "timestamp": float,
            "age_seconds": float,
            "receive_time": float,
            "reception_count": int,
            "reception_rate": float (Hz)
        }
    """
    try:
        with state_lock:
            ensure_coordinate_manager()
            if last_head_pose is None:
                return jsonify({'error': 'No head pose data available'}), 404

            pose_data = last_head_pose.copy()
            metadata = last_head_pose_metadata.copy()

        # Calculate age
        current_time = time.time()
        age = current_time - metadata['receive_time'] if metadata['receive_time'] else None

        return jsonify({
            'position': pose_data['position'],
            'quaternion': pose_data['quaternion'],
            'timestamp': pose_data['timestamp'],
            'age_seconds': age,
            'receive_time': metadata['receive_time'],
            'reception_count': metadata['reception_count'],
            'reception_rate': metadata.get('reception_rate', None)
        }), 200

    except Exception as e:
        logger.error(f"Error getting head pose: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/calibrate_rs', methods=['POST'])
def calibrate_rs():
    """
    Calibrate RealSense camera using ArUco board.

    - Captures frame from RealSense
    - Detects ArUco board
    - Saves T_world_rs calibration

    Returns:
        JSON with success, T_world_rs matrix, and reprojection error
    """
    try:
        with state_lock:
            if realsense_client is None or not realsense_client.is_running:
                return jsonify({'error': 'RealSenseClient not available'}), 500

            # Capture frame
            frame_data = realsense_client.capture()
            if frame_data is None:
                return jsonify({'error': 'Failed to capture RealSense frame'}), 500

            rgb = frame_data['rgb']
            depth = frame_data['depth']
            K = frame_data['K']

            # Detect ArUco board
            result = detect_aruco_board(rgb)
            if result is None or not result.get('success', False):
                return jsonify({'error': 'Failed to detect ArUco board'}), 400

            T_world_rs = result.get('T_world_board')
            reprojection_error = result.get('reprojection_error', 0.0)

            if T_world_rs is None:
                return jsonify({'error': 'Failed to compute transformation'}), 400

            # Save calibration
            calibration_file = Path(CONFIG["paths"]["calibration_file"])
            calibration_file.parent.mkdir(parents=True, exist_ok=True)
            save_calibration(str(calibration_file), T_world_rs)

            # Update CoordinateManager
            if coordinate_manager is not None:
                coordinate_manager.set_rs_calibration(T_world_rs)

        logger.info(f"RealSense calibration successful - reprojection error: {reprojection_error}")
        return jsonify({
            'success': True,
            'T_world_rs': T_world_rs.tolist(),
            'reprojection_error': float(reprojection_error)
        }), 200

    except Exception as e:
        logger.error(f"Error calibrating RealSense: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/calibrate_avp', methods=['POST'])
def calibrate_avp():
    """
    Calibrate AVP camera using ArUco board.

    Expected JSON:
        {
            "rgb_frame": base64_string,
            "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        }

    Returns:
        JSON with success, T_world_avp matrix
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Decode RGB frame
        if 'rgb_frame' not in data:
            return jsonify({'error': 'Missing rgb_frame'}), 400

        try:
            rgb_data = base64.b64decode(data['rgb_frame'])
            nparr = np.frombuffer(rgb_data, np.uint8)
            rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if rgb is None:
                return jsonify({'error': 'Failed to decode RGB frame'}), 400

        except Exception as e:
            logger.error(f"Error decoding RGB frame: {e}")
            return jsonify({'error': f'Failed to decode RGB frame: {e}'}), 400

        # Get camera matrix
        if 'K' not in data:
            return jsonify({'error': 'Missing camera matrix K'}), 400

        K = np.array(data['K'], dtype=np.float32)

        # Detect ArUco board in AVP view
        result = detect_aruco_board(rgb, camera_matrix=K)
        if result is None or not result.get('success', False):
            return jsonify({'error': 'Failed to detect ArUco board in AVP view'}), 400

        T_world_avp = result.get('T_world_board')
        if T_world_avp is None:
            return jsonify({'error': 'Failed to compute AVP transformation'}), 400

        # Update CoordinateManager
        with state_lock:
            if coordinate_manager is None:
                return jsonify({'error': 'CoordinateManager not initialized'}), 500

            coordinate_manager.set_avp_calibration(T_world_avp)

        logger.info("AVP calibration successful")
        return jsonify({
            'success': True,
            'T_world_avp': T_world_avp.tolist()
        }), 200

    except Exception as e:
        logger.error(f"Error calibrating AVP: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/estimate_pose', methods=['POST'])
def estimate_pose_endpoint():
    """
    Estimate object pose from AVP view.

    Expected JSON:
        {
            "rgb_frame_avp": base64_string,
            "mask": base64_string,
            "K_avp": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "model_name": "object.ply"
        }

    Process:
        1. Decode mask from base64
        2. Transform mask from AVP view to RS view
        3. Capture RealSense RGB, depth, K
        4. Estimate pose in RS view
        5. Transform pose from RS to AVP
        6. Get RS camera pose in AVP

    Returns:
        JSON with poses and debug information
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Validate required fields
        required_fields = ['rgb_frame_avp', 'mask', 'K_avp', 'model_name']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing {field}'}), 400

        # Decode mask
        try:
            mask_data = base64.b64decode(data['mask'])
            mask_avp = np.frombuffer(mask_data, np.uint8).reshape(-1)
        except Exception as e:
            logger.error(f"Error decoding mask: {e}")
            return jsonify({'error': f'Failed to decode mask: {e}'}), 400

        # Decode RGB frame
        try:
            rgb_data = base64.b64decode(data['rgb_frame_avp'])
            nparr = np.frombuffer(rgb_data, np.uint8)
            rgb_avp = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if rgb_avp is None:
                return jsonify({'error': 'Failed to decode RGB frame'}), 400

        except Exception as e:
            logger.error(f"Error decoding RGB frame: {e}")
            return jsonify({'error': f'Failed to decode RGB frame: {e}'}), 400

        K_avp = np.array(data['K_avp'], dtype=np.float32)
        model_name = str(data['model_name'])

        with state_lock:
            # Validate system state
            if realsense_client is None or not realsense_client.is_running:
                return jsonify({'error': 'RealSenseClient not available'}), 500

            if coordinate_manager is None:
                return jsonify({'error': 'CoordinateManager not initialized'}), 500

            if not coordinate_manager.is_calibrated():
                return jsonify({'error': 'System not calibrated'}), 400

            # Transform mask from AVP to RS view
            try:
                mask_rs = transform_mask_avp_to_rs(
                    mask_avp,
                    coordinate_manager.T_world_avp,
                    coordinate_manager.T_world_rs
                )
            except Exception as e:
                logger.error(f"Error transforming mask: {e}")
                return jsonify({'error': f'Failed to transform mask: {e}'}), 500

            # Capture RealSense frame
            frame_data = realsense_client.capture()
            if frame_data is None:
                return jsonify({'error': 'Failed to capture RealSense frame'}), 500

            rgb_rs = frame_data['rgb']
            depth_rs = frame_data['depth']
            K_rs = frame_data['K']

            # Estimate pose in RS view
            try:
                pose_result = estimate_pose(
                    rgb_rs,
                    mask_rs,
                    K_rs,
                    model_name,
                    depth_rs
                )
            except Exception as e:
                logger.error(f"Error estimating pose: {e}")
                return jsonify({'error': f'Failed to estimate pose: {e}'}), 500

            if pose_result is None or not pose_result.get('success', False):
                return jsonify({'error': 'Pose estimation failed'}), 400

            pose_object_in_rs = pose_result.get('pose')
            if pose_object_in_rs is None:
                return jsonify({'error': 'No pose returned from estimator'}), 400

            # Transform pose from RS to AVP
            try:
                pose_object_in_avp = coordinate_manager.transform_pose_rs_to_avp(
                    pose_object_in_rs
                )
            except Exception as e:
                logger.error(f"Error transforming pose: {e}")
                return jsonify({'error': f'Failed to transform pose: {e}'}), 500

            # Get RS camera pose in AVP
            try:
                pose_rs_in_avp = coordinate_manager.get_rs_pose_in_avp()
            except Exception as e:
                logger.error(f"Error getting RS pose in AVP: {e}")
                return jsonify({'error': f'Failed to get RS camera pose: {e}'}), 500

        # Prepare debug information
        debug_info = {
            'mask_transform_success': True,
            'pose_estimation_confidence': pose_result.get('confidence', None),
            'mask_shape': mask_rs.shape if hasattr(mask_rs, 'shape') else None,
            'rgb_rs_shape': rgb_rs.shape if rgb_rs is not None else None,
            'depth_rs_shape': depth_rs.shape if depth_rs is not None else None
        }

        logger.info(f"Pose estimation successful for {model_name}")
        return jsonify({
            'success': True,
            'pose_rs_in_avp': pose_rs_in_avp.tolist(),
            'pose_object_in_avp': pose_object_in_avp.tolist(),
            'debug': debug_info
        }), 200

    except Exception as e:
        logger.error(f"Error in pose estimation endpoint: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    """
    Receive RGB frame from UxPlay capture service.

    Expected JSON:
        {
            "rgb_frame": "data:image/jpeg;base64,...",
            "purpose": "aruco_calibration" | "roi_selection" | "general"
        }

    Stores frame in global state for later use.

    Returns:
        JSON with success status
    """
    global last_avp_frame, last_avp_frame_timestamp

    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Get frame
        frame_b64 = data.get('rgb_frame', '')
        if not frame_b64:
            return jsonify({'error': 'Missing rgb_frame'}), 400

        # Handle data URL format
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',')[1]

        try:
            frame_bytes = base64.b64decode(frame_b64)
            frame_np = cv2.imdecode(
                np.frombuffer(frame_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame_np is None:
                return jsonify({'error': 'Failed to decode frame'}), 400

        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return jsonify({'error': f'Failed to decode frame: {e}'}), 400

        # Store frame with metadata
        purpose = data.get('purpose', 'general')
        timestamp = data.get('timestamp', time.time())
        with state_lock:
            store_avp_frame(frame_np, timestamp, purpose)

        logger.debug(f"AVP frame received: {purpose}, shape: {frame_np.shape}, ts: {timestamp}")

        return jsonify({'success': True, 'timestamp': timestamp}), 200

    except Exception as e:
        logger.error(f"Error receiving frame: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/capture_frame', methods=['POST'])
def trigger_frame_capture():
    """
    Trigger UxPlay frame capture.

    Query parameters:
        purpose: "aruco_calibration" | "roi_selection" | "general"

    Process:
        1. Calls uxplay_capture.py to capture latest frame
        2. Frame is sent to /receive_frame endpoint
        3. Stored in global state

    Returns:
        JSON with success status
    """
    purpose = request.args.get('purpose', 'general')

    try:
        from uxplay_capture import UxPlayCapture
        capture = UxPlayCapture()

        success = capture.capture_and_send(purpose=purpose)

        if success:
            logger.info(f"Frame captured for {purpose}")
            return jsonify({
                "success": True,
                "message": f"Frame captured for {purpose}"
            }), 200
        else:
            fallback_used = False
            age = None
            with state_lock:
                if last_avp_frame is not None and last_avp_frame_metadata.get('receive_time'):
                    fallback_used = True
                    age = time.time() - last_avp_frame_metadata['receive_time']

            if fallback_used:
                logger.warning(f"Capture failed for {purpose}, serving cached AVP frame (age {age:.2f}s)")
                return jsonify({
                    "success": True,
                    "message": f"Used cached AVP frame (age {age:.2f}s) for {purpose}",
                    "cached": True
                }), 200

            logger.error(f"Frame capture failed for {purpose} (no cached AVP frame available)")
            return jsonify({
                "success": False,
                "error": "Frame capture failed and no cached frame available"
            }), 503

    except Exception as e:
        logger.error(f"Error triggering frame capture: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/get_avp_latest_frame', methods=['GET'])
def get_avp_latest_frame():
    """
    Get the most recent AVP frame received from UxPlay.

    Returns:
        JSON with base64-encoded RGB image and timestamp:
        {
            "rgb": "data:image/jpeg;base64,...",
            "timestamp": float,
            "age_seconds": float,
            "width": int,
            "height": int
        }
    """
    try:
        purpose = request.args.get('purpose', 'general')
        with state_lock:
            frame, timestamp, metadata = get_avp_frame_for_purpose(purpose)
            if frame is None:
                return jsonify({'error': f'No AVP frame available for {purpose}'}), 404
            frame = frame.copy()
            metadata = metadata.copy() if metadata else {}

        # Calculate age
        age = time.time() - metadata.get('receive_time', time.time())

        # Convert BGR to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rgb_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'rgb': f'data:image/jpeg;base64,{rgb_b64}',
            'timestamp': timestamp,
            'age_seconds': age,
            'width': metadata.get('width'),
            'height': metadata.get('height'),
            'purpose': purpose
        }), 200

    except Exception as e:
        logger.error(f"Error getting AVP frame: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/get_avp_aruco_frame', methods=['GET'])
def get_avp_aruco_frame():
    """
    Get AVP frame with detected ArUco markers and calculate intrinsics.

    Automatically calculates camera intrinsics when enough samples collected.

    Returns:
        JSON with annotated image and detection info:
        {
            "rgb": "data:image/jpeg;base64,...",
            "markers_detected": int,
            "marker_ids": [id1, id2, ...],
            "timestamp": float,
            "intrinsics_calculated": bool,
            "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] or null,
            "samples_collected": int
        }
    """
    global avp_intrinsics, avp_calib_buffer

    try:
        purpose = request.args.get('purpose', 'aruco_calibration')
        with state_lock:
            frame, timestamp, _ = get_avp_frame_for_purpose(purpose)
            if frame is None:
                return jsonify({'error': f'No AVP frame available for {purpose}'}), 404
            frame = frame.copy()

        # Import ArUco detector
        from aruco_detector import ArucoDetector

        # Detect ArUco markers
        detector = ArucoDetector()
        corners, ids = detector.detect_markers(frame)

        pose_matrix = None
        # Draw detected markers
        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw IDs
            for corner, marker_id in zip(corners, ids):
                center = corner[0].mean(axis=0).astype(int)
                cv2.putText(frame, f"ID:{marker_id[0]}",
                           tuple(center),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)

            markers_detected = len(ids)
            marker_ids = ids.flatten().tolist()

            # Add to calibration buffer
            avp_calib_buffer.add(
                corners, ids, frame.shape,
                CONFIG["aruco"]["marker_size_m"],
                CONFIG["aruco"]["separation_m"],
                CONFIG["aruco"]["rows"],
                CONFIG["aruco"]["cols"]
            )

            # Estimate pose
            K = avp_intrinsics['K']
            if K is None:
                K_default, dist = ArucoDetector.create_default_camera_matrix(frame.shape[1], frame.shape[0])
                cam_K = K_default
            else:
                dist = np.zeros((5, 1), dtype=np.float32)
                cam_K = K
            pose = detector.estimate_board_pose(corners, ids, cam_K, dist)
            if pose is not None:
                rvec, tvec = pose
                pose_matrix = detector.pose_to_transformation_matrix(rvec, tvec)

            # Try to calculate intrinsics
            if avp_calib_buffer.ready() and not avp_intrinsics['calculated']:
                K_calc = avp_calib_buffer.calibrate()
                if K_calc is not None:
                    avp_intrinsics['K'] = K_calc
                    avp_intrinsics['calculated'] = True
                    avp_intrinsics['method'] = 'aruco_calibration'
                    avp_intrinsics['timestamp'] = time.time()
                    logger.info(f"AVP intrinsics calculated: fx={K_calc[0,0]:.1f}, fy={K_calc[1,1]:.1f}")

                    # Draw notification
                    cv2.putText(frame, "AVP Intrinsics Calculated!",
                               (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (0, 255, 0), 2)
        else:
            markers_detected = 0
            marker_ids = []

        # Convert BGR to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rgb_b64 = base64.b64encode(buffer).decode('utf-8')

        response = {
            'rgb': f'data:image/jpeg;base64,{rgb_b64}',
            'markers_detected': markers_detected,
            'marker_ids': marker_ids,
            'timestamp': timestamp,
            'intrinsics_calculated': avp_intrinsics['calculated'],
            'K': avp_intrinsics['K'].tolist() if avp_intrinsics['K'] is not None else None,
            'samples_collected': len(avp_calib_buffer.imgpoints),
            'pose_matrix': pose_matrix.tolist() if pose_matrix is not None else None,
            'purpose': purpose
        }

        with state_lock:
            if pose_matrix is not None:
                global last_avp_board_pose
                last_avp_board_pose = pose_matrix.copy()

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error getting AVP ArUco frame: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/get_intrinsics', methods=['GET'])
def get_intrinsics():
    """
    Get calculated intrinsics for both RS and AVP cameras.

    Returns:
        JSON with intrinsics for both cameras:
        {
            "rs": {
                "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                "calculated": bool,
                "method": str,
                "timestamp": float
            },
            "avp": {
                "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                "calculated": bool,
                "method": str,
                "timestamp": float
            }
        }
    """
    try:
        # Get RS intrinsics
        rs_data = rs_intrinsics.copy()
        if rs_data['K'] is not None:
            rs_data['K'] = rs_data['K'].tolist()

        # Get AVP intrinsics
        avp_data = avp_intrinsics.copy()
        if avp_data['K'] is not None:
            avp_data['K'] = avp_data['K'].tolist()

        return jsonify({
            'rs': rs_data,
            'avp': avp_data
        }), 200

    except Exception as e:
        logger.error(f"Error getting intrinsics: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/get_transformation', methods=['GET'])
def get_transformation():
    """
    Get transformation matrix between AVP and RS cameras.

    Requires both cameras to be calibrated with ArUco board visible in both.

    Returns:
        JSON with transformation:
        {
            "T_avp_rs": 4x4 transformation matrix,
            "calibrated": bool,
            "timestamp": float
        }
    """
    try:
        with state_lock:
            ensure_coordinate_manager()
            if coordinate_manager is None:
                return jsonify({'error': 'CoordinateManager not initialized'}), 500

            # Get transformation from world to both cameras
            T_world_rs = coordinate_manager.T_world_rs
            T_world_avp = coordinate_manager.T_world_avp

            if T_world_avp is None:
                return jsonify({
                    'calibrated': False,
                    'T_avp_rs': np.eye(4).tolist(),
                    'message': 'System not calibrated. Detect ArUco board in both cameras.',
                    'T_world_rs': T_world_rs.tolist(),
                    'T_world_avp': None,
                    'timestamp': time.time()
                }), 200

            # Calculate T_avp_rs = T_avp_world * T_world_rs
            # T_avp_world = inv(T_world_avp)
            T_avp_world = np.linalg.inv(T_world_avp)
            T_avp_rs = T_avp_world @ T_world_rs

        return jsonify({
            'calibrated': True,
            'T_avp_rs': T_avp_rs.tolist(),
            'T_world_rs': T_world_rs.tolist(),
            'T_world_avp': T_world_avp.tolist(),
            'timestamp': time.time()
        }), 200

    except Exception as e:
        logger.error(f"Error getting transformation: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/get_rs_pose_in_avp', methods=['GET'])
def get_rs_pose_in_avp():
    """
    Get RealSense camera pose in AVP coordinate frame.

    Computes the position and orientation of the RS camera relative to the AVP
    (head-mounted) coordinate frame. This is used to visualize the RS camera
    location in the VisionOS immersive space.

    Returns:
        JSON with RS camera pose in AVP frame:
        {
            "position": [x, y, z] in meters,
            "quaternion": [x, y, z, w],
            "T_avp_rs": 4x4 transformation matrix,
            "calibrated": bool,
            "head_pose_age": float (seconds since last head pose update),
            "timestamp": float
        }
    """
    try:
        with state_lock:
            ensure_coordinate_manager()
            if coordinate_manager is None:
                return jsonify({'error': 'CoordinateManager not initialized'}), 500

            if not coordinate_manager.is_calibrated():
                return jsonify({
                    'calibrated': False,
                    'position': None,
                    'quaternion': None,
                    'T_avp_rs': np.eye(4).tolist(),
                    'message': 'System not calibrated. Perform ArUco calibration on both cameras.'
                }), 200

            # Get transformation from world to both cameras
            T_world_rs = coordinate_manager.T_world_rs
            T_world_avp = coordinate_manager.T_world_avp

            if T_world_rs is None or T_world_avp is None:
                return jsonify({
                    'calibrated': False,
                    'position': None,
                    'quaternion': None,
                    'T_avp_rs': np.eye(4).tolist(),
                    'message': 'Missing calibration data'
                }), 200

            # Calculate T_avp_rs = inv(T_world_avp) @ T_world_rs
            # This gives the transformation from RS frame to AVP frame
            T_avp_world = np.linalg.inv(T_world_avp)
            T_avp_rs = T_avp_world @ T_world_rs

            # Extract position (translation vector from 4x4 matrix)
            position = T_avp_rs[:3, 3].tolist()

            # Extract rotation matrix and convert to quaternion
            R = T_avp_rs[:3, :3]

            # Convert rotation matrix to quaternion (w, x, y, z)
            # Using Shepperd's method for numerical stability
            trace = np.trace(R)
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                w = 0.25 / s
                x = (R[2, 1] - R[1, 2]) * s
                y = (R[0, 2] - R[2, 0]) * s
                z = (R[1, 0] - R[0, 1]) * s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

            # Return quaternion in [x, y, z, w] format (VisionOS convention)
            quaternion = [x, y, z, w]

            # Check head pose age
            head_pose_age = None
            if last_head_pose_metadata['receive_time'] is not None:
                head_pose_age = time.time() - last_head_pose_metadata['receive_time']

        return jsonify({
            'calibrated': True,
            'position': position,
            'quaternion': quaternion,
            'T_avp_rs': T_avp_rs.tolist(),
            'head_pose_age': head_pose_age,
            'timestamp': time.time()
        }), 200

    except Exception as e:
        logger.error(f"Error getting RS pose in AVP: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/get_transformed_depth', methods=['GET'])
def get_transformed_depth():
    """
    Transform RealSense depth map to AVP view.

    Process:
        1. Capture RS depth map and K_rs
        2. Get T_avp_rs transformation from coordinate_manager
        3. Create point cloud from RS depth + K_rs
        4. Transform point cloud to AVP frame
        5. Project to AVP image plane using K_avp
        6. Generate depth colormap in AVP view
        7. Return as base64 JPEG

    Query parameters:
        colormap (optional): OpenCV colormap to use (default: COLORMAP_JET)

    Returns:
        JSON with base64-encoded depth visualization
    """
    try:
        colormap_name = request.args.get('colormap', 'COLORMAP_JET')
        colormap = getattr(cv2, colormap_name, cv2.COLORMAP_JET)

        frame_data, using_cache = get_latest_rs_frame(allow_cache=True)
        if frame_data is None:
            return jsonify({'error': 'RealSense not connected or no frames available'}), 503

        depth_rs = frame_data['depth']
        K_rs = frame_data['K']

        with state_lock:
            ensure_coordinate_manager()
            if coordinate_manager is None:
                return jsonify({'error': 'CoordinateManager not initialized'}), 500

            if not coordinate_manager.is_calibrated():
                depth_normalized = cv2.normalize(depth_rs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_normalized, colormap)
                _, buffer = cv2.imencode('.jpg', depth_colormap, [cv2.IMWRITE_JPEG_QUALITY, 85])
                depth_b64 = base64.b64encode(buffer).decode('utf-8')

                return jsonify({
                    'depth_colormap': f'data:image/jpeg;base64,{depth_b64}',
                    'timestamp': frame_data.get('timestamp', time.time()),
                    'transformation_applied': False,
                    'message': 'System not calibrated - returning RS depth view',
                    'min_depth': float(np.min(depth_rs[depth_rs > 0])) if np.any(depth_rs > 0) else 0.0,
                    'max_depth': float(np.max(depth_rs))
                }), 200

            T_avp_rs = coordinate_manager.get_T_avp_rs()
            K_avp = avp_intrinsics['K']

            if K_avp is None:
                return jsonify({'error': 'AVP intrinsics not calculated yet'}), 400

        h_rs, w_rs = depth_rs.shape
        u, v = np.meshgrid(np.arange(w_rs), np.arange(h_rs))
        u = u.flatten()
        v = v.flatten()
        z = depth_rs.flatten()

        valid_mask = z > 0
        u = u[valid_mask]
        v = v[valid_mask]
        z = z[valid_mask]

        if len(z) == 0:
            return jsonify({'error': 'No valid depth data'}), 400

        fx_rs, fy_rs = K_rs[0, 0], K_rs[1, 1]
        cx_rs, cy_rs = K_rs[0, 2], K_rs[1, 2]

        X_rs = (u - cx_rs) * z / fx_rs
        Y_rs = (v - cy_rs) * z / fy_rs
        Z_rs = z

        points_rs = np.vstack([X_rs, Y_rs, Z_rs, np.ones_like(Z_rs)])
        points_avp = T_avp_rs @ points_rs

        X_avp = points_avp[0, :]
        Y_avp = points_avp[1, :]
        Z_avp = points_avp[2, :]

        valid_depth = Z_avp > 0
        X_avp = X_avp[valid_depth]
        Y_avp = Y_avp[valid_depth]
        Z_avp = Z_avp[valid_depth]

        if len(Z_avp) == 0:
            return jsonify({'error': 'No points visible in AVP view'}), 400

        fx_avp, fy_avp = K_avp[0, 0], K_avp[1, 1]
        cx_avp, cy_avp = K_avp[0, 2], K_avp[1, 2]

        u_avp = (X_avp * fx_avp / Z_avp) + cx_avp
        v_avp = (Y_avp * fy_avp / Z_avp) + cy_avp

        with state_lock:
            frame, _, metadata = get_avp_frame_for_purpose('general')
            if frame is None:
                h_avp = int(2 * cy_avp)
                w_avp = int(2 * cx_avp)
            else:
                h_avp, w_avp = frame.shape[:2]

        in_bounds = (u_avp >= 0) & (u_avp < w_avp) & (v_avp >= 0) & (v_avp < h_avp)
        u_avp = u_avp[in_bounds]
        v_avp = v_avp[in_bounds]
        Z_avp = Z_avp[in_bounds]

        if len(Z_avp) == 0:
            return jsonify({'error': 'No points project into AVP image bounds'}), 400

        depth_avp = np.zeros((h_avp, w_avp), dtype=np.float32)
        u_int = u_avp.astype(np.int32)
        v_int = v_avp.astype(np.int32)

        for i in range(len(u_int)):
            depth_avp[v_int[i], u_int[i]] = max(depth_avp[v_int[i], u_int[i]], Z_avp[i])

        mask = (depth_avp > 0).astype(np.uint8)
        depth_avp_filled = cv2.inpaint(
            (depth_avp * 1000).astype(np.uint16),
            1 - mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_NS
        ).astype(np.float32) / 1000.0

        depth_normalized = cv2.normalize(depth_avp_filled, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_normalized, colormap)

        _, buffer = cv2.imencode('.jpg', depth_colormap, [cv2.IMWRITE_JPEG_QUALITY, 85])
        depth_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'depth_colormap': f'data:image/jpeg;base64,{depth_b64}',
            'timestamp': frame_data.get('timestamp', time.time()),
            'transformation_applied': True,
            'min_depth': float(np.min(Z_avp)),
            'max_depth': float(np.max(Z_avp)),
            'num_points': int(len(Z_avp)),
            'stale': using_cache
        }), 200

    except Exception as e:
        logger.error(f"Error in get_transformed_depth: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/get_roi_rgb', methods=['GET'])
def get_roi_rgb():
    """
    Get ROI (Region of Interest) RGB image from AVP frame.

    Query parameters:
        x, y, width, height: ROI bounds
        purpose: Frame purpose (default: 'general')

    Returns:
        JSON with base64-encoded cropped RGB image
    """
    try:
        purpose = request.args.get('purpose', 'general')

        with state_lock:
            frame, timestamp, metadata = get_avp_frame_for_purpose(purpose)
            if frame is None:
                return jsonify({'error': f'No AVP frame available for {purpose}'}), 404
            frame = frame.copy()

        h, w = frame.shape[:2]

        x = int(request.args.get('x', 0))
        y = int(request.args.get('y', 0))
        width = int(request.args.get('width', w))
        height = int(request.args.get('height', h))

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        width = max(1, min(width, w - x))
        height = max(1, min(height, h - y))

        roi = frame[y:y+height, x:x+width]

        if roi.size == 0:
            return jsonify({'error': 'Invalid ROI - empty region'}), 400

        _, buffer = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 90])
        roi_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'roi_rgb': f'data:image/jpeg;base64,{roi_b64}',
            'roi_x': x,
            'roi_y': y,
            'roi_width': width,
            'roi_height': height,
            'original_width': w,
            'original_height': h,
            'timestamp': timestamp,
            'purpose': purpose
        }), 200

    except ValueError as e:
        logger.error(f"Invalid parameter in get_roi_rgb: {e}")
        return jsonify({'error': f'Invalid parameter: {e}'}), 400
    except Exception as e:
        logger.error(f"Error in get_roi_rgb: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/get_roi_binary_mask', methods=['POST'])
def get_roi_binary_mask():
    """
    Apply HSV color filter to ROI and generate binary mask.

    Expected JSON:
        {
            "x": int, "y": int, "width": int, "height": int,
            "hsv_lower": [h, s, v], "hsv_upper": [h, s, v],
            "purpose": str (optional)
        }

    Returns:
        JSON with base64-encoded binary mask as PNG
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        required_fields = ['x', 'y', 'width', 'height', 'hsv_lower', 'hsv_upper']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        purpose = data.get('purpose', 'general')
        x = int(data['x'])
        y = int(data['y'])
        width = int(data['width'])
        height = int(data['height'])
        hsv_lower = np.array(data['hsv_lower'], dtype=np.uint8)
        hsv_upper = np.array(data['hsv_upper'], dtype=np.uint8)

        if hsv_lower.shape != (3,) or hsv_upper.shape != (3,):
            return jsonify({'error': 'HSV bounds must be arrays of length 3'}), 400

        with state_lock:
            frame, timestamp, metadata = get_avp_frame_for_purpose(purpose)
            if frame is None:
                return jsonify({'error': f'No AVP frame available for {purpose}'}), 404
            frame = frame.copy()

        h, w = frame.shape[:2]

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        width = max(1, min(width, w - x))
        height = max(1, min(height, h - y))

        roi_bgr = frame[y:y+height, x:x+width]

        if roi_bgr.size == 0:
            return jsonify({'error': 'Invalid ROI - empty region'}), 400

        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        binary_mask = cv2.inRange(roi_hsv, hsv_lower, hsv_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        mask_pixels = int(np.sum(binary_mask > 0))
        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        coverage = (mask_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

        _, buffer = cv2.imencode('.png', binary_mask)
        mask_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'binary_mask': f'data:image/png;base64,{mask_b64}',
            'roi_x': x,
            'roi_y': y,
            'roi_width': width,
            'roi_height': height,
            'mask_pixels': mask_pixels,
            'total_pixels': total_pixels,
            'coverage': float(coverage),
            'original_width': w,
            'original_height': h,
            'timestamp': timestamp,
            'purpose': purpose,
            'hsv_lower': hsv_lower.tolist(),
            'hsv_upper': hsv_upper.tolist()
        }), 200

    except ValueError as e:
        logger.error(f"Invalid parameter in get_roi_binary_mask: {e}")
        return jsonify({'error': f'Invalid parameter: {e}'}), 400
    except Exception as e:
        logger.error(f"Error in get_roi_binary_mask: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/foundation_pose_request', methods=['POST'])
def foundation_pose_request():
    """
    Forward FoundationPose request to FoundationPose API.

    Expected JSON:
        {
            "roi_rgb": base64_string (JPEG),
            "transformed_depth": base64_string (PNG disparity),
            "avp_intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "mask": base64_string (PNG),
            "mesh_path": "path/to/mesh.ply"
        }

    Returns:
        JSON with pose directly in AVP frame (no transformation needed)
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        required_fields = ['roi_rgb', 'transformed_depth', 'avp_intrinsics', 'mask', 'mesh_path']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing {field}'}), 400

        roi_rgb_b64 = data['roi_rgb']
        transformed_depth_b64 = data['transformed_depth']
        K_avp = np.array(data['avp_intrinsics'], dtype=np.float32)
        mask_b64 = data['mask']
        mesh_path = data['mesh_path']

        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(CONFIG["paths"]["models_dir"], mesh_path)

        if not os.path.exists(mesh_path):
            return jsonify({'error': f'Mesh file not found: {mesh_path}'}), 400

        try:
            if ',' in roi_rgb_b64:
                roi_rgb_b64 = roi_rgb_b64.split(',')[1]
            rgb_data = base64.b64decode(roi_rgb_b64)
            nparr = np.frombuffer(rgb_data, np.uint8)
            roi_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if roi_rgb is None:
                return jsonify({'error': 'Failed to decode ROI RGB'}), 400
        except Exception as e:
            logger.error(f"Error decoding ROI RGB: {e}")
            return jsonify({'error': f'Failed to decode ROI RGB: {e}'}), 400

        try:
            if ',' in transformed_depth_b64:
                transformed_depth_b64 = transformed_depth_b64.split(',')[1]
            depth_data = base64.b64decode(transformed_depth_b64)
            nparr = np.frombuffer(depth_data, np.uint8)
            transformed_depth = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if transformed_depth is None:
                return jsonify({'error': 'Failed to decode transformed depth'}), 400

            transformed_depth = transformed_depth.astype(np.float32)
        except Exception as e:
            logger.error(f"Error decoding transformed depth: {e}")
            return jsonify({'error': f'Failed to decode transformed depth: {e}'}), 400

        try:
            if ',' in mask_b64:
                mask_b64 = mask_b64.split(',')[1]
            mask_data = base64.b64decode(mask_b64)
            nparr = np.frombuffer(mask_data, np.uint8)
            mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                return jsonify({'error': 'Failed to decode mask'}), 400
        except Exception as e:
            logger.error(f"Error decoding mask: {e}")
            return jsonify({'error': f'Failed to decode mask: {e}'}), 400

        try:
            from foundationpose_client import estimate_pose

            foundationpose_url = CONFIG["network"]["foundationpose_url"]

            pose_result = estimate_pose(
                rgb=roi_rgb,
                depth=transformed_depth,
                mask=mask,
                K=K_avp,
                mesh_path=mesh_path,
                api_url=foundationpose_url
            )

            if pose_result is None:
                return jsonify({
                    'success': False,
                    'error': 'FoundationPose API returned no result'
                }), 500

            return jsonify({
                'success': True,
                'pose_avp': pose_result.tolist(),
                'confidence': 1.0
            }), 200

        except Exception as e:
            logger.error(f"Error calling FoundationPose: {e}", exc_info=True)
            return jsonify({'error': f'FoundationPose failed: {e}'}), 500

    except Exception as e:
        logger.error(f"Error in foundation_pose_request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/transform_depth_rs_to_avp', methods=['POST'])
def transform_depth_rs_to_avp():
    """
    Transform depth map from RealSense view to AVP view.

    Expected JSON:
        {
            "K_avp": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "target_width": int,
            "target_height": int
        }

    Returns:
        JSON with transformed depth array in AVP view
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        if 'K_avp' not in data:
            return jsonify({'error': 'Missing K_avp'}), 400

        K_avp = np.array(data['K_avp'], dtype=np.float32)
        target_width = data.get('target_width', 640)
        target_height = data.get('target_height', 480)

        with state_lock:
            if realsense_client is None or not realsense_client.is_running:
                return jsonify({'error': 'RealSenseClient not available'}), 500

            if coordinate_manager is None or not coordinate_manager.is_calibrated():
                return jsonify({'error': 'System not calibrated'}), 400

            frame_data, using_cache = get_latest_rs_frame(allow_cache=True)
            if frame_data is None:
                return jsonify({'error': 'Failed to capture RealSense frame'}), 500

            depth_rs = frame_data['depth']
            K_rs = frame_data['K']
            T_avp_rs = coordinate_manager.get_T_avp_rs()

            h_rs, w_rs = depth_rs.shape
            u, v = np.meshgrid(np.arange(w_rs), np.arange(h_rs))
            u = u.flatten()
            v = v.flatten()
            z = depth_rs.flatten()

            valid_mask = z > 0.01
            u = u[valid_mask]
            v = v[valid_mask]
            z = z[valid_mask]

            if len(z) == 0:
                return jsonify({'error': 'No valid depth data'}), 400

            fx_rs, fy_rs = K_rs[0, 0], K_rs[1, 1]
            cx_rs, cy_rs = K_rs[0, 2], K_rs[1, 2]

            X_rs = (u - cx_rs) * z / fx_rs
            Y_rs = (v - cy_rs) * z / fy_rs
            Z_rs = z

            points_rs = np.vstack([X_rs, Y_rs, Z_rs, np.ones_like(Z_rs)])
            points_avp = T_avp_rs @ points_rs

            X_avp = points_avp[0, :]
            Y_avp = points_avp[1, :]
            Z_avp = points_avp[2, :]

            valid_depth = Z_avp > 0.01
            X_avp = X_avp[valid_depth]
            Y_avp = Y_avp[valid_depth]
            Z_avp = Z_avp[valid_depth]

            if len(Z_avp) == 0:
                return jsonify({'error': 'No points visible in AVP view'}), 400

            fx_avp, fy_avp = K_avp[0, 0], K_avp[1, 1]
            cx_avp, cy_avp = K_avp[0, 2], K_avp[1, 2]

            u_avp = (X_avp * fx_avp / Z_avp) + cx_avp
            v_avp = (Y_avp * fy_avp / Z_avp) + cy_avp

            in_bounds = (u_avp >= 0) & (u_avp < target_width) & (v_avp >= 0) & (v_avp < target_height)
            u_avp = u_avp[in_bounds]
            v_avp = v_avp[in_bounds]
            Z_avp = Z_avp[in_bounds]

            if len(Z_avp) == 0:
                return jsonify({'error': 'No points project into AVP image bounds'}), 400

            depth_avp = np.zeros((target_height, target_width), dtype=np.float32)
            u_int = u_avp.astype(np.int32)
            v_int = v_avp.astype(np.int32)

            for i in range(len(u_int)):
                if depth_avp[v_int[i], u_int[i]] == 0 or Z_avp[i] < depth_avp[v_int[i], u_int[i]]:
                    depth_avp[v_int[i], u_int[i]] = Z_avp[i]

            mask = (depth_avp > 0).astype(np.uint8)
            depth_avp_filled = cv2.inpaint(
                (depth_avp * 1000).astype(np.uint16),
                1 - mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_NS
            ).astype(np.float32) / 1000.0

        try:
            disparity = 1.0 / (depth_avp_filled + 1e-6)
            disparity[np.isinf(disparity)] = 0

            disparity_min = np.min(disparity[disparity > 0]) if np.any(disparity > 0) else 1.0
            disparity_max = np.max(disparity)

            if disparity_max - disparity_min < 1e-6:
                disparity_normalized = np.zeros_like(disparity, dtype=np.uint8)
            else:
                disparity_normalized = (
                    (disparity - disparity_min) / (disparity_max - disparity_min) * 255
                ).astype(np.uint8)

            success, encoded = cv2.imencode('.png', disparity_normalized)
            if not success:
                return jsonify({'error': 'Failed to encode transformed depth'}), 500

            depth_b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

            return jsonify({
                'success': True,
                'transformed_depth': f'data:image/png;base64,{depth_b64}',
                'shape': [target_height, target_width],
                'stale': using_cache
            }), 200

        except Exception as e:
            logger.error(f"Error encoding transformed depth: {e}", exc_info=True)
            return jsonify({'error': f'Failed to encode depth: {e}'}), 500

    except Exception as e:
        logger.error(f"Error in transform_depth_rs_to_avp: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request errors."""
    return jsonify({'error': 'Bad request'}), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


def shutdown_api():
    """
    Shutdown the API server gracefully.

    Stops RealSenseClient and cleans up resources.
    """
    global realsense_client

    logger.info("Shutting down API server...")

    if realsense_client is not None:
        try:
            realsense_client.stop()
            logger.info("RealSenseClient stopped")
        except Exception as e:
            logger.error(f"Error stopping RealSenseClient: {e}")

    logger.info("API shutdown complete")


if __name__ == '__main__':
    try:
        # Initialize the API
        initialize_api()

        # Run the Flask server
        # Note: use_reloader=False prevents double initialization of hardware
        app.run(
            host=CONFIG["network"]["main_api_host"],
            port=CONFIG["network"]["main_api_port"],
            debug=True,
            use_reloader=False  # Must be False to prevent camera re-initialization
        )

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

    finally:
        # Cleanup
        shutdown_api()
