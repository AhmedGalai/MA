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
        logger.error(f"Error getting RGB frame: {e}", exc_info=True)
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
        logger.info(f"Selected model set to {selected_model}")
        return jsonify({'success': True, 'model_name': selected_model}), 200
    except Exception as e:
        logger.error(f"Error selecting model: {e}", exc_info=True)
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
