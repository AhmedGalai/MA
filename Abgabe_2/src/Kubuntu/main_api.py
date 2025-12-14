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

# UxPlay frame storage
last_avp_frame = None
last_avp_frame_timestamp = None


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
    extrinsics_dir = Path(CONFIG["paths"]["extrinsics_dir"])
    extrinsics_dir.mkdir(parents=True, exist_ok=True)
    extrinsics_path = extrinsics_dir
    if extrinsics_path.exists():
        try:
            T_world_rs = load_calibration(str(extrinsics_path))
            logger.info("Loaded T_world_rs from extrinsics")
        except Exception as e:
            logger.warning(f"Failed to load T_world_rs: {e}")

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
        calibrated = coordinate_manager is not None and coordinate_manager.is_calibrated()

    return jsonify({
        'status': 'ok',
        'rs_connected': rs_connected,
        'calibrated': calibrated
    }), 200


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
            if coordinate_manager is None:
                return jsonify({'error': 'CoordinateManager not initialized'}), 500

            coordinate_manager.update_head_pose(position, quaternion, timestamp)

        logger.info(f"Updated head pose - position: {position}, timestamp: {timestamp}")
        return jsonify({'success': True}), 200

    except Exception as e:
        logger.error(f"Error updating head pose: {e}", exc_info=True)
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
            rgb, depth, K = realsense_client.get_frame()
            if rgb is None:
                return jsonify({'error': 'Failed to capture RealSense frame'}), 500

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
            rgb_rs, depth_rs, K_rs = realsense_client.get_frame()
            if rgb_rs is None:
                return jsonify({'error': 'Failed to capture RealSense frame'}), 500

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

        # Store frame
        with state_lock:
            last_avp_frame = frame_np
            last_avp_frame_timestamp = time.time()

        # Get purpose (for logging)
        purpose = data.get('purpose', 'general')
        logger.info(f"Frame received for purpose: {purpose}, shape: {frame_np.shape}")

        return jsonify({'success': True}), 200

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
            logger.error(f"Frame capture failed for {purpose}")
            return jsonify({
                "success": False,
                "error": "Frame capture failed"
            }), 500

    except Exception as e:
        logger.error(f"Error triggering frame capture: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
        app.run(
            host=CONFIG["network"]["main_api_host"],
            port=CONFIG["network"]["main_api_port"],
            debug=CONFIG.get("debug", False)
        )

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

    finally:
        # Cleanup
        shutdown_api()
