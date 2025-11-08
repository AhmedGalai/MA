#!/usr/bin/env python3
"""
AVP API - ArUco Vision Processing API
Receives RGB frames and processes them through a pipeline:
1. Receive RGB frames from external capture program
2. Detect ArUco patterns
3. Find pose
4. Detect ROI mask (without active contours)

Provides endpoints to:
- Receive frames from capture program
- Configure processing parameters
- Get camera intrinsics
- Get pose of the pattern board
- Get mask image
- Get processed frames
- Receive/send head pose data
"""

import numpy as np
import cv2 as cv
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import threading
import time
from dataclasses import dataclass
import os
import requests
from typing import List, Optional

app = Flask(__name__)
CORS(app)

# ------------------ Configuration ------------------
MODELS_DIR = "../full_project_python/models"  # Folder containing .ply files
POSE_FORWARD_URL = "http://localhost:9000/pose"  # Forward to pose estimation API

# ------------------ Helper Functions ------------------
def list_ply_models(folder):
    """List all .ply files in the models directory"""
    try:
        if not os.path.exists(folder):
            print(f"[WARNING] Models directory not found: {folder}")
            return []
        return [f for f in os.listdir(folder) if f.lower().endswith(".ply")]
    except Exception as e:
        print(f"[ERROR] list_ply_models: {e}")
        return []

def load_mesh_as_b64(path: str) -> str:
    """Load a .ply file and encode as base64"""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        print(f"[ERROR] load_mesh_as_b64: {e}")
        raise

# ------------------ Screen Capture ------------------
@dataclass
class CaptureConfig:
    left: int = 934
    top: int = 100
    width: int = 812
    height: int = 1080
    fps: int = 30
    enabled: bool = False

# ------------------ ArUco Configuration ------------------
def get_aruco_handles():
    if not hasattr(cv, "aruco"):
        print("[ERROR] cv2.aruco not available; detection disabled")
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

def create_roi_mask(frame_bgr, hsv_center, h_tol=12, s_tol=50, v_tol=50):
    """Create ROI mask based on HSV color range"""
    h, s, v = hsv_center
    # Convert tolerances to HSV ranges
    dh = int(round((h_tol / 360.0) * 179.0))
    ds = int(round((s_tol / 100.0) * 255.0))
    dv = int(round((v_tol / 100.0) * 255.0))

    lo = np.array([max(0, h - dh), max(0, s - ds), max(0, v - dv)], dtype=np.uint8)
    hi = np.array([min(179, h + dh), min(255, s + ds), min(255, v + dv)], dtype=np.uint8)

    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lo, hi)
    return mask

# ------------------ Helper Functions ------------------
def encode_image_to_base64(img_bgr):
    """Encode numpy array (BGR) to base64 string"""
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# ------------------ Global State ------------------
class ProcessorState:
    def __init__(self):
        self.lock = threading.Lock()

        # Capture configuration
        self.capture_config = CaptureConfig()

        # HSV settings
        self.hsv_center = [90, 128, 128]
        self.tolerances = {'h': 12, 's': 50, 'v': 50}

        # Processed results
        self.last_frame = None
        self.last_intrinsics = None
        self.last_pose = None
        self.last_mask = None

        # Head pose data (from external AVP)
        self.last_head_pose = None
        self.head_pose_timestamp = None

        # Model selection (for AVP pose estimation)
        self.selected_model = None

        # ArUco
        self.aruco_dict, self.detector, self.api = get_aruco_handles()

        # Stats
        self.frame_count = 0
        self.capture_thread = None
        self.capture_alive = False

state = ProcessorState()

# ------------------ Frame Processing ------------------
def decode_base64_image(base64_str):
    """Decode base64 string to numpy array (BGR)"""
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    img_rgb = np.array(img)
    img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
    return img_bgr

def process_frame(frame_bgr):
    """Process a single frame through the pipeline"""
    try:
        # Get current settings
        with state.lock:
            hsv_center = state.hsv_center.copy()
            tolerances = state.tolerances.copy()

        # Get frame dimensions
        h, w = frame_bgr.shape[:2]
        K, dist = default_K_for_size(w, h)

        # Detect ArUco markers
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        corners, ids, rejected = None, None, None

        if state.aruco_dict is not None:
            if state.api == "new":
                corners, ids, rejected = state.detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv.aruco.detectMarkers(
                    gray, state.aruco_dict, parameters=state.detector
                )

        # Solve pose
        pose = None
        if ids is not None and len(ids) > 0:
            result = solve_board_pose(corners, ids, K, dist)
            if result is not None:
                rvec, tvec = result
                pose = {
                    "rvec": rvec.tolist(),
                    "tvec": tvec.tolist(),
                    "markers_detected": len(ids)
                }

        # Create ROI mask
        mask = create_roi_mask(
            frame_bgr,
            hsv_center,
            tolerances['h'],
            tolerances['s'],
            tolerances['v']
        )

        # Update state
        with state.lock:
            state.last_frame = frame_bgr
            state.last_intrinsics = {"K": K.tolist(), "dist": dist.tolist()}
            state.last_pose = pose
            state.last_mask = mask
            state.frame_count += 1

        return True

    except Exception as e:
        print(f"[ERROR] Processing frame: {e}")
        return False

# ------------------ API Endpoints ------------------
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    with state.lock:
        return jsonify({
            "status": "ok",
            "frames_processed": state.frame_count
        })

@app.route('/config', methods=['GET', 'POST'])
def config():
    """Get or set capture configuration"""
    if request.method == 'GET':
        with state.lock:
            return jsonify({
                "left": state.capture_config.left,
                "top": state.capture_config.top,
                "width": state.capture_config.width,
                "height": state.capture_config.height,
                "fps": state.capture_config.fps,
                "enabled": state.capture_config.enabled,
                "hsv_center": state.hsv_center,
                "tolerances": state.tolerances
            })

    elif request.method == 'POST':
        data = request.json
        with state.lock:
            if 'left' in data:
                state.capture_config.left = int(data['left'])
            if 'top' in data:
                state.capture_config.top = int(data['top'])
            if 'width' in data:
                state.capture_config.width = int(data['width'])
            if 'height' in data:
                state.capture_config.height = int(data['height'])
            if 'fps' in data:
                state.capture_config.fps = int(data['fps'])
            if 'hsv_center' in data:
                state.hsv_center = data['hsv_center']
            if 'tolerances' in data:
                state.tolerances = data['tolerances']

        return jsonify({"success": True})

@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    """Receive RGB frame from external capture program"""
    try:
        data = request.json
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame data provided"}), 400

        # Decode frame
        frame_bgr = decode_base64_image(data['frame'])

        # Process frame through pipeline
        success = process_frame(frame_bgr)

        if success:
            return jsonify({"success": True, "frame_count": state.frame_count})
        else:
            return jsonify({"error": "Frame processing failed"}), 500

    except Exception as e:
        print(f"[ERROR] receive_frame: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/intrinsics', methods=['GET'])
def get_intrinsics():
    """Get camera intrinsics matrix"""
    with state.lock:
        if state.last_intrinsics is None:
            return jsonify({"error": "No intrinsics available"}), 404
        return jsonify(state.last_intrinsics)

@app.route('/pose', methods=['GET'])
def get_pose():
    """Get current board pose (rvec, tvec)"""
    with state.lock:
        if state.last_pose is None:
            return jsonify({"error": "No pose available"}), 404
        return jsonify(state.last_pose)

@app.route('/mask', methods=['GET'])
def get_mask():
    """Get ROI mask as base64 encoded image"""
    with state.lock:
        if state.last_mask is None:
            return jsonify({"error": "No mask available"}), 404

        # Convert mask to 3-channel BGR for encoding
        mask_bgr = cv.cvtColor(state.last_mask, cv.COLOR_GRAY2BGR)
        mask_base64 = encode_image_to_base64(mask_bgr)
        return jsonify({"mask": mask_base64})

@app.route('/rgb_frame', methods=['GET'])
def get_rgb_frame():
    """Get the last RGB frame without any processing overlays"""
    with state.lock:
        if state.last_frame is None:
            return jsonify({"error": "No frame available"}), 404
        frame_base64 = encode_image_to_base64(state.last_frame)
        return jsonify({"frame": frame_base64})

@app.route('/detected_frame', methods=['GET'])
def get_detected_frame():
    """Get the last frame with ArUco markers drawn"""
    with state.lock:
        if state.last_frame is None:
            return jsonify({"error": "No frame available"}), 404

        frame = state.last_frame.copy()

        # Detect and draw markers
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, rejected = None, None, None

        if state.aruco_dict is not None:
            if state.api == "new":
                corners, ids, rejected = state.detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv.aruco.detectMarkers(
                    gray, state.aruco_dict, parameters=state.detector
                )

        # Draw detected markers
        if ids is not None and len(ids) > 0:
            try:
                cv.aruco.drawDetectedMarkers(frame, corners, ids)
            except Exception:
                for c in corners:
                    pts = np.asarray(c, dtype=np.int32).reshape(-1, 2)
                    for i in range(4):
                        cv.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]), (0, 255, 255), 2)

        frame_base64 = encode_image_to_base64(frame)
        return jsonify({"frame": frame_base64, "markers_detected": len(ids) if ids is not None else 0})

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get processing statistics"""
    with state.lock:
        return jsonify({
            "frames_processed": state.frame_count,
            "has_intrinsics": state.last_intrinsics is not None,
            "has_pose": state.last_pose is not None,
            "has_mask": state.last_mask is not None,
            "has_head_pose": state.last_head_pose is not None
        })

@app.route('/head_pose', methods=['GET', 'POST'])
def head_pose():
    """Send or receive head pose data"""
    if request.method == 'POST':
        # Receive head pose data from AVP
        try:
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400

            with state.lock:
                state.last_head_pose = {
                    "position": data.get("position", [0, 0, 0]),
                    "rotation": data.get("rotation", [0, 0, 0]),
                    "quaternion": data.get("quaternion", [0, 0, 0, 1]),
                    "timestamp": data.get("timestamp", time.time()),
                    "confidence": data.get("confidence", 1.0),
                    "metadata": data.get("metadata", {})
                }
                state.head_pose_timestamp = time.time()

            return jsonify({"success": True, "received_at": state.head_pose_timestamp})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif request.method == 'GET':
        # Retrieve head pose data
        with state.lock:
            if state.last_head_pose is None:
                return jsonify({"error": "No head pose data available"}), 404

            # Calculate age of data
            age = time.time() - state.head_pose_timestamp if state.head_pose_timestamp else None

            return jsonify({
                "head_pose": state.last_head_pose,
                "received_at": state.head_pose_timestamp,
                "age_seconds": age
            })

# ------------------ Model Endpoints ------------------
@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available .ply models"""
    try:
        names = list_ply_models(MODELS_DIR)
        return jsonify({"models": [{"name": n} for n in names]})
    except Exception as e:
        print(f"[ERROR] /models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/model', methods=['GET'])
def get_model():
    """Get a specific .ply model by name"""
    try:
        name = request.args.get('name')
        if not name:
            return jsonify({"error": "Missing 'name' parameter"}), 400

        available = set(list_ply_models(MODELS_DIR))
        if name not in available:
            return jsonify({
                "error": f"Unknown model '{name}'",
                "available": sorted(available)
            }), 404

        mesh_b64 = load_mesh_as_b64(os.path.join(MODELS_DIR, name))
        return jsonify({"name": name, "mesh": mesh_b64})

    except Exception as e:
        print(f"[ERROR] /model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/select_model', methods=['POST'])
def select_model():
    """Select a model for AVP pose estimation"""
    try:
        data = request.json
        if not data or 'model_name' not in data:
            return jsonify({"error": "Missing 'model_name' in request"}), 400

        model_name = data['model_name']

        # Validate model exists
        available = set(list_ply_models(MODELS_DIR))
        if model_name not in available:
            return jsonify({
                "error": f"Unknown model '{model_name}'",
                "available": sorted(available)
            }), 404

        # Store selection
        with state.lock:
            state.selected_model = model_name

        return jsonify({
            "success": True,
            "selected_model": model_name
        })

    except Exception as e:
        print(f"[ERROR] /select_model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/avp_pose', methods=['POST'])
def avp_pose():
    """
    Final pose endpoint for AVP.
    Receives: RGB frame, depth map (placeholder), AVP intrinsics, selected model, mask
    Forwards to pose estimation API and returns result.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract parameters
        rgb_frame = data.get('rgb_frame')  # base64 encoded
        depth_map = data.get('depth_map', '')  # Placeholder for now
        camera_matrix = data.get('camera_matrix')
        mask = data.get('mask', '')
        model_name = data.get('model_name')

        # Validate required fields
        if not rgb_frame:
            return jsonify({"error": "Missing 'rgb_frame'"}), 400
        if not camera_matrix:
            return jsonify({"error": "Missing 'camera_matrix'"}), 400

        # Get model selection
        with state.lock:
            if model_name is None:
                model_name = state.selected_model

        if not model_name:
            return jsonify({"error": "No model selected. Use /select_model first"}), 400

        # Validate model exists
        available = set(list_ply_models(MODELS_DIR))
        if model_name not in available:
            return jsonify({
                "error": f"Unknown model '{model_name}'",
                "available": sorted(available)
            }), 404

        # Load model mesh
        try:
            mesh_b64 = load_mesh_as_b64(os.path.join(MODELS_DIR, model_name))
        except Exception as e:
            return jsonify({"error": f"Failed to load model '{model_name}': {e}"}), 500

        # Prepare payload for pose estimation API
        payload = {
            "camera_matrix": camera_matrix,
            "images": [rgb_frame],  # List of images
            "mesh": mesh_b64,
            "mask": mask,
            "depthscale": data.get('depthscale', 1000.0)
        }

        # Forward to pose estimation API
        try:
            response = requests.post(POSE_FORWARD_URL, json=payload, timeout=20)
            response.raise_for_status()
            pose_result = response.json()

            return jsonify({
                "success": True,
                "pose": pose_result,
                "model_used": model_name
            })

        except requests.exceptions.Timeout:
            return jsonify({"error": "Pose estimation API timeout"}), 504
        except requests.exceptions.ConnectionError:
            return jsonify({"error": f"Cannot connect to pose API at {POSE_FORWARD_URL}"}), 503
        except requests.exceptions.HTTPError as e:
            return jsonify({"error": f"Pose API error: {e}"}), 502
        except Exception as e:
            return jsonify({"error": f"Pose API error: {e}"}), 500

    except Exception as e:
        print(f"[ERROR] /avp_pose: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------ Main ------------------
if __name__ == "__main__":
    print("=" * 60)
    print("AVP API - ArUco Vision Processing")
    print("=" * 60)
    print("Server starting on http://localhost:5000")
    print("\n⚠️  NOTE: This API receives frames from external capture program")
    print("    Start screen_capture.py to begin sending frames\n")
    print("Available endpoints:")
    print("  POST /receive_frame  - Receive RGB frame for processing")
    print("  GET  /config         - Get current configuration")
    print("  POST /config         - Update configuration")
    print("  GET  /intrinsics     - Get camera intrinsics")
    print("  GET  /pose           - Get board pose")
    print("  GET  /mask           - Get ROI mask")
    print("  GET  /rgb_frame      - Get raw RGB frame")
    print("  GET  /detected_frame - Get frame with markers drawn")
    print("  POST /head_pose      - Send head pose data from AVP")
    print("  GET  /head_pose      - Get latest head pose data")
    print("  GET  /stats          - Get processing statistics")
    print("  GET  /health         - Health check")
    print("\nModel Endpoints:")
    print("  GET  /models         - List available .ply models")
    print("  GET  /model?name=X   - Get specific .ply model")
    print("  POST /select_model   - Select model for pose estimation")
    print("  POST /avp_pose       - Final pose endpoint for AVP")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
