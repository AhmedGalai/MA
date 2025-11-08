#!/usr/bin/env python3
"""
Main API Server
Receives RGB frames and coordinates with CV pipeline.
Provides endpoints for AVP integration and debugging.
"""

import numpy as np
import math
import cv2 as cv
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import threading
import os

# Import CV pipeline
import computer_vision_pipeline as cvp
try:
    from app_config import APP_CONFIG
except Exception:
    APP_CONFIG = {
        "pose_api": {"base_url": "http://localhost:9000", "route": "/pose"},
        "defaults": {"model_name": "cube.ply"}
    }

app = Flask(__name__)
CORS(app)

# ------------------ Configuration ------------------
MODELS_DIR = "models"  # Local models folder
POSE_API_URL = f"{APP_CONFIG.get('pose_api', {}).get('base_url', 'http://localhost:9000')}{APP_CONFIG.get('pose_api', {}).get('route', '/pose')}"

# ------------------ Global State ------------------
class APIState:
    def __init__(self):
        self.lock = threading.Lock()

        # Head pose data (from external AVP)
        self.last_head_pose = None
        self.head_pose_timestamp = None

        # Model selection (for AVP pose estimation)
        # Default to 'cube.ply' if present; can be overwritten via /select_model
        try:
            default_name = APP_CONFIG.get("defaults", {}).get("model_name", "cube.ply")
            default_model = os.path.join(MODELS_DIR, default_name)
            self.selected_model = default_name if os.path.exists(default_model) else None
        except Exception:
            self.selected_model = None

        # Statistics
        self.frame_count = 0

        # Pose behavior
        self.use_random_pose = APP_CONFIG.get('defaults', {}).get('use_random_pose', True)
        self._pose_t0 = time.perf_counter()

state = APIState()

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
    import base64
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        print(f"[ERROR] load_mesh_as_b64: {e}")
        raise

# ------------------ API Endpoints ------------------
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    with state.lock:
        cv_stats = cvp.get_stats()
        return jsonify({
            "status": "ok",
            "frames_processed": state.frame_count,
            "cv_pipeline": cv_stats
        })

@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    """
    Receive RGB frame from external capture program
    Processes through CV pipeline
    """
    try:
        data = request.json
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame data provided"}), 400

        # Decode frame
        frame_bgr = cvp.decode_base64_image(data['frame'])

        # Process through CV pipeline
        # Only estimate depth if explicitly requested (expensive)
        # Default set to True by request
        estimate_depth = data.get('estimate_depth', True)
        results = cvp.process_frame(frame_bgr, estimate_depth=estimate_depth)

        if results.get('success'):
            with state.lock:
                state.frame_count += 1
            return jsonify({
                "success": True,
                "frame_count": state.frame_count,
                "has_pose": results.get('pose') is not None
            })
        else:
            return jsonify({"error": results.get('error', 'Unknown error')}), 500

    except Exception as e:
        print(f"[ERROR] receive_frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/config', methods=['GET', 'POST'])
def config():
    """Get or set CV pipeline configuration"""
    if request.method == 'GET':
        cv_config = cvp.get_config()
        cv_config.update({
            'use_random_pose': state.use_random_pose
        })
        return jsonify(cv_config)

    elif request.method == 'POST':
        data = request.json
        cvp.update_config(
            hsv_center=data.get('hsv_center'),
            h_tol=data.get('h_tol'),
            s_tol=data.get('s_tol'),
            v_tol=data.get('v_tol')
        )
        if 'use_random_pose' in data:
            with state.lock:
                state.use_random_pose = bool(data.get('use_random_pose'))
        return jsonify({"success": True})

@app.route('/intrinsics', methods=['GET'])
def get_intrinsics():
    """Get camera intrinsics matrix"""
    with cvp.state.lock:
        if cvp.state.last_intrinsics is None:
            return jsonify({"error": "No intrinsics available"}), 404
        return jsonify(cvp.state.last_intrinsics)

@app.route('/pose', methods=['GET'])
def get_pose():
    """Get current board pose (rvec, tvec)"""
    with cvp.state.lock:
        if cvp.state.last_pose is None:
            return jsonify({"error": "No pose available"}), 404
        return jsonify(cvp.state.last_pose)

@app.route('/mask', methods=['GET'])
def get_mask():
    """Get ROI mask as base64 encoded image"""
    with cvp.state.lock:
        if cvp.state.last_mask is None:
            return jsonify({"error": "No mask available"}), 404

        mask_base64 = cvp.encode_image_to_base64(cvp.state.last_mask)
        return jsonify({"mask": mask_base64})

@app.route('/depth', methods=['GET'])
def get_depth():
    """Get depth map as base64 encoded image"""
    with cvp.state.lock:
        if cvp.state.last_depth is None:
            return jsonify({"error": "No depth available"}), 404

        depth_base64 = cvp.encode_image_to_base64(cvp.state.last_depth)
        return jsonify({"depth": depth_base64})

@app.route('/disparity', methods=['GET'])
def get_disparity():
    """Get disparity map as base64 encoded image"""
    with cvp.state.lock:
        if cvp.state.last_disparity is None:
            return jsonify({"error": "No disparity available"}), 404

        disparity_base64 = cvp.encode_image_to_base64(cvp.state.last_disparity)
        return jsonify({"disparity": disparity_base64})

@app.route('/rgb_frame', methods=['GET'])
def get_rgb_frame():
    """Get the last RGB frame without any processing overlays"""
    with cvp.state.lock:
        if cvp.state.last_frame is None:
            return jsonify({"error": "No frame available"}), 404

        frame_base64 = cvp.encode_image_to_base64(cvp.state.last_frame)
        return jsonify({"frame": frame_base64})

@app.route('/detected_frame', methods=['GET'])
def get_detected_frame():
    """Get the last frame with ArUco markers drawn"""
    with cvp.state.lock:
        if cvp.state.last_frame is None:
            return jsonify({"error": "No frame available"}), 404

        frame = cvp.state.last_frame.copy()

        # Detect and draw markers
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, rejected = None, None, None

        if cvp.state.aruco_dict is not None:
            if cvp.state.api == "new":
                corners, ids, rejected = cvp.state.detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv.aruco.detectMarkers(
                    gray, cvp.state.aruco_dict, parameters=cvp.state.detector
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

        frame_base64 = cvp.encode_image_to_base64(frame)
        return jsonify({
            "frame": frame_base64,
            "markers_detected": len(ids) if ids is not None else 0
        })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get processing statistics"""
    with state.lock:
        cv_stats = cvp.get_stats()
        return jsonify({
            "frames_received": state.frame_count,
            "cv_pipeline": cv_stats,
            "has_head_pose": state.last_head_pose is not None,
            "selected_model": state.selected_model,
            "use_random_pose": state.use_random_pose
        })

# ------------------ Head Pose Endpoints ------------------
@app.route('/head_pose', methods=['GET', 'POST'])
def head_pose():
    """Send or receive head pose data"""
    if request.method == 'POST':
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

            return jsonify({
                "success": True,
                "received_at": state.head_pose_timestamp
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif request.method == 'GET':
        with state.lock:
            if state.last_head_pose is None:
                return jsonify({"error": "No head pose data available"}), 404

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
    Final pose endpoint for AVP
    Receives: RGB frame, depth map, AVP intrinsics, mask
    Uses: Last processed disparity from CV pipeline OR provided depth
    Forwards to pose estimation API
    """
    import requests

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract parameters
        rgb_frame = data.get('rgb_frame')  # base64 encoded
        depth_map = data.get('depth_map', '')  # Can be empty, will use pipeline disparity
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

        # If no depth provided, use last disparity from pipeline
        if not depth_map:
            with cvp.state.lock:
                if cvp.state.last_disparity is not None:
                    depth_map = cvp.encode_image_to_base64(cvp.state.last_disparity)
                    print("[INFO] Using disparity from CV pipeline")
                else:
                    print("[WARNING] No depth/disparity available")

        # If no mask provided, use last mask from pipeline
        if not mask:
            with cvp.state.lock:
                if cvp.state.last_mask is not None:
                    mask = cvp.encode_image_to_base64(cvp.state.last_mask)
                    print("[INFO] Using mask from CV pipeline")

        # Load model mesh
        try:
            mesh_b64 = load_mesh_as_b64(os.path.join(MODELS_DIR, model_name))
        except Exception as e:
            return jsonify({"error": f"Failed to load model '{model_name}': {e}"}), 500

        # Prepare payload for pose estimation API (match pose_api schema)
        # pose_api expects: images: [{filename, rgb, depth}]
        image_item = {
            "filename": "frame",
            "rgb": rgb_frame,
            "depth": depth_map or ""
        }
        payload = {
            "camera_matrix": camera_matrix,
            "images": [image_item],
            "mask": mask,
            "mesh": mesh_b64,
            "sequence": False,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

        if state.use_random_pose:
            # Synthesize a deterministic transform
            t = time.perf_counter() - state._pose_t0
            yaw = math.radians(25.0 * t)
            pitch = math.radians(60.0 * t)
            cY, sY = math.cos(yaw), math.sin(yaw)
            cP, sP = math.cos(pitch), math.sin(pitch)
            Rz = np.array([[ cY,-sY, 0.0],[ sY, cY, 0.0],[0.0,0.0, 1.0]], dtype=float)
            Ry = np.array([[ cP, 0.0, sP],[0.0, 1.0,0.0],[-sP, 0.0, cP]], dtype=float)
            R = Rz @ Ry
            T = np.eye(4, dtype=float)
            T[:3, :3] = R
            T[:3,  3] = np.array([0.0, 0.0, 10.0], dtype=float)
            pose_result = {
                "status": "Mock pose (integrated)",
                "transformation_matrix": [T.tolist()],
                "debug": {"yaw_deg": 25.0 * t, "pitch_deg": 60.0 * t}
            }
            return jsonify({
                "success": True,
                "pose": pose_result,
                "model_used": model_name,
                "used_pipeline_disparity": not bool(data.get('depth_map')),
                "used_pipeline_mask": not bool(data.get('mask'))
            })
        else:
            # Forward to real pose estimation API
            try:
                response = requests.post(POSE_API_URL, json=payload, timeout=20)
                response.raise_for_status()
                pose_result = response.json()

                return jsonify({
                    "success": True,
                    "pose": pose_result,
                    "model_used": model_name,
                    "used_pipeline_disparity": not bool(data.get('depth_map')),
                    "used_pipeline_mask": not bool(data.get('mask'))
                })

            except requests.exceptions.Timeout:
                return jsonify({"error": "Pose estimation API timeout"}), 504
            except requests.exceptions.ConnectionError:
                return jsonify({"error": f"Cannot connect to pose API at {POSE_API_URL}"}), 503
            except requests.exceptions.HTTPError as e:
                return jsonify({"error": f"Pose API error: {e}"}), 502
            except Exception as e:
                return jsonify({"error": f"Pose API error: {e}"}), 500

    except Exception as e:
        print(f"[ERROR] /avp_pose: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------ Integrated Pose API-compatible endpoint ------------------
@app.route('/pose', methods=['POST'])
def integrated_pose():
    """
    Pose API-compatible endpoint.
    If state.use_random_pose is True: validate request structure and return mock pose.
    Else: forward to real Pose API defined in APP_CONFIG.pose_api.
    """
    import requests
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Basic structure validation
        errors = []
        if 'camera_matrix' not in data:
            errors.append("camera_matrix missing")
        if 'images' not in data or not isinstance(data['images'], list) or len(data['images']) == 0:
            errors.append("images[] missing or empty")
        else:
            item = data['images'][0]
            for k in ('filename', 'rgb', 'depth'):
                if k not in item:
                    errors.append(f"images[0].{k} missing")
        if 'mesh' not in data:
            errors.append("mesh missing")
        if errors and state.use_random_pose:
            return jsonify({"error": "Invalid payload", "details": errors}), 422

        if state.use_random_pose:
            # Generate deterministic mock pose
            t = time.perf_counter() - state._pose_t0
            yaw = math.radians(25.0 * t)
            pitch = math.radians(60.0 * t)
            cY, sY = math.cos(yaw), math.sin(yaw)
            cP, sP = math.cos(pitch), math.sin(pitch)
            Rz = np.array([[ cY,-sY, 0.0],[ sY, cY, 0.0],[0.0,0.0, 1.0]], dtype=float)
            Ry = np.array([[ cP, 0.0, sP],[0.0, 1.0,0.0],[-sP, 0.0, cP]], dtype=float)
            R = Rz @ Ry
            T = np.eye(4, dtype=float)
            T[:3, :3] = R
            T[:3,  3] = np.array([0.0, 0.0, 10.0], dtype=float)
            return jsonify({
                "status": "Mock pose (integrated)",
                "transformation_matrix": [T.tolist()],
                "debug": {"note": "use_random_pose=True"}
            })
        else:
            # Forward to real API
            try:
                r = requests.post(POSE_API_URL, json=data, timeout=20)
                r.raise_for_status()
                return jsonify(r.json())
            except requests.exceptions.Timeout:
                return jsonify({"error": "Pose estimation API timeout"}), 504
            except requests.exceptions.ConnectionError:
                return jsonify({"error": f"Cannot connect to pose API at {POSE_API_URL}"}), 503
            except requests.exceptions.HTTPError as e:
                return jsonify({"error": f"Pose API error: {e}"}), 502
            except Exception as e:
                return jsonify({"error": f"Pose API error: {e}"}), 500
    except Exception as e:
        print(f"[ERROR] /pose: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------ Main ------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Main API - Computer Vision Pipeline Integration")
    print("=" * 60)
    print("Server starting on http://localhost:5000")
    print(f"\nCV Pipeline Device: {cvp.DEVICE}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Pose API URL: {POSE_API_URL}")
    print("\n⚠️  NOTE: Start screen_capture.py to begin sending frames")
    print("\nAvailable endpoints:")
    print("  POST /receive_frame  - Receive RGB frame for processing")
    print("  GET  /config         - Get CV pipeline configuration")
    print("  POST /config         - Update CV pipeline configuration")
    print("  GET  /intrinsics     - Get camera intrinsics")
    print("  GET  /pose           - Get board pose")
    print("  GET  /mask           - Get ROI mask")
    print("  GET  /depth          - Get depth map")
    print("  GET  /disparity      - Get disparity map")
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
    print("  POST /pose           - Pose API-compatible endpoint (integrated)")
    print("  POST /avp_pose       - Convenience endpoint (builds pose payload and forwards)")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
