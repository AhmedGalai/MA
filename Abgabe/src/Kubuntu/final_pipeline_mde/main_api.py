#!/usr/bin/env python3
"""
Main API Server for Final Pipeline
Integrated server that combines RealSense depth, pose estimation,
and AVP frame processing with coordinate transformations.
"""

import numpy as np
import cv2 as cv
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import threading
import base64
import io
from PIL import Image

from pipeline_core import FinalPipeline
from config import API_CONFIG

app = Flask(__name__)
CORS(app)

# ------------------ Global State ------------------
class APIState:
    def __init__(self):
        self.lock = threading.Lock()

        # AVP frame data (from screen capture)
        self.last_avp_frame = None
        self.last_avp_frame_timestamp = None

        # Headset pose data (from AVP)
        self.last_head_pose = None
        self.head_pose_timestamp = None

        # Processing results
        self.last_pose_result = None
        self.last_mask = None

        # Statistics
        self.frame_count = 0
        self.pose_requests = 0
        self.save_next_pose_request = False

state = APIState()

# Initialize pipeline
pipeline = None

def initialize_pipeline():
    """Initialize the final pipeline"""
    global pipeline
    print("[API] Initializing Final Pipeline...")
    pipeline = FinalPipeline()
    print("[API] Pipeline initialized")

# ------------------ Helper Functions ------------------
def decode_base64_image(base64_str):
    """Decode base64 string to numpy array (BGR format)"""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        img_rgb = np.array(img)

        # Convert to BGR if RGB
        if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
            img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
        else:
            img_bgr = img_rgb

        return img_bgr
    except Exception as e:
        print(f"[ERROR] decode_base64_image: {e}")
        return None

def encode_image_to_base64(img):
    """Encode numpy array to base64 JPEG string"""
    if img is None:
        return None

    try:
        # Handle grayscale
        if len(img.shape) == 2:
            img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        else:
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img_rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"[ERROR] encode_image_to_base64: {e}")
        return None

# ------------------ API Endpoints ------------------
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if pipeline is None:
        return jsonify({"status": "initializing"}), 503

    stats = pipeline.get_stats()
    return jsonify({
        "status": "ok",
        "calibrated": pipeline.pose_manager.is_calibrated(),
        "realsense_available": pipeline.realsense.available,
        "frames_processed": state.frame_count,
        "pipeline_stats": stats
    })

@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    """
    Receive RGB frame from external capture program (e.g., screen_capture.py)
    Stores frame for later processing

    Request JSON:
    {
        "frame": "data:image/jpeg;base64,...",
        "timestamp": 1234567890.123
    }
    """
    try:
        data = request.json
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame data provided"}), 400

        # Decode frame
        frame_bgr = decode_base64_image(data['frame'])
        if frame_bgr is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        # Store frame
        with state.lock:
            state.last_avp_frame = frame_bgr
            state.last_avp_frame_timestamp = data.get('timestamp', time.time())
            state.frame_count += 1

        return jsonify({
            "success": True,
            "frame_count": state.frame_count,
            "timestamp": state.last_avp_frame_timestamp
        })

    except Exception as e:
        print(f"[ERROR] receive_frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/update_head_pose', methods=['POST'])
def update_head_pose():
    """
    Update headset pose from AVP

    Request JSON:
    {
        "position": [x, y, z],
        "rotation": [rx, ry, rz],
        "quaternion": [qw, qx, qy, qz],  // optional
        "timestamp": 1234567890.123
    }
    """
    try:
        pose_data = request.json
        if not pose_data:
            return jsonify({"error": "No pose data provided"}), 400

        # Update pose in pipeline
        pipeline.pose_manager.update_headset_pose(pose_data)

        # Store in state
        with state.lock:
            state.last_head_pose = pose_data
            state.head_pose_timestamp = pose_data.get('timestamp', time.time())

        return jsonify({
            "success": True,
            "timestamp": state.head_pose_timestamp
        })

    except Exception as e:
        print(f"[ERROR] update_head_pose: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process_with_mask', methods=['POST'])
def process_with_mask():
    """
    Process frame with mask to estimate 6D pose

    Request JSON:
    {
        "mask": "data:image/png;base64,...",  // Binary mask from AVP
        "rgb": "data:image/jpeg;base64,...",   // Optional RGB for visualization
        "use_latest_pose": true                // Use latest headset pose
    }

    Response:
    {
        "success": true,
        "pose_avp_view": {
            "rvec": [rx, ry, rz],
            "tvec": [x, y, z],
            "confidence": 0.95
        },
        "pose_rs_view": {...},
        "confidence": 0.95,
        "processing_time_ms": 45.2,
        "visualization": "data:image/jpeg;base64,..."  // If rgb provided
    }
    """
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 503

    try:
        data = request.json
        if not data or 'mask' not in data:
            return jsonify({"error": "No mask provided"}), 400

        # Decode mask
        mask = decode_base64_image(data['mask'])
        if mask is None:
            return jsonify({"error": "Failed to decode mask"}), 400

        # Convert to grayscale if needed
        if len(mask.shape) == 3:
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        # Get RGB (optional)
        rgb = None
        if 'rgb' in data:
            rgb = decode_base64_image(data['rgb'])

        # Get headset pose
        headset_pose = None
        if data.get('use_latest_pose', True):
            with state.lock:
                headset_pose = state.last_head_pose

        # Check if we need to save this frame's data
        save_data = False
        with state.lock:
            if state.save_next_pose_request:
                save_data = True
                state.save_next_pose_request = False

        # Process through pipeline
        result = pipeline.process_frame(
            avp_rgb=rgb,
            avp_mask=mask,
            headset_pose=headset_pose,
            save_pose_request_data=save_data
        )

        # Store result
        with state.lock:
            state.last_pose_result = result
            state.last_mask = mask
            state.pose_requests += 1

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] process_with_mask: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/trigger_save_pose_request', methods=['POST'])
def trigger_save_pose_request():
    """Sets a flag to save the next pose request's inputs and outputs."""
    with state.lock:
        state.save_next_pose_request = True
    return jsonify({"success": True, "message": "Next pose request will be saved."})

@app.route('/calibrate', methods=['POST'])
def calibrate():
    """
    Perform ArUco calibration

    Request JSON:
    {
        "headset_image": "data:image/jpeg;base64,...",
        "headset_intrinsics": {
            "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "dist": [k1, k2, p1, p2, k3]
        }
    }
    """
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 503

    try:
        data = request.json

        # Decode headset image
        headset_image = decode_base64_image(data["headset_image"])
        if headset_image is None:
            return jsonify({"error": "Failed to decode headset image"}), 400

        # Get intrinsics
        intrinsics = data["headset_intrinsics"]
        K = np.array(intrinsics["K"], dtype=np.float32)
        dist = np.array(intrinsics["dist"], dtype=np.float32).reshape(-1, 1)

        # Perform calibration
        success = pipeline.calibrate_with_aruco(headset_image, K, dist)

        if success:
            return jsonify({
                "success": True,
                "message": "Calibration successful"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Calibration failed - check marker visibility"
            }), 400

    except Exception as e:
        print(f"[ERROR] calibrate: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------ Data Access Endpoints (for debugging) ------------------
@app.route('/stats', methods=['GET'])
def get_stats():
    """Get pipeline statistics"""
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 503

    stats = pipeline.get_stats()
    with state.lock:
        stats.update({
            "calibrated": pipeline.pose_manager.is_calibrated(),
            "realsense_available": pipeline.realsense.available,
            "api_frame_count": state.frame_count,
            "api_pose_requests": state.pose_requests,
            "has_avp_frame": state.last_avp_frame is not None,
            "has_head_pose": state.last_head_pose is not None
        })

    return jsonify(stats)

@app.route('/avp_frame', methods=['GET'])
def get_avp_frame():
    """Get latest AVP frame as base64"""
    with state.lock:
        if state.last_avp_frame is None:
            return jsonify({"error": "No AVP frame available"}), 404

        frame_base64 = encode_image_to_base64(state.last_avp_frame)
        return jsonify({
            "frame": frame_base64,
            "timestamp": state.last_avp_frame_timestamp
        })

@app.route('/mask', methods=['GET'])
def get_mask():
    """Get latest mask as base64"""
    with state.lock:
        if state.last_mask is None:
            return jsonify({"error": "No mask available"}), 404

        mask_base64 = encode_image_to_base64(state.last_mask)
        return jsonify({"mask": mask_base64})

@app.route('/pose_result', methods=['GET'])
def get_pose_result():
    """Get latest pose estimation result"""
    with state.lock:
        if state.last_pose_result is None:
            return jsonify({"error": "No pose result available"}), 404

        return jsonify(state.last_pose_result)

@app.route('/head_pose', methods=['GET'])
def get_head_pose():
    """Get latest head pose"""
    with state.lock:
        if state.last_head_pose is None:
            return jsonify({"error": "No head pose available"}), 404

        return jsonify(state.last_head_pose)

@app.route('/intrinsics', methods=['GET'])
def get_intrinsics():
    """Get RealSense camera intrinsics"""
    if pipeline is None or not pipeline.realsense.available:
        return jsonify({"error": "RealSense not available"}), 404

    intrinsics = pipeline.realsense.intrinsics
    if intrinsics is None:
        return jsonify({"error": "No intrinsics available"}), 404

    return jsonify({
        "K": intrinsics["K"].tolist(),
        "dist": intrinsics["dist"].tolist(),
        "width": intrinsics["width"],
        "height": intrinsics["height"],
        "fx": intrinsics["fx"],
        "fy": intrinsics["fy"],
        "ppx": intrinsics["ppx"],
        "ppy": intrinsics["ppy"]
    })

@app.route('/pose_history', methods=['GET'])
def get_pose_history():
    """
    Get recent pose history

    Query params:
        duration: Time window in seconds (default 1.0)
    """
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 503

    try:
        duration = float(request.args.get('duration', 1.0))
        history = pipeline.pose_manager.get_pose_history(duration)

        return jsonify({
            "poses": history,
            "count": len(history)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown pipeline and cleanup"""
    global pipeline

    if pipeline is not None:
        pipeline.shutdown()
        pipeline = None

    return jsonify({"success": True, "message": "Pipeline shutdown complete"})

# ------------------ Startup ------------------
def main():
    """Main entry point"""
    print("=" * 60)
    print("Final Pipeline Main API Server")
    print("=" * 60)

    # Initialize pipeline
    initialize_pipeline()

    # Print endpoints
    print("\nEndpoints:")
    print("  GET  /health                - Health check & status")
    print("  POST /receive_frame         - Receive AVP RGB frame")
    print("  POST /update_head_pose      - Update headset pose")
    print("  POST /process_with_mask     - Process frame + mask → pose")
    print("  POST /calibrate             - ArUco calibration")
    print("\n  GET  /stats                 - Pipeline statistics")
    print("  GET  /avp_frame             - Get latest AVP frame")
    print("  GET  /mask                  - Get latest mask")
    print("  GET  /pose_result           - Get latest pose result")
    print("  GET  /head_pose             - Get latest head pose")
    print("  GET  /intrinsics            - Get RealSense intrinsics")
    print("  GET  /pose_history          - Get pose history")
    print("  POST /shutdown              - Shutdown pipeline")

    print(f"\nStarting server on {API_CONFIG['host']}:{API_CONFIG['port']}...")
    print("=" * 60)

    # Run Flask app
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug'],
        threaded=True
    )

if __name__ == "__main__":
    main()
