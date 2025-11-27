"""
Final Pipeline REST API
Clean API interface for the complete pose estimation pipeline
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2 as cv
import base64
import io
from PIL import Image
import traceback

from .pipeline_core import FinalPipeline
from .config import API_CONFIG

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global pipeline instance
pipeline = None


def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decode base64 string to numpy array

    Args:
        base64_str: Base64 encoded image string

    Returns:
        Image as numpy array (BGR format)
    """
    # Remove data URL prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]

    # Decode base64
    img_data = base64.b64decode(base64_str)

    # Convert to PIL Image
    img = Image.open(io.BytesIO(img_data))

    # Convert to numpy array
    img_rgb = np.array(img)

    # Convert RGB to BGR for OpenCV
    if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
        img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
    else:
        img_bgr = img_rgb

    return img_bgr


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "pipeline_initialized": pipeline is not None,
        "realsense_available": pipeline.realsense.available if pipeline else False,
        "calibrated": pipeline.pose_manager.is_calibrated() if pipeline else False
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get pipeline statistics"""
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 500

    stats = pipeline.get_stats()
    stats["calibrated"] = pipeline.pose_manager.is_calibrated()
    stats["realsense_available"] = pipeline.realsense.available

    return jsonify(stats)


@app.route('/calibrate', methods=['POST'])
def calibrate():
    """
    Perform ArUco calibration

    Request JSON:
    {
        "headset_image": "base64_image_string",
        "headset_intrinsics": {
            "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "dist": [k1, k2, p1, p2, k3]
        }
    }

    Response:
    {
        "success": true/false,
        "message": "..."
    }
    """
    global pipeline

    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 500

    try:
        data = request.get_json()

        # Decode headset image
        headset_image = decode_base64_image(data["headset_image"])

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
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/process', methods=['POST'])
def process_frame():
    """
    Process frame through complete pipeline

    Request JSON:
    {
        "rgb": "base64_image_string" (optional),
        "mask": "base64_mask_string" (required),
        "headset_pose": {
            "position": [x, y, z],
            "rotation": [rx, ry, rz]  // rotation vector or Euler angles
        } (optional)
    }

    Response:
    {
        "success": true/false,
        "pose_avp_view": {
            "rvec": [rx, ry, rz],
            "tvec": [x, y, z],
            "confidence": 0.95
        },
        "pose_rs_view": {
            "rvec": [rx, ry, rz],
            "tvec": [x, y, z],
            "confidence": 0.95
        },
        "confidence": 0.95,
        "processing_time_ms": 45.2,
        "visualization": "base64_image_string" (if rgb provided)
    }
    """
    global pipeline

    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 500

    if not pipeline.pose_manager.is_calibrated():
        return jsonify({
            "error": "Pipeline not calibrated - call /calibrate first"
        }), 400

    try:
        data = request.get_json()

        # Decode mask (required)
        if "mask" not in data:
            return jsonify({"error": "Mask is required"}), 400

        mask = decode_base64_image(data["mask"])

        # Convert to grayscale if needed
        if len(mask.shape) == 3:
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        # Decode RGB (optional)
        rgb = None
        if "rgb" in data:
            rgb = decode_base64_image(data["rgb"])

        # Get headset pose (optional)
        headset_pose = data.get("headset_pose", None)

        # Process frame
        result = pipeline.process_frame(
            avp_rgb=rgb,
            avp_mask=mask,
            headset_pose=headset_pose
        )

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/update_pose', methods=['POST'])
def update_pose():
    """
    Update headset pose (for streaming pose updates)

    Request JSON:
    {
        "position": [x, y, z],
        "rotation": [rx, ry, rz]  // or quaternion
    }

    Response:
    {
        "success": true,
        "timestamp": 1234567890.123
    }
    """
    global pipeline

    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 500

    try:
        pose = request.get_json()
        pipeline.pose_manager.update_headset_pose(pose)

        return jsonify({
            "success": True,
            "timestamp": pipeline.pose_manager.last_headset_update
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/pose_history', methods=['GET'])
def get_pose_history():
    """
    Get recent pose history

    Query params:
        duration: Time window in seconds (default 1.0)

    Response:
    {
        "poses": [
            {
                "pose": {...},
                "timestamp": 123.456
            },
            ...
        ],
        "count": 30
    }
    """
    global pipeline

    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 500

    try:
        duration = float(request.args.get('duration', 1.0))
        history = pipeline.pose_manager.get_pose_history(duration)

        return jsonify({
            "poses": history,
            "count": len(history)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown pipeline and cleanup"""
    global pipeline

    if pipeline is not None:
        pipeline.shutdown()

    return jsonify({"success": True, "message": "Pipeline shutdown complete"})


def initialize_pipeline():
    """Initialize global pipeline instance"""
    global pipeline
    print("[API] Initializing pipeline...")
    pipeline = FinalPipeline()
    print("[API] Pipeline ready")


def main():
    """Run API server"""
    print("=" * 60)
    print("Final Pipeline API Server")
    print("=" * 60)

    # Initialize pipeline
    initialize_pipeline()

    # Print endpoints
    print("\nEndpoints:")
    print("  GET  /health         - Health check")
    print("  GET  /stats          - Pipeline statistics")
    print("  POST /calibrate      - Perform ArUco calibration")
    print("  POST /process        - Process frame (main endpoint)")
    print("  POST /update_pose    - Update headset pose")
    print("  GET  /pose_history   - Get pose history")
    print("  POST /shutdown       - Shutdown pipeline")

    print(f"\nStarting server on {API_CONFIG['host']}:{API_CONFIG['port']}...")
    print("=" * 60)

    # Run Flask app
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )


if __name__ == "__main__":
    main()
