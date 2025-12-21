#!/usr/bin/env python3
"""
Simple RealSense to FoundationPose Pipeline
No UxPlay, just RGB-D from RealSense with ROI selection and model picker
"""

import argparse
import cv2
import json
import logging
import numpy as np
import os
import threading
import time
from datetime import datetime
from flask import Flask, Response, jsonify, request, send_from_directory
from pathlib import Path
from typing import Dict, Any, Optional

from config import CONFIG
from realsense_client import RealSenseClient
from foundationpose_client import estimate_pose

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Global state
rs_client: Optional[RealSenseClient] = None
rs_lock = threading.Lock()
latest_frame: Dict[str, Any] = {
    "rgb": None,
    "depth": None,
    "K": None,
    "timestamp": None
}

# ROI configuration
roi_lock = threading.Lock()
roi_config = {
    "x_center": 320,
    "y_center": 240,
    "radius": 100
}

# Model selection
model_lock = threading.Lock()
selected_model = "cube.ply"
available_models = []

# FoundationPose result
fp_lock = threading.Lock()
fp_result = {
    "pose_matrix": None,
    "timestamp": None,
    "error": None
}

# Save next request flag
save_lock = threading.Lock()
save_next = False


def scan_models():
    """Scan models directory for .ply files"""
    global available_models
    models_dir = Path(CONFIG["paths"]["models_dir"])
    if models_dir.exists():
        available_models = sorted([f.name for f in models_dir.glob("*.ply")])
        logger.info(f"Found {len(available_models)} models: {available_models}")
    else:
        logger.warning(f"Models directory not found: {models_dir}")
        available_models = ["textured_simple.ply"]


def rs_capture_thread():
    """Background thread to continuously capture RealSense frames"""
    global latest_frame

    logger.info("RealSense capture thread started")

    while True:
        try:
            if rs_client and rs_client.is_running:
                frame_data = rs_client.capture()
                if frame_data:
                    with rs_lock:
                        latest_frame = frame_data
            else:
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in RS capture thread: {e}", exc_info=True)
            time.sleep(0.5)


def foundationpose_worker():
    """Background worker to process FoundationPose requests"""
    global fp_result, save_next

    logger.info("FoundationPose worker started")
    consecutive_errors = 0
    max_consecutive_errors = 5

    while True:
        try:
            time.sleep(0.5)  # Process at ~2 Hz

            # Back off if too many consecutive errors
            if consecutive_errors >= max_consecutive_errors:
                logger.warning(f"Too many errors ({consecutive_errors}), backing off...")
                time.sleep(10)
                consecutive_errors = 0

            # Get latest frame (quick lock)
            with rs_lock:
                rgb = latest_frame.get("rgb")
                depth = latest_frame.get("depth")
                K = latest_frame.get("K")

            if rgb is None or depth is None or K is None:
                time.sleep(0.1)
                continue

            # Get ROI config (quick lock)
            with roi_lock:
                cx = int(roi_config["x_center"])
                cy = int(roi_config["y_center"])
                radius = int(roi_config["radius"])

            # Create circular mask (no lock needed)
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)

            # Get selected model (quick lock)
            with model_lock:
                model_name = selected_model

            model_path = Path(CONFIG["paths"]["models_dir"]) / model_name
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                time.sleep(1)
                continue

            # Check if we should save this request (quick lock)
            should_save = False
            with save_lock:
                if save_next:
                    should_save = True
                    save_next = False

            # Save request data if requested (no lock needed)
            if should_save:
                try:
                    save_request_data(rgb, depth, mask, K, str(model_path))
                except Exception as e:
                    logger.error(f"Failed to save request: {e}")

            # Call FoundationPose API (BLOCKING - no locks held)
            try:
                api_url = CONFIG["network"]["foundationpose_url"]
                pose_matrix = estimate_pose(rgb, depth, mask, K, str(model_path), api_url)

                # Update result (quick lock)
                with fp_lock:
                    if pose_matrix is not None:
                        fp_result["pose_matrix"] = pose_matrix
                        fp_result["timestamp"] = time.time()
                        fp_result["error"] = None
                        logger.info("FoundationPose: Success")
                        consecutive_errors = 0  # Reset
                    else:
                        fp_result["error"] = "Estimation failed"
                        logger.warning("FoundationPose: Failed")
                        consecutive_errors += 1

            except Exception as api_error:
                logger.error(f"FoundationPose API error: {api_error}")
                with fp_lock:
                    fp_result["error"] = str(api_error)
                consecutive_errors += 1
                time.sleep(2)  # Back off on API errors

        except Exception as e:
            logger.error(f"Error in FoundationPose worker: {e}", exc_info=True)
            with fp_lock:
                fp_result["error"] = str(e)
            consecutive_errors += 1
            time.sleep(1)


def save_request_data(rgb, depth, mask, K, model_path):
    """Save the request data for debugging"""
    try:
        output_dir = Path("fp_requests")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save images
        cv2.imwrite(str(output_dir / f"{timestamp}_rgb.png"), rgb)
        cv2.imwrite(str(output_dir / f"{timestamp}_mask.png"), mask)

        # Save depth as 16-bit PNG
        depth_mm = (depth * 1000).astype(np.uint16)
        cv2.imwrite(str(output_dir / f"{timestamp}_depth.png"), depth_mm)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "K": K.tolist(),
            "model_path": model_path,
            "roi": dict(roi_config)
        }
        with open(output_dir / f"{timestamp}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved request data to {output_dir}/{timestamp}_*")
    except Exception as e:
        logger.error(f"Failed to save request data: {e}")


def draw_axes_on_image(img, rvec, tvec, K, length=0.05):
    """Draw 3D axes on image"""
    points = np.float32([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length]
    ]).reshape(-1, 3)

    img_points, _ = cv2.projectPoints(points, rvec, tvec, K, np.zeros(5))
    img_points = img_points.reshape(-1, 2).astype(int)

    img = img.copy()
    # X-axis (red)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 3)
    # Y-axis (green)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 3)
    # Z-axis (blue)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 3)

    return img


def T_to_rvec_tvec(T):
    """Convert 4x4 transformation matrix to rvec, tvec"""
    R = T[:3, :3]
    t = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    return rvec, tvec


# ============================================================================
# Flask Routes
# ============================================================================

@app.route("/")
def index():
    return "<h1>RealSense FoundationPose Pipeline</h1><p>Go to <a href='/debug'>/debug</a></p>"


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "rs_running": rs_client.is_running if rs_client else False
    })


@app.route("/roi_config", methods=["GET", "POST"])
def roi_config_route():
    """Get or set ROI configuration"""
    global roi_config

    if request.method == "GET":
        with roi_lock:
            return jsonify(dict(roi_config))

    data = request.get_json(silent=True) or {}
    with roi_lock:
        if "x_center" in data:
            roi_config["x_center"] = int(data["x_center"])
        if "y_center" in data:
            roi_config["y_center"] = int(data["y_center"])
        if "radius" in data:
            roi_config["radius"] = int(data["radius"])

    return jsonify(dict(roi_config))


@app.route("/model", methods=["GET", "POST"])
def model_route():
    """Get or set selected model"""
    global selected_model

    if request.method == "GET":
        with model_lock:
            return jsonify({
                "selected": selected_model,
                "available": available_models
            })

    data = request.get_json(silent=True) or {}
    model_name = data.get("model")
    if model_name and model_name in available_models:
        with model_lock:
            selected_model = model_name
        logger.info(f"Model changed to: {model_name}")
        return jsonify({"selected": selected_model})

    return jsonify({"error": "Invalid model"}), 400


@app.route("/save_next", methods=["POST"])
def save_next_route():
    """Enable saving of next FoundationPose request"""
    global save_next

    with save_lock:
        save_next = True

    return jsonify({"status": "next request will be saved"})


@app.route("/pose", methods=["GET"])
def pose_route():
    """Get latest FoundationPose result"""
    with fp_lock:
        if fp_result["pose_matrix"] is not None:
            return jsonify({
                "pose_matrix": fp_result["pose_matrix"].tolist(),
                "timestamp": fp_result["timestamp"],
                "error": fp_result["error"]
            })
        else:
            return jsonify({
                "pose_matrix": None,
                "timestamp": fp_result["timestamp"],
                "error": fp_result["error"] or "No pose yet"
            })


@app.route("/mjpeg")
def mjpeg():
    """MJPEG stream with multiple views"""
    view = request.args.get("view", "rgb")

    def gen():
        while True:
            out = None

            # Get latest frame
            with rs_lock:
                rgb = latest_frame.get("rgb")
                depth = latest_frame.get("depth")
                K = latest_frame.get("K")

            if rgb is None:
                time.sleep(0.05)
                continue

            # Get ROI
            with roi_lock:
                cx = int(roi_config["x_center"])
                cy = int(roi_config["y_center"])
                radius = int(roi_config["radius"])

            if view == "rgb":
                out = rgb.copy()
                cv2.putText(out, "RGB", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            elif view == "depth":
                if depth is None:
                    time.sleep(0.05)
                    continue
                depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                out = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                cv2.putText(out, "Depth", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            elif view == "roi":
                out = rgb.copy()
                # Draw ROI circle
                cv2.circle(out, (cx, cy), radius, (0, 255, 255), 2)
                cv2.circle(out, (cx, cy), 3, (0, 255, 255), -1)

                # Draw pose if available
                with fp_lock:
                    pose = fp_result.get("pose_matrix")

                if pose is not None and K is not None:
                    rvec, tvec = T_to_rvec_tvec(pose)
                    out = draw_axes_on_image(out, rvec, tvec, K, length=0.1)

                    # Show translation
                    txt = f"t: x={tvec[0][0]:.3f} y={tvec[1][0]:.3f} z={tvec[2][0]:.3f}"
                    cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(out, "No pose", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            elif view == "mask":
                mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (cx, cy), radius, 255, -1)
                out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(out, "Mask", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if out is None:
                time.sleep(0.05)
                continue

            # Encode as JPEG
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                time.sleep(0.05)
                continue

            jpg = buf.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n"
                b"\r\n" + jpg + b"\r\n"
            )

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/debug")
def debug():
    """Debug interface with controls"""
    with rs_lock:
        rs_w = rs_client.width if rs_client else 640
        rs_h = rs_client.height if rs_client else 480

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <title>RS FoundationPose Pipeline</title>
  <meta charset="utf-8">
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 20px; background: #f5f5f5; }}
    h1 {{ color: #333; }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px; }}
    .panel {{ border: 2px solid #ddd; border-radius: 8px; overflow: hidden; background: white; }}
    .hdr {{ padding: 10px 15px; background: #4CAF50; color: white; font-weight: 600; font-size: 14px; }}
    img {{ width: 100%; display: block; background: #000; min-height: 360px; }}
    .controls {{ background: white; padding: 20px; border: 2px solid #ddd; border-radius: 8px; }}
    .controls h2 {{ margin-top: 0; color: #333; font-size: 18px; }}
    .row {{ display: flex; align-items: center; gap: 15px; margin: 15px 0; }}
    .row label {{ width: 120px; font-weight: 500; color: #555; }}
    input[type="range"] {{ flex: 1; max-width: 400px; }}
    select {{ flex: 1; max-width: 400px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }}
    .value {{ font-family: 'Courier New', monospace; font-weight: bold; color: #4CAF50; min-width: 60px; }}
    button {{
      background: #2196F3;
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
    }}
    button:hover {{ background: #0b7dda; }}
    button:active {{ background: #0960a5; }}
    .status {{
      margin-top: 10px;
      padding: 10px;
      background: #e8f5e9;
      border-radius: 4px;
      font-size: 13px;
      display: none;
    }}
    .status.show {{ display: block; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>🎯 RealSense → FoundationPose Pipeline</h1>

    <div class="grid">
      <div class="panel">
        <div class="hdr">ROI + Pose Overlay (Primary View)</div>
        <img id="roi" src="/mjpeg?view=roi" />
      </div>
      <div class="panel">
        <div class="hdr">Mask Preview</div>
        <img id="mask" src="/mjpeg?view=mask" />
      </div>
    </div>

    <details style="margin-bottom: 20px;">
      <summary style="cursor: pointer; padding: 10px; background: white; border: 2px solid #ddd; border-radius: 4px; font-weight: 600;">
        📷 Show Additional Views (RGB, Depth)
      </summary>
      <div class="grid" style="margin-top: 15px;">
        <div class="panel">
          <div class="hdr">RGB Feed</div>
          <img id="rgb" data-src="/mjpeg?view=rgb" />
        </div>
        <div class="panel">
          <div class="hdr">Depth Feed</div>
          <img id="depth" data-src="/mjpeg?view=depth" />
        </div>
      </div>
    </details>

    <div class="controls">
      <h2>⚙️ Controls</h2>

      <div class="row">
        <label>ROI X Center:</label>
        <input id="roiX" type="range" min="0" max="{rs_w}" value="{rs_w // 2}" />
        <span class="value" id="roiXVal"></span>
      </div>

      <div class="row">
        <label>ROI Y Center:</label>
        <input id="roiY" type="range" min="0" max="{rs_h}" value="{rs_h // 2}" />
        <span class="value" id="roiYVal"></span>
      </div>

      <div class="row">
        <label>ROI Radius:</label>
        <input id="roiR" type="range" min="20" max="300" value="100" />
        <span class="value" id="roiRVal"></span>
      </div>

      <div class="row">
        <label>Model:</label>
        <select id="modelSelect">
          <option>Loading...</option>
        </select>
      </div>

      <div class="row">
        <label></label>
        <button id="saveBtn" onclick="saveNext()">💾 Save Next Request</button>
        <div id="saveStatus" class="status">Next request will be saved!</div>
      </div>
    </div>
  </div>

  <script>
    // Update value displays
    function updateValue(id, val) {{
      document.getElementById(id + 'Val').textContent = val;
    }}

    // Post ROI config
    async function postROI(payload) {{
      await fetch('/roi_config', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload)
      }});
    }}

    // Load models
    async function loadModels() {{
      const resp = await fetch('/model');
      const data = await resp.json();
      const select = document.getElementById('modelSelect');
      select.innerHTML = '';
      data.available.forEach(model => {{
        const opt = document.createElement('option');
        opt.value = model;
        opt.textContent = model;
        if (model === data.selected) opt.selected = true;
        select.appendChild(opt);
      }});
    }}

    // Change model
    async function changeModel(model) {{
      await fetch('/model', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{model}})
      }});
    }}

    // Save next request
    async function saveNext() {{
      await fetch('/save_next', {{method: 'POST'}});
      const status = document.getElementById('saveStatus');
      status.classList.add('show');
      setTimeout(() => status.classList.remove('show'), 3000);
    }}

    // Wire up controls
    const roiX = document.getElementById('roiX');
    const roiY = document.getElementById('roiY');
    const roiR = document.getElementById('roiR');
    const modelSelect = document.getElementById('modelSelect');

    roiX.addEventListener('input', e => {{
      updateValue('roiX', e.target.value);
      postROI({{x_center: parseInt(e.target.value)}});
    }});

    roiY.addEventListener('input', e => {{
      updateValue('roiY', e.target.value);
      postROI({{y_center: parseInt(e.target.value)}});
    }});

    roiR.addEventListener('input', e => {{
      updateValue('roiR', e.target.value);
      postROI({{radius: parseInt(e.target.value)}});
    }});

    modelSelect.addEventListener('change', e => {{
      changeModel(e.target.value);
    }});

    // Lazy load additional views when expanded
    const details = document.querySelector('details');
    details.addEventListener('toggle', () => {{
      if (details.open) {{
        // Load RGB and Depth streams
        const rgb = document.getElementById('rgb');
        const depth = document.getElementById('depth');
        if (rgb.getAttribute('data-src')) {{
          rgb.src = rgb.getAttribute('data-src');
          rgb.removeAttribute('data-src');
        }}
        if (depth.getAttribute('data-src')) {{
          depth.src = depth.getAttribute('data-src');
          depth.removeAttribute('data-src');
        }}
      }}
    }});

    // Initialize
    (async function() {{
      // Load initial ROI
      const roiResp = await fetch('/roi_config');
      const roi = await roiResp.json();
      roiX.value = roi.x_center;
      roiY.value = roi.y_center;
      roiR.value = roi.radius;
      updateValue('roiX', roi.x_center);
      updateValue('roiY', roi.y_center);
      updateValue('roiR', roi.radius);

      // Load models
      await loadModels();
    }})();
  </script>
</body>
</html>
"""
    return html


def main():
    global rs_client

    parser = argparse.ArgumentParser(description="RealSense FoundationPose Pipeline")
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8001, help="API port")
    parser.add_argument("--rs-width", type=int, default=640, help="RealSense width")
    parser.add_argument("--rs-height", type=int, default=480, help="RealSense height")
    parser.add_argument("--rs-fps", type=int, default=30, help="RealSense FPS")
    args = parser.parse_args()

    # Scan available models
    scan_models()

    # Initialize RealSense
    rs_client = RealSenseClient(width=args.rs_width, height=args.rs_height, fps=args.rs_fps)

    if not rs_client.start():
        logger.error("Failed to start RealSense camera")
        return 1

    # Update ROI defaults based on resolution
    with roi_lock:
        roi_config["x_center"] = args.rs_width // 2
        roi_config["y_center"] = args.rs_height // 2

    # Start background threads
    rs_thread = threading.Thread(target=rs_capture_thread, daemon=True)
    rs_thread.start()

    fp_thread = threading.Thread(target=foundationpose_worker, daemon=True)
    fp_thread.start()

    # Run Flask app
    logger.info(f"Starting API on http://{args.host}:{args.port}")
    logger.info(f"Debug interface: http://{args.host}:{args.port}/debug")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)

    return 0


if __name__ == "__main__":
    exit(main())
