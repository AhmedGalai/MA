# Final Pipeline Usage Guide

## Complete System Overview

The Final Pipeline system consists of three main components:

1. **`main_api.py`** - Main API server (port 5000) that receives AVP frames and processes them
2. **RealSense Camera** - Fixed camera providing metric depth
3. **`tk_debugging.py`** - Debugging GUI showing all feeds and pose overlays

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPLETE SYSTEM FLOW                         │
└─────────────────────────────────────────────────────────────────┘

[Screen Capture]          [AVP Headset]          [RealSense D435]
     │                         │                        │
     │ RGB frames              │ Headset pose           │ Depth + RGB
     │                         │                        │
     ▼                         ▼                        ▼
┌────────────────────────────────────────────────────────────────┐
│                      main_api.py (Port 5000)                   │
│                                                                │
│  • POST /receive_frame      - Stores AVP RGB frames           │
│  • POST /update_head_pose   - Streams headset pose            │
│  • POST /process_with_mask  - Main processing endpoint        │
│  • GET  /avp_frame          - Returns latest AVP frame        │
│  • GET  /mask               - Returns latest mask             │
│  • GET  /pose_result        - Returns pose estimation         │
│  • GET  /stats              - Pipeline statistics             │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │           Final Pipeline Core Processing                  │ │
│  │                                                            │ │
│  │  1. Capture RealSense depth                               │ │
│  │  2. Update headset pose (Kalman filtering)                │ │
│  │  3. Transform mask (AVP → RealSense view)                 │ │
│  │  4. Estimate 6D pose (in RealSense view)                  │ │
│  │  5. Transform pose back (RealSense → AVP view)            │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP REST API
                              ▼
┌────────────────────────────────────────────────────────────────┐
│              tk_debugging.py (Debugging GUI)                   │
│                                                                │
│  ┌────────────┬────────────┐                                  │
│  │ RealSense  │ RealSense  │    Row 1: RealSense feeds        │
│  │ RGB        │ Depth      │                                  │
│  └────────────┴────────────┘                                  │
│  ┌────────────┬────────────┐                                  │
│  │ AVP Mask   │ RS Pose    │    Row 2: Mask + RS pose overlay│
│  │            │ Overlay    │                                  │
│  └────────────┴────────────┘                                  │
│  ┌────────────┬────────────┐                                  │
│  │ AVP RGB    │ AVP Pose   │    Row 3: AVP feed + transformed │
│  │ Feed       │ Overlay    │           pose overlay           │
│  └────────────┴────────────┘                                  │
│                                                                │
│  Pipeline Statistics: Calibration, Success Rate, Timing       │
└────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd final_pipeline
pip install -r requirements.txt
```

**Required packages:**
- `numpy`
- `opencv-python` (cv2)
- `pyrealsense2`
- `scipy`
- `flask`
- `flask-cors`
- `pillow`
- `requests`

### 2. One-Time Calibration

Before using the pipeline, you must calibrate the coordinate transformations:

```python
from final_pipeline import FinalPipeline
import numpy as np
import cv2 as cv

# Initialize pipeline
pipeline = FinalPipeline()

# Capture headset image (with ArUco marker visible)
headset_image = ...  # RGB image from AVP

# Headset camera intrinsics
K_headset = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_headset = np.zeros(5, dtype=np.float32)

# Perform calibration
success = pipeline.calibrate_with_aruco(headset_image, K_headset, dist_headset)

if success:
    print("Calibration successful!")
    # Calibration saved to calibration/*.json
else:
    print("Calibration failed - check marker visibility")
```

**Calibration files created:**
- `calibration/headset_to_world.json`
- `calibration/realsense_to_world.json`
- `calibration/avp_to_realsense.json`

### 3. Start Main API Server

```bash
cd final_pipeline
python main_api.py
```

**Output:**
```
============================================================
Final Pipeline Main API Server
============================================================
[API] Initializing Final Pipeline...
[REALSENSE] Searching for RealSense cameras...
[REALSENSE] Camera found: Intel RealSense D435
[API] Pipeline initialized

Endpoints:
  GET  /health                - Health check & status
  POST /receive_frame         - Receive AVP RGB frame
  POST /update_head_pose      - Update headset pose
  POST /process_with_mask     - Process frame + mask → pose
  POST /calibrate             - ArUco calibration

  GET  /stats                 - Pipeline statistics
  GET  /avp_frame             - Get latest AVP frame
  GET  /mask                  - Get latest mask
  GET  /pose_result           - Get latest pose result
  GET  /head_pose             - Get latest head pose
  GET  /intrinsics            - Get RealSense intrinsics
  GET  /pose_history          - Get pose history
  POST /shutdown              - Shutdown pipeline

Starting server on 0.0.0.0:5000...
============================================================
```

### 4. Start Debugging GUI

In a new terminal:

```bash
cd final_pipeline
python tk_debugging.py
```

The GUI will open showing:
- **Connection controls** (host: localhost, port: 5000)
- **Six video feeds** (3 rows × 2 columns)
- **Pipeline statistics**
- **Control buttons** (Save Frame, Test Pose API, etc.)

## Using the System

### Workflow 1: Processing with Screen Capture

**Step 1: Send AVP RGB frames**

Use a screen capture program to send frames to the API:

```python
import cv2 as cv
import requests
import base64
import io
from PIL import Image

def encode_image(img):
    pil_img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"

# Capture screen
frame = ...  # Your screen capture code

# Send to API
requests.post("http://localhost:5000/receive_frame", json={
    "frame": encode_image(frame),
    "timestamp": time.time()
})
```

**Step 2: Send headset pose updates**

```python
requests.post("http://localhost:5000/update_head_pose", json={
    "position": [x, y, z],
    "rotation": [rx, ry, rz],
    "timestamp": time.time()
})
```

**Step 3: Process with mask**

```python
# Create or capture mask
mask = ...  # Binary mask (0/255)

result = requests.post("http://localhost:5000/process_with_mask", json={
    "mask": encode_image(mask),
    "use_latest_pose": True
}).json()

if result['success']:
    pose_avp = result['pose_avp_view']
    print(f"Position: {pose_avp['tvec']}")
    print(f"Rotation: {pose_avp['rvec']}")
    print(f"Confidence: {pose_avp['confidence']}")
else:
    print(f"Error: {result['error']}")
```

### Workflow 2: Using Debugging GUI

**Features:**

1. **API Connection**
   - Enter host/port (default: localhost:5000)
   - Click "Connect"
   - Status indicator shows green when connected

2. **Video Feeds** (auto-updates at configurable Hz)
   - **RealSense RGB**: Live RGB from fixed camera
   - **RealSense Depth**: Colorized depth (JET colormap)
   - **AVP Mask**: Binary mask from AVP (via API)
   - **RS Pose Overlay**: Pose visualization in RealSense view
   - **AVP RGB Feed**: Captured screen frames (via API)
   - **AVP Pose Overlay**: Transformed pose in AVP view

3. **Controls**
   - **UI Refresh**: 1-60 Hz slider
   - **Pause/Resume**: Toggle auto-updates
   - **Refresh Now**: Manual refresh
   - **Save Next Frame**: Saves RGB, depth, intrinsics to `saved_frames/`
   - **Test Pose API**: Sends test request to verify pipeline

4. **Statistics Panel**
   - Calibration status
   - RealSense availability
   - Frames processed
   - Success/failure counts
   - Success rate percentage
   - Average processing time

### Workflow 3: Direct Python API

```python
from final_pipeline import FinalPipeline
import numpy as np

# Initialize
pipeline = FinalPipeline()

# Ensure calibration
if not pipeline.pose_manager.is_calibrated():
    print("Pipeline not calibrated!")
    exit(1)

# Process frame
result = pipeline.process_frame(
    avp_rgb=avp_frame,      # Optional RGB for visualization
    avp_mask=mask,          # Required: binary mask
    headset_pose={          # Optional: current headset pose
        "position": [x, y, z],
        "rotation": [rx, ry, rz]
    }
)

if result['success']:
    # Pose in AVP coordinates
    pose_avp = result['pose_avp_view']

    # Pose in RealSense coordinates (for debugging)
    pose_rs = result['pose_rs_view']

    # Confidence score
    confidence = result['confidence']

    # Processing time
    time_ms = result['processing_time_ms']
```

## API Endpoints Reference

### POST /receive_frame

**Description**: Receive and store AVP RGB frame from screen capture

**Request:**
```json
{
  "frame": "data:image/jpeg;base64,...",
  "timestamp": 1234567890.123
}
```

**Response:**
```json
{
  "success": true,
  "frame_count": 42,
  "timestamp": 1234567890.123
}
```

### POST /update_head_pose

**Description**: Update headset pose for Kalman filtering

**Request:**
```json
{
  "position": [x, y, z],
  "rotation": [rx, ry, rz],
  "quaternion": [qw, qx, qy, qz],  // optional
  "timestamp": 1234567890.123
}
```

**Response:**
```json
{
  "success": true,
  "timestamp": 1234567890.123
}
```

### POST /process_with_mask

**Description**: Main processing endpoint - estimates 6D pose

**Request:**
```json
{
  "mask": "data:image/png;base64,...",
  "rgb": "data:image/jpeg;base64,...",  // optional
  "use_latest_pose": true
}
```

**Response:**
```json
{
  "success": true,
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
  "visualization": "data:image/jpeg;base64,..."  // if rgb provided
}
```

### GET /avp_frame

**Description**: Get latest AVP RGB frame

**Response:**
```json
{
  "frame": "data:image/jpeg;base64,...",
  "timestamp": 1234567890.123
}
```

### GET /mask

**Description**: Get latest mask

**Response:**
```json
{
  "mask": "data:image/png;base64,..."
}
```

### GET /pose_result

**Description**: Get latest pose estimation result

**Response:**
```json
{
  "success": true,
  "pose_avp_view": {...},
  "pose_rs_view": {...},
  "confidence": 0.95,
  "processing_time_ms": 45.2
}
```

### GET /stats

**Description**: Get pipeline statistics

**Response:**
```json
{
  "calibrated": true,
  "realsense_available": true,
  "frames_processed": 1234,
  "successful_poses": 1180,
  "failed_poses": 54,
  "avg_processing_time_ms": 47.3,
  "api_frame_count": 1234,
  "api_pose_requests": 567,
  "has_avp_frame": true,
  "has_head_pose": true
}
```

## Keyboard Shortcuts (GUI)

- **Space**: Pause/Resume updates
- **S**: Save next frame
- **T**: Test pose API
- **R**: Refresh now
- **Q**: Quit application

## Saved Frame Data

When you click "Save Next Frame", the following files are created in `saved_frames/`:

```
saved_frames/
├── rgb_20250116_230145.png           # RGB image (640×480)
├── depth_20250116_230145.npy         # Depth array (uint16, mm)
├── depth_viz_20250116_230145.png     # Depth visualization
└── intrinsics_20250116_230145.json   # Camera intrinsics
```

**Loading saved depth:**
```python
import numpy as np

# Load depth
depth = np.load('saved_frames/depth_20250116_230145.npy')
print(f"Depth shape: {depth.shape}")  # (480, 640)
print(f"Depth range: {depth.min()} - {depth.max()} mm")

# Convert to meters
depth_meters = depth / 1000.0
```

## Performance Metrics

**Expected latency breakdown:**
- RealSense depth capture: ~33ms
- Kalman filtering: <1ms
- Mask transformation: ~5ms
- Pose estimation: ~8ms
- Pose transformation: ~1ms
- **Total: ~50ms** (20 Hz)

## Troubleshooting

### RealSense Not Found

**Problem**: `[ERROR] No RealSense cameras found`

**Solution:**
1. Verify camera is connected via USB 3.0
2. Run: `rs-enumerate-devices`
3. Reinstall SDK:
   ```bash
   pip uninstall pyrealsense2
   pip install pyrealsense2
   ```

### Pipeline Not Calibrated

**Problem**: `{"success": false, "error": "Not calibrated"}`

**Solution:**
1. Perform one-time ArUco calibration
2. Ensure `calibration/*.json` files exist
3. Check marker visibility during calibration

### Low Success Rate

**Problem**: Success rate < 80%

**Possible causes:**
- Poor mask quality
- Object out of RealSense range (0.3-3m)
- Insufficient depth points (min 4 required)
- Incorrect calibration

**Solution:**
- Verify mask covers object clearly
- Ensure object is within depth range
- Recalibrate if needed

### API Connection Failed

**Problem**: Red status indicator in GUI

**Solution:**
1. Check API server is running:
   ```bash
   curl http://localhost:5000/health
   ```
2. Verify firewall settings
3. Update host/port in GUI

## Advanced Usage

### Custom Camera Intrinsics

If you know the AVP camera intrinsics, provide them during processing:

```python
result = pipeline.process_frame(
    avp_mask=mask,
    avp_intrinsics={
        "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        "dist": [k1, k2, p1, p2, k3]
    }
)
```

### Pose History Queries

Get recent pose history for filtering/validation:

```python
response = requests.get("http://localhost:5000/pose_history?duration=1.0")
history = response.json()

print(f"Poses in last 1 second: {history['count']}")
for pose in history['poses']:
    print(f"  {pose['position']} @ {pose['timestamp']}")
```

### Multi-threaded Processing

For concurrent requests, use locks:

```python
import threading

pipeline_lock = threading.Lock()

def process_request(mask, pose):
    with pipeline_lock:
        result = pipeline.process_frame(avp_mask=mask, headset_pose=pose)
    return result
```

## System Requirements

- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8+
- **RAM**: 4GB minimum
- **USB**: USB 3.0 port for RealSense
- **Camera**: Intel RealSense D435/D455
- **GPU**: Not required (CPU only)

## Summary

The Final Pipeline provides a complete solution for 6D pose estimation with:

✅ **Fixed RealSense depth** for metric accuracy
✅ **Kalman filtering** for smooth pose estimates
✅ **Explicit coordinate transformations** (AVP ↔ RealSense)
✅ **REST API** for easy integration
✅ **Debugging GUI** with 6 video feeds
✅ **Comprehensive documentation**

Perfect for AR/VR applications requiring precise object tracking!
