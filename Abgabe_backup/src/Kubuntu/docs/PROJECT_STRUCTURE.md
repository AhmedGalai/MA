# Project Structure Documentation

## Overview

This document describes the complete project structure for the AR pose estimation system integrating Python backend processing with Apple Vision Pro client.

## Directory Structure

```
AW19/
├── full_python_pipeline/          # Unified CV pipeline package
│   ├── __init__.py                # Package initialization
│   ├── app_config.py              # Configuration (duplicated for module import)
│   ├── computer_vision_pipeline.py # Main CV pipeline with dual depth support
│   ├── realsense_adapter_adjusted.py # RealSense depth alignment
│   ├── tk_debugging_unified.py    # Debug visualization GUI
│   ├── tk_rs_adjusted_panel.py    # RealSense-specific debug panel
│   ├── fullsystem.tex             # LaTeX documentation (this implementation)
│   └── README.md                  # Module documentation
│
├── models/                         # CAD model library (.ply files)
│   ├── cube.ply
│   ├── ball.ply
│   ├── Banana.ply
│   ├── Football.ply
│   ├── Power Drill-ply.ply
│   └── ...
│
├── main_api.py                    # Main Flask API server
├── app_config.py                  # Central configuration
├── computer_vision_pipeline.py    # CV pipeline (symlink/copy)
├── screen_capture.py              # Screen capture client
├── test_system.py                 # System integration tests
├── select_default_model.py        # Model selection utility
├── tk_debugging_unified.py        # Debug GUI (symlink/copy)
└── __pycache__/                   # Python bytecode cache

../../sandbox/repo/MA/AW19/PoseOverlayApp/  # VisionOS Application
├── PoseOverlayApp/
│   ├── PoseOverlayAppApp.swift       # App entry point
│   ├── AppModel.swift                # Global state
│   ├── ContentView.swift             # Main UI
│   ├── ImmersiveSpaceView.swift      # AR scene with RealityKit
│   ├── SensorDataModel.swift         # ARKit integration
│   ├── LogsView.swift                # Debug logging UI
│   ├── SensorMonitorView.swift       # Sensor data display
│   ├── ROIOverlayView.swift          # ROI selection (circular)
│   ├── ROIOverlayFreeformView.swift  # ROI selection (freeform)
│   ├── Services/
│   │   ├── PoseService.swift         # Backend API client
│   │   ├── ModelService.swift        # Model management
│   │   └── HeadPoseService.swift     # Head tracking sender
│   ├── Models/
│   │   └── PoseResponse.swift        # API response DTOs
│   └── Info.plist                    # App permissions/entitlements
└── Packages/
    └── RealityKitContent/            # RealityKit assets
```

## Key Files Description

### Backend (Python)

#### Core Pipeline
- **`main_api.py`** - Flask server providing RESTful API for:
  - Frame processing (`/receive_frame`)
  - Intrinsics, pose, depth, mask retrieval
  - Head pose tracking (`/head_pose`)
  - Model management (`/models`, `/select_model`)
  - AVP pose estimation (`/avp_pose`)

- **`full_python_pipeline/computer_vision_pipeline.py`** - Unified CV module:
  - ArUco marker detection (3×4 grid, DICT_4X4_50)
  - PnP pose estimation with IPPE
  - Dual depth: RealSense hardware OR Transformers (Depth-Anything-V2)
  - ROI mask extraction (HSV-based)
  - GPU acceleration (CUDA)
  - Time synchronization (all outputs timestamped)
  - PNG depth encoding

- **`full_python_pipeline/realsense_adapter_adjusted.py`** - RealSense integration:
  - D435/D455 camera interface
  - Depth alignment to color frame
  - 3D reprojection to AVP coordinates
  - Z-buffer splatting for occlusion
  - Extrinsics calibration support

#### Configuration
- **`app_config.py`** - Central config:
  ```python
  {
    "main_api": {"host": "0.0.0.0", "port": 5000},
    "pose_api": {"base_url": "http://localhost:9000"},
    "defaults": {
      "use_realsense": False,      # Depth mode toggle
      "use_random_pose": True,     # Mock vs real pose API
      "model_name": "cube.ply"
    }
  }
  ```

#### Utilities
- **`screen_capture.py`** - Desktop frame capture, sends frames to `/receive_frame`
- **`tk_debugging_unified.py`** - Tkinter GUI showing RGB, mask, depth, disparity, pose overlays
- **`test_system.py`** - Integration tests for API endpoints
- **`select_default_model.py`** - CLI for model selection

### Frontend (VisionOS - Swift)

#### Core App
- **`PoseOverlayAppApp.swift`** - App lifecycle, scene configuration
- **`AppModel.swift`** - `@ObservableObject` holding:
  - Base URL for backend
  - Selected model name
  - Immersive space state

#### AR Scene
- **`ImmersiveSpaceView.swift`** - RealityKit immersive space:
  - Head-locked anchor
  - Polling task (fetches poses every 2s)
  - Entity management (pose arrows/CAD models)
  - Coordinate conversion (OpenCV → RealityKit)

#### Networking
- **`Services/PoseService.swift`** - Backend API client:
  - Concurrent snapshot fetching (`async let`)
  - RGB frame, intrinsics, mask, disparity retrieval
  - Pose request to `/avp_pose`
  - Matrix conversion utilities

- **`Services/HeadPoseService.swift`** - Sends ARKit head pose to backend `/head_pose`

- **`Services/ModelService.swift`** - Fetches model list, downloads .ply files

#### Sensor Integration
- **`SensorDataModel.swift`** - ARKit wrapper:
  - World tracking session
  - Head position/orientation as `@Published` properties
  - Periodic backend updates

#### UI Components
- **`ContentView.swift`** - Main menu:
  - URL input field
  - Model picker
  - Immersive space toggle
  - Logs viewer

- **`LogsView.swift`** - Scrollable debug log with timestamps
- **`SensorMonitorView.swift`** - Live sensor data display
- **`ROIOverlayView.swift`** - Circular ROI selector (gaze + pinch)

#### Data Models
- **`Models/PoseResponse.swift`** - Decodable structs:
  ```swift
  struct PoseResponse: Decodable {
      let status: String?
      let transformation_matrix: [Matrix4x4DTO]
  }
  typealias Matrix4x4DTO = [[Double]]
  ```

## Workflow Integration

### Development Setup

1. **Backend Terminal 1:**
   ```bash
   cd AW19
   python main_api.py
   # Starts Flask on http://0.0.0.0:5000
   ```

2. **Backend Terminal 2 (optional - for screen capture):**
   ```bash
   python screen_capture.py
   # Captures screen region, sends to /receive_frame
   ```

3. **Backend Terminal 3 (optional - debug viewer):**
   ```bash
   python full_python_pipeline/tk_debugging_unified.py
   # Shows 6-panel view: RGB, mask, depth, disparity, etc.
   ```

4. **VisionOS (Xcode):**
   - Open `PoseOverlayApp.xcodeproj`
   - Build for Vision Pro device/simulator
   - Run app, enter backend URL (e.g., `http://192.168.1.100:5000`)
   - Select model, open immersive space

### Runtime Dataflow

```
[Vision Pro] --HTTP GET--> [/rgb_frame, /intrinsics, /mask, /disparity]
             <--JSON------ (snapshot data)

[Vision Pro] --HTTP POST-> [/avp_pose] (snapshot + model_name)

[main_api.py] --> [computer_vision_pipeline] (if depth/mask not provided)
              --> [Pose API] (external service at :9000)
              <-- [4x4 transformation matrix]
              --> [apply head pose correction]

[Vision Pro] <--JSON----- [corrected transformation matrices]
             --> [convert OpenCV → RealityKit coords]
             --> [render CAD overlay in AR]
```

## LaTeX Documentation

The file `full_python_pipeline/fullsystem.tex` contains detailed technical documentation suitable for a master's thesis, including:

- System architecture overview
- Mathematical formulations for geometric transformations
- RealSense depth reprojection (3D → 2D with Z-buffering)
- ArUco board layout and PnP pose estimation
- Head pose correction using quaternions
- VisionOS integration and coordinate conversions
- Performance analysis and latency breakdown
- Deployment instructions

### Key Equations Documented

1. **Back-projection (pixel → 3D):**
   $$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = Z \cdot \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

2. **Rigid transformation (RS → AVP):**
   $$\mathbf{p}_{\text{AVP}} = \mathbf{R}_{\text{AVP←RS}} \mathbf{p}_{\text{RS}} + \mathbf{t}_{\text{AVP←RS}}$$

3. **Projection (3D → pixel):**
   $$\begin{bmatrix} u' \\ v' \\ 1 \end{bmatrix} = \frac{1}{Z} \mathbf{K} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}$$

4. **Quaternion → Rotation Matrix:**
   $$\mathbf{R} = \begin{bmatrix}
   1-2(q_y^2+q_z^2) & 2(q_xq_y-q_wq_z) & 2(q_xq_z+q_wq_y) \\
   2(q_xq_y+q_wq_z) & 1-2(q_x^2+q_z^2) & 2(q_yq_z-q_wq_x) \\
   2(q_xq_z-q_wq_y) & 2(q_yq_z+q_wq_x) & 1-2(q_x^2+q_y^2)
   \end{bmatrix}$$

5. **Head pose correction:**
   $$\mathbf{T}_{\text{corrected}} = \mathbf{T}_{\text{head}} \cdot \mathbf{T}_{\text{object}}$$

6. **OpenCV → RealityKit conversion:**
   $$\mathbf{T}_{\text{RK}} = \begin{bmatrix} 1&0&0&0 \\ 0&-1&0&0 \\ 0&0&-1&0 \\ 0&0&0&1 \end{bmatrix} \cdot \mathbf{T}_{\text{CV}}$$

## Dependencies

### Python Backend
```
numpy
opencv-python (cv2)
torch
transformers
pillow
flask
flask-cors
requests
pyrealsense2  # Optional, only if use_realsense=True
```

### VisionOS Client
- Xcode 15+
- visionOS SDK
- Swift 5.9+
- RealityKit
- ARKit
- SwiftUI

## Configuration Flags

### `use_realsense` (app_config.py)
- `True`: Use Intel RealSense D435/D455 hardware depth
- `False`: Use Transformers-based monocular depth (Depth-Anything-V2)

### `use_random_pose` (app_config.py)
- `True`: Return mock animated poses (for testing without pose API)
- `False`: Forward to real pose estimation API at `pose_api.base_url`

## Testing

### Unit Tests
```bash
python test_system.py
# Tests: /health, /intrinsics, /pose, /models, /avp_pose
```

### Integration Test
1. Start `main_api.py`
2. Start `screen_capture.py` (or point RealSense at ArUco board)
3. Open `tk_debugging_unified.py` to verify pipeline outputs
4. Check console for detection logs
5. Query endpoints via `curl` or Vision Pro app

### Expected Outputs
- ArUco markers detected: Check `/detected_frame` endpoint
- Intrinsics available: `GET /intrinsics` returns 3×3 matrix
- Depth visible: `/depth` or `/disparity` returns base64 PNG
- Pose available: `/pose` returns rvec, tvec
- Mock pose works: Set `use_random_pose=True`, query `/avp_pose`

## Troubleshooting

### "RealSense not available"
- Install: `pip install pyrealsense2`
- Connect camera via USB 3.0
- OR set `use_realsense=False` in config

### "Depth model loading failed"
- Ensure internet connection (first run downloads ~400MB)
- Check: `pip install transformers torch`
- GPU recommended for acceptable performance

### "No pose available"
- Ensure ArUco markers visible in frame
- Check lighting (markers need good contrast)
- Verify camera intrinsics are reasonable
- Use `/detected_frame` to visualize detections

### Vision Pro connection issues
- Ensure backend running on `0.0.0.0:5000`
- Check firewall allows incoming connections
- Verify IP address reachable from Vision Pro
- Use `http://`, not `https://` (unless TLS configured)

## Future Enhancements

1. **Real-time video streaming:** Replace polling with WebSocket continuous stream
2. **On-device depth:** Use Vision Pro's LiDAR for depth instead of backend estimation
3. **Multi-object tracking:** Support multiple CAD models simultaneously
4. **Gesture interactions:** Pinch-to-scale, rotate overlays
5. **Persistent anchors:** Use ARKit anchors to persist poses across sessions
6. **Model download caching:** Cache .ply files on device
7. **Offline mode:** Bundle models and run pose estimation on-device (if feasible)

## References

- Apple Vision Pro Developer Documentation: https://developer.apple.com/visionos/
- RealityKit: https://developer.apple.com/documentation/realitykit/
- Depth-Anything-V2: https://huggingface.co/depth-anything/
- OpenCV ArUco: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- Intel RealSense SDK: https://github.com/IntelRealSense/librealsense

---

**Last Updated:** 2025-11-09
**Authors:** Ahmed W.
**Project:** AR Pose Estimation with Apple Vision Pro and Python Backend
