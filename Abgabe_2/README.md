# Abgabe_2 - Simplified 6D Pose Estimation Pipeline

Clean implementation of the AR pose estimation system for Apple Vision Pro integration with simplified architecture.

## Overview

This implementation provides a streamlined pipeline that:
1. **Calibrates** coordinate transformations using ArUco markers (one-time setup)
2. **Receives** mask from AVP (single RGB frame)
3. **Transforms** mask from AVP view to RealSense view
4. **Estimates** 6D pose using FoundationPose with RealSense depth
5. **Returns** both RealSense camera pose AND object pose for dual overlay in VisionOS

## Architecture

```
AVP (VisionOS)          →  Backend (Kubuntu)       →  FoundationPose API
- Head pose streaming      - Mask transformation       - 6D pose estimation
- ROI/mask selection       - Coordinate management
- Dual pose overlay        - RealSense capture
```

## Directory Structure

```
Abgabe_2/
├── src/
│   ├── Kubuntu/              # Backend pipeline
│   │   ├── main_api.py       # Flask REST API server
│   │   ├── config.py         # Configuration
│   │   ├── aruco_calibration.py
│   │   ├── realsense_client.py
│   │   ├── coordinate_manager.py
│   │   ├── mask_transformer.py
│   │   ├── foundationpose_client.py
│   │   ├── models/           # .ply 3D models
│   │   ├── extrinsics/       # Calibration data
│   │   └── requirements.txt
│   │
│   └── VisionOS/            # Apple Vision Pro app
│       └── PoseOverlayApp/  # Swift/SwiftUI app
│
└── README.md
```

## Quick Start

### 1. Backend Setup (Kubuntu)

```bash
cd src/Kubuntu

# Install dependencies
pip install -r requirements.txt

# Start API server
python main_api.py
```

Server will run at `http://0.0.0.0:8000`

### 2. Calibration

#### a) Calibrate RealSense Camera
```bash
curl -X POST http://localhost:8000/calibrate_rs
```
- Place ArUco board (3x4, DICT_4X4_50, 30mm markers, 10mm separation) in view of RealSense
- Returns `T_world_rs` transformation and saves to `extrinsics/T_world_rs.json`

#### b) Calibrate AVP Camera
From VisionOS app, capture single RGB frame of the same ArUco board, then:
```bash
curl -X POST http://localhost:8000/calibrate_avp \
  -H "Content-Type: application/json" \
  -d '{"rgb_frame": "<base64>", "K": [[fx,0,cx],[0,fy,cy],[0,0,1]]}'
```

### 3. VisionOS App

1. Open `src/VisionOS/PoseOverlayApp/PoseOverlayApp.xcodeproj` in Xcode
2. Configure backend URL in app (e.g., `http://192.168.1.10:8000`)
3. Build and run on Vision Pro device
4. Select ROI, request poses
5. View dual overlay: RS camera pose (blue) + object pose (arrows)

## API Endpoints

### Calibration
- `POST /calibrate_rs` - Calibrate RealSense with ArUco
- `POST /calibrate_avp` - Calibrate AVP with RGB frame + intrinsics

### Pose Estimation
- `POST /estimate_pose` - Main endpoint
  - Input: `{rgb_frame_avp, mask, K_avp, model_name}`
  - Output: `{pose_rs_in_avp, pose_object_in_avp}`

### Head Pose
- `POST /head_pose` - Stream head pose for drift correction
  - Input: `{position, quaternion, timestamp}`

### Utilities
- `GET /health` - Check system status
- `GET /models` - List available .ply models

## Workflow

1. **Calibration (one-time)**:
   - Calibrate RealSense camera: `POST /calibrate_rs`
   - Calibrate AVP camera: `POST /calibrate_avp`
   - Both calibrations saved automatically

2. **Runtime**:
   - VisionOS streams head pose at 6.67Hz: `POST /head_pose`
   - User selects ROI/mask in VisionOS
   - VisionOS sends mask + RGB to backend: `POST /estimate_pose`
   - Backend:
     - Transforms mask from AVP → RS view
     - Captures RealSense RGB + depth
     - Calls FoundationPose API
     - Transforms object pose RS → AVP
     - Computes RS camera pose in AVP
   - Returns both poses
   - VisionOS renders dual overlay

## Key Simplifications

Compared to the original pipeline:
- ✅ Single depth source (RealSense only, no Transformers)
- ✅ Minimal API endpoints (6 instead of 15+)
- ✅ Clear coordinate frame management
- ✅ One RGB frame for mask, one for calibration
- ✅ Dual pose output (RS camera + object)
- ❌ Removed: Transformers depth, dual-mode switching, complex fallbacks

## Requirements

### Hardware
- Intel RealSense D435/D455 camera
- Apple Vision Pro
- Kubuntu/Ubuntu 20.04+ machine (or macOS)

### Software
- Python 3.8+
- Xcode 15.0+ (for VisionOS app)
- FoundationPose API running on port 5000

## Coordinate Frames

- **World Frame**: ArUco board origin
- **RealSense Frame**: RealSense camera
- **AVP Frame**: Apple Vision Pro camera
- **Object Frame**: Target object

### Transformations
```
T_rs_avp = inv(T_world_rs) @ T_world_avp @ T_head_correction
T_avp_object = inv(T_rs_avp) @ T_rs_object
```

## Troubleshooting

**RealSense not detected**:
```bash
rs-enumerate-devices
```

**Calibration fails**:
- Ensure ArUco board is clearly visible
- Good lighting conditions
- Board is planar and not distorted

**Mask transformation error**:
- Check calibration quality
- Verify head pose is streaming
- Ensure depth data is valid

## Performance

Typical latency on modern hardware:
- RealSense capture: ~33ms (30fps)
- Mask transformation: <50ms
- FoundationPose API: 100-300ms
- Total pipeline: 200-400ms

## License

Part of Master's Thesis: "Augmented Reality-Enhanced Programming by Demonstration"
by Ahmed Galai, 2025
