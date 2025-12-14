# Final Pipeline - Cleaned Depth Estimation with Pose Correction

A streamlined pipeline that uses RealSense for depth estimation with continuous headset pose correction.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FINAL PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DEPTH ACQUISITION (RealSense Fixed Camera)                  │
│     └─ Get metric depth from RealSense D435/D455                │
│                                                                  │
│  2. POSE STREAMS                                                 │
│     ├─ Continuous: AVP Headset Pose (streaming)                 │
│     └─ One-time: ArUco-based Calibration                        │
│         ├─ Real World Head Pose (ArUco on headset)              │
│         └─ Real World RealSense Pose (ArUco pattern)            │
│                                                                  │
│  3. PROBABILISTIC POSE CORRECTION                                │
│     └─ Use streamed headset data to correct transformations     │
│                                                                  │
│  4. COORDINATE TRANSFORMATION PIPELINE                           │
│     ├─ Transform mask from AVP view → RealSense view            │
│     ├─ Get 6D pose estimate in RealSense view                   │
│     └─ Transform final pose back to AVP view                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. `realsense_depth.py`
- RealSense D435/D455 interface
- Metric depth capture (aligned to color)
- Intrinsics extraction

### 2. `pose_manager.py`
- Headset pose streaming (continuous)
- ArUco-based calibration (one-time)
- Pose history for probabilistic correction

### 3. `coordinate_transformer.py`
- AVP ↔ RealSense coordinate transformations
- Probabilistic pose correction using Kalman filtering
- Mask projection between views

### 4. `pose_estimator.py`
- 6D object pose estimation in RealSense view
- Uses depth + mask for robust estimation

### 5. `pipeline_api.py`
- Clean REST API
- Single endpoint for complete pipeline execution

## Pipeline Flow

```
Input: RGB frame from AVP + Mask + Headset Pose
  ↓
[1] Capture RealSense Depth (fixed camera)
  ↓
[2] Get Current Headset Pose (streaming)
  ↓
[3] Apply Probabilistic Correction
    (using ArUco calibration + pose history)
  ↓
[4] Transform Mask: AVP → RealSense View
  ↓
[5] Estimate 6D Pose in RealSense View
    (using depth + transformed mask)
  ↓
[6] Transform Pose: RealSense → AVP View
  ↓
Output: Final 6D Pose in AVP coordinates
```

## Installation

```bash
# Required
pip install numpy opencv-python pyrealsense2 scipy filterpy

# Optional (for API)
pip install flask flask-cors
```

## Usage

### Quick Start

```python
from final_pipeline import FinalPipeline

# Initialize pipeline
pipeline = FinalPipeline()

# One-time calibration (place ArUco markers)
pipeline.calibrate_with_aruco()

# Process frames
result = pipeline.process_frame(
    avp_rgb=rgb_frame,
    avp_mask=mask,
    headset_pose=current_pose
)

# Get final pose
final_pose = result['pose_avp_view']
```

### API Server

```bash
python pipeline_api.py

# POST /process
{
    "rgb": "base64_image",
    "mask": "base64_mask",
    "headset_pose": {...}
}

# Response
{
    "pose": {
        "position": [x, y, z],
        "rotation": [rx, ry, rz]
    },
    "confidence": 0.95
}
```

## Calibration

### One-Time ArUco Calibration

1. **Setup**:
   - Attach ArUco marker to headset
   - Place ArUco board visible to RealSense

2. **Capture**:
   ```python
   pipeline.calibrate_with_aruco()
   ```

3. **Saves**:
   - `calibration/headset_to_world.json`
   - `calibration/realsense_to_world.json`
   - `calibration/avp_to_realsense.json`

## Coordinate Frames

- **World Frame**: ArUco board origin
- **AVP Frame**: Apple Vision Pro headset
- **RealSense Frame**: Fixed camera origin
- **Object Frame**: Target object being tracked

## Features

### Probabilistic Pose Correction
- Kalman filter for smooth pose estimates
- Handles noisy headset tracking
- Adapts to drift over time

### Robust 6D Pose Estimation
- Combines depth + mask for accuracy
- PnP solver with RANSAC
- Confidence scoring

### Performance
- RealSense depth: ~33ms @ 30fps
- Pose transformation: <5ms
- Total latency: <50ms

## File Structure

```
final_pipeline/
├── README.md                      # This file
├── __init__.py                    # Package init
├── config.py                      # Configuration
├── realsense_depth.py             # RealSense interface
├── pose_manager.py                # Pose streaming & calibration
├── coordinate_transformer.py      # Transform pipeline
├── pose_estimator.py              # 6D pose estimation
├── pipeline_core.py               # Main pipeline logic
├── pipeline_api.py                # REST API server
├── calibration/                   # Saved calibrations
│   ├── headset_to_world.json
│   ├── realsense_to_world.json
│   └── avp_to_realsense.json
└── utils/                         # Helper utilities
    ├── kalman_filter.py
    └── visualization.py
```

## Differences from Original Pipeline

### Removed
- ❌ Transformers-based depth (using RealSense only)
- ❌ Dual-mode depth switching
- ❌ HSV-based mask creation (mask comes from AVP)
- ❌ Unnecessary fallback logic

### Simplified
- ✅ Single depth source (RealSense)
- ✅ Clear coordinate transformation chain
- ✅ Probabilistic correction integrated
- ✅ One-time calibration workflow

### Added
- ✅ Continuous headset pose streaming
- ✅ Kalman filtering for pose correction
- ✅ Explicit coordinate frame management
- ✅ Confidence scoring

## Testing

```bash
# Test RealSense connection
python -m final_pipeline.realsense_depth

# Test calibration
python -m final_pipeline.pose_manager calibrate

# Test full pipeline
python -m final_pipeline.pipeline_core test
```

## Troubleshooting

**RealSense not detected**
- Check USB 3.0 connection
- Run: `rs-enumerate-devices`
- Install: `pip install pyrealsense2`

**Calibration fails**
- Ensure ArUco markers visible
- Check marker size configuration
- Verify lighting conditions

**High pose error**
- Recalibrate with ArUco
- Check RealSense depth quality
- Verify mask alignment
