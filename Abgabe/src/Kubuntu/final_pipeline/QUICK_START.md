# Quick Start Guide - Final Pipeline

## Installation

```bash
cd final_pipeline
pip install -r requirements.txt
```

## Setup

### 1. Connect RealSense Camera
- Connect Intel RealSense D435/D455 via USB 3.0
- Verify connection: `rs-enumerate-devices`

### 2. Test RealSense
```bash
python -m final_pipeline.realsense_depth
```

Expected output:
```
Testing RealSense Depth Camera...
[RealSense] Camera initialized successfully
[RealSense] Intrinsics: fx=615.2, fy=615.5
Capturing 10 frames...
Frame 1: RGB shape=(480, 640, 3), Depth range=[300, 4000]mm
...
Test complete!
```

## Calibration (One-Time)

### Prepare ArUco Markers
1. **Headset marker**: Print ArUco marker ID 0 (5cm size)
   - Attach to headset where camera can see it
2. **Calibration board**: Print ArUco board (3x4 markers, 3cm each)
   - Place on flat surface visible to RealSense

### Run Calibration
```python
from final_pipeline import FinalPipeline
import numpy as np

# Initialize pipeline
pipeline = FinalPipeline()

# Load headset image with ArUco marker visible
headset_image = load_your_image()  # Your implementation

# Headset camera intrinsics
K_headset = np.array([
    [615, 0, 320],
    [0, 615, 240],
    [0, 0, 1]
], dtype=np.float32)
dist_headset = np.zeros(5, dtype=np.float32)

# Perform calibration
success = pipeline.calibrate_with_aruco(
    headset_image,
    K_headset,
    dist_headset
)

if success:
    print("Calibration successful! Files saved to calibration/")
else:
    print("Calibration failed - check marker visibility")
```

Calibration files saved to:
- `calibration/headset_to_world.json`
- `calibration/realsense_to_world.json`
- `calibration/avp_to_realsense.json`

## Usage

### Method 1: Python API

```python
from final_pipeline import FinalPipeline
import numpy as np
import cv2

# Initialize pipeline
pipeline = FinalPipeline()

# Load mask from AVP
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

# Current headset pose
headset_pose = {
    "position": [0.1, 0.2, 0.5],
    "rotation": [0.0, 0.0, 0.1]
}

# Process frame
result = pipeline.process_frame(
    avp_mask=mask,
    headset_pose=headset_pose
)

if result['success']:
    print("Pose in AVP view:")
    print(f"  Position: {result['pose_avp_view']['tvec']}")
    print(f"  Rotation: {result['pose_avp_view']['rvec']}")
    print(f"  Confidence: {result['confidence']:.3f}")
else:
    print(f"Error: {result['error']}")

# Cleanup
pipeline.shutdown()
```

### Method 2: REST API

#### Start Server
```bash
python -m final_pipeline.pipeline_api
```

Server runs on `http://localhost:5001`

#### API Endpoints

**Health Check**
```bash
curl http://localhost:5001/health
```

**Calibrate**
```bash
curl -X POST http://localhost:5001/calibrate \
  -H "Content-Type: application/json" \
  -d '{
    "headset_image": "data:image/jpeg;base64,...",
    "headset_intrinsics": {
      "K": [[615, 0, 320], [0, 615, 240], [0, 0, 1]],
      "dist": [0, 0, 0, 0, 0]
    }
  }'
```

**Process Frame**
```bash
curl -X POST http://localhost:5001/process \
  -H "Content-Type: application/json" \
  -d '{
    "mask": "data:image/png;base64,...",
    "headset_pose": {
      "position": [0.1, 0.2, 0.5],
      "rotation": [0.0, 0.0, 0.1]
    }
  }'
```

Response:
```json
{
  "success": true,
  "pose_avp_view": {
    "rvec": [0.1, 0.2, 0.3],
    "tvec": [0.5, 0.1, 1.2],
    "confidence": 0.95
  },
  "pose_rs_view": {
    "rvec": [0.15, 0.18, 0.25],
    "tvec": [0.45, 0.15, 1.1],
    "confidence": 0.95
  },
  "confidence": 0.95,
  "processing_time_ms": 42.3,
  "num_points": 1250
}
```

**Update Headset Pose (Streaming)**
```bash
curl -X POST http://localhost:5001/update_pose \
  -H "Content-Type: application/json" \
  -d '{
    "position": [0.1, 0.2, 0.5],
    "rotation": [0.0, 0.0, 0.1]
  }'
```

**Get Statistics**
```bash
curl http://localhost:5001/stats
```

**Get Pose History**
```bash
curl "http://localhost:5001/pose_history?duration=2.0"
```

## Pipeline Flow

```
1. CAPTURE REALSENSE DEPTH
   └─ Get metric depth from fixed camera

2. UPDATE HEADSET POSE
   └─ Receive streaming pose data
   └─ Apply Kalman filtering for correction

3. TRANSFORM MASK
   └─ Mask (AVP view) → Mask (RealSense view)
   └─ Uses calibrated transformation

4. ESTIMATE POSE
   └─ Use depth + mask in RealSense view
   └─ PCA or PnP-based pose estimation

5. TRANSFORM BACK
   └─ Pose (RealSense) → Pose (AVP view)
   └─ Final result in AVP coordinates
```

## Performance

| Component | Time |
|-----------|------|
| RealSense depth capture | ~33ms |
| Mask transformation | ~5ms |
| Pose estimation | ~8ms |
| Kalman filtering | <1ms |
| **Total** | **~50ms** |

## Troubleshooting

### "RealSense not available"
```bash
# Check connection
rs-enumerate-devices

# Reinstall driver
pip uninstall pyrealsense2
pip install pyrealsense2
```

### "Pipeline not calibrated"
- Run calibration first
- Check `calibration/` directory for saved files
- Ensure ArUco markers are visible and correct size

### "Pose estimation failed"
- Check mask quality (should have clear object region)
- Verify depth data is valid (not all zeros)
- Ensure object is within RealSense range (0.3m - 3m)

### "Low confidence score"
- Improve lighting conditions
- Use larger/clearer mask
- Ensure object has sufficient depth variation
- Check RealSense depth quality

## Advanced Usage

### Custom Object Model for PnP
```python
# Define 3D object model points
object_model = np.array([
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.1, 0.1, 0.0],
    [0.0, 0.1, 0.0]
], dtype=np.float32)

# Process with object model
result = pipeline.pose_estimator.estimate_pose_from_depth_and_mask(
    depth_map, mask, K, dist,
    object_model=object_model
)
```

### Adjust Kalman Filter Parameters
Edit `config.py`:
```python
KALMAN_CONFIG = {
    "process_noise": 0.01,      # Lower = smoother, higher = more responsive
    "measurement_noise": 0.1,   # Adjust based on sensor noise
    "initial_uncertainty": 1.0
}
```

### Custom Pose Estimation Parameters
Edit `config.py`:
```python
POSE_ESTIMATION_CONFIG = {
    "min_points": 4,            # Minimum points for valid pose
    "ransac_threshold": 3.0,    # RANSAC inlier threshold (pixels)
    "ransac_iterations": 100,
    "confidence_threshold": 0.7
}
```

## Directory Structure After Setup

```
final_pipeline/
├── calibration/
│   ├── headset_to_world.json         ✓ Created by calibration
│   ├── realsense_to_world.json       ✓ Created by calibration
│   └── avp_to_realsense.json         ✓ Created by calibration
├── *.py                               (Source files)
├── README.md
├── QUICK_START.md
└── requirements.txt
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Connect RealSense
3. ✅ Test camera
4. ✅ Run calibration
5. ✅ Process test frame
6. 🚀 Integrate with your application!

## Support

For issues, check:
1. RealSense connection: `rs-enumerate-devices`
2. Calibration files: `ls calibration/`
3. Server logs: Check console output
4. API health: `curl http://localhost:5001/health`
