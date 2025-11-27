# Full Python Pipeline - Unified Depth Estimation

This directory contains the complete unified pipeline supporting both RealSense hardware depth and Transformers-based monocular depth estimation.

## Directory Contents

### Core Pipeline Files

1. **`computer_vision_pipeline.py`** (19KB)
   - Main unified CV pipeline module
   - Supports both RealSense and Transformers depth
   - ArUco marker detection & pose estimation
   - ROI mask extraction (HSV-based)
   - Time synchronization (timestamps)
   - PNG depth encoding (API-compatible)
   - Automatic fallback system

2. **`app_config.py`** (1KB)
   - Central configuration
   - Contains `use_realsense` flag (depth mode selector)
   - HSV parameters, API endpoints, capture settings

### RealSense Components

3. **`realsense_adapter_adjusted.py`** (14KB)
   - Intel RealSense camera interface
   - Depth capture and alignment to color frame
   - AVP coordinate transformation (3D reprojection)
   - Z-buffering for proper occlusion handling
   - Extrinsics loading/calibration support

4. **`tk_rs_adjusted_panel.py`** (4KB)
   - RealSense-specific debug panel
   - Shows: RealSense color, depth, aligned disparity
   - Displays extrinsics calibration status

### Debug & Visualization

5. **`tk_debugging_unified.py`** (17KB)
   - Unified debug viewer for both depth modes
   - 6-panel display: RGB, Mask, Depth, Disparity, Colormaps
   - Proper depth scaling (0-255 normalization)
   - Timestamp monitoring for sync checking
   - Connection status indicator

6. **`__init__.py`** - Package initialization

7. **`README.md`** - This documentation

## Quick Start

### As Module (from parent directory):

```python
cd ../  # Go to AW19/
import full_python_pipeline.computer_vision_pipeline as cvp

# Process frame
result = cvp.process_frame(frame_bgr, estimate_depth=True)
print(f"Depth method: {result['depth_method']}")  # 'realsense' or 'transformers'
```

### Standalone Test:

```bash
cd full_python_pipeline
python computer_vision_pipeline.py

# Expected output:
# Device: cuda
# RealSense available: True/False
# Use RealSense: True/False
```

### Debug Viewer:

```bash
# From parent directory (AW19/):
python main_api.py                              # Terminal 1
python screen_capture.py                        # Terminal 2
python full_python_pipeline/tk_debugging_unified.py  # Terminal 3
```

## Configuration

### Depth Mode Selection

Edit `app_config.py`:

```python
"defaults": {
    "use_realsense": False,  # True = RealSense hardware, False = Transformers AI
    ...
}
```

**OR** via Runtime API:

```bash
# Switch to RealSense
curl -X POST http://localhost:5000/config -d '{"use_realsense": true}'

# Switch to Transformers
curl -X POST http://localhost:5000/config -d '{"use_realsense": false}'
```

## Key Features

### 1. Dual Depth Support

| Feature | Transformers Mode | RealSense Mode |
|---------|------------------|----------------|
| Model | Depth-Anything-V2 | D435/D455 Hardware |
| Speed (GPU) | 50-100ms | 33ms @ 30fps |
| Hardware | Any RGB camera | RealSense required |
| Depth Type | Relative | Metric (mm) |
| Accuracy | Scale ambiguous | Metric accurate |

### 2. Automatic Fallback

If `use_realsense=True` but RealSense unavailable:
- Library not installed
- Camera not connected
- Depth capture fails

→ **Automatically falls back to Transformers mode**

### 3. Time Synchronization

All pipeline data includes timestamps:
- `frame_timestamp`, `intrinsics_timestamp`
- `pose_timestamp`, `depth_timestamp`, `mask_timestamp`

Enables sync verification and stale data detection.

### 4. PNG Depth Encoding

Depth/disparity encoded as **PNG** (lossless):

```python
depth_png_base64 = encode_depth_as_png_base64(depth_array)
# Returns: "data:image/png;base64,iVBORw0KGgo..."
```

✅ **Compatible with PoseTestAPIrequest format**

### 5. Display Normalization

Depth values auto-scaled to 0-255 for visualization:

```python
depth_display = normalize_for_display(depth_raw)
# Input: any range (0-1, 0-10000mm, normalized, etc.)
# Output: 0-255 uint8 for clear display
```

Includes TURBO colormap for gradient visualization.

## Dependencies

### Required:
```bash
pip install numpy opencv-python torch transformers pillow requests flask flask-cors
```

### Optional (RealSense only):
```bash
pip install pyrealsense2  # Only if use_realsense=True
```

## RealSense Setup

### Hardware:
- Intel RealSense D435/D435i/D455
- USB 3.0 connection
- Calibrated camera

### Extrinsics Calibration:

Create `extrinsics_avp_from_rs_color.json`:

```json
{
  "R": [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ],
  "t": [0.0, 0.0, 0.0]
}
```

If not present, pipeline derives extrinsics from simultaneous ArUco detections (less stable).

## Testing

### Quick Import Test:
```bash
python -c "import full_python_pipeline.computer_vision_pipeline as cvp; print(f'Device: {cvp.DEVICE}, RealSense: {cvp.REALSENSE_AVAILABLE}')"
```

### RealSense Panel Test:
```bash
python full_python_pipeline/tk_rs_adjusted_panel.py
```

### Unified Debug Viewer:
```bash
# From parent directory
python main_api.py
python full_python_pipeline/tk_debugging_unified.py
```

## Troubleshooting

**"RealSense not available"**
- Install: `pip install pyrealsense2`
- Connect camera via USB 3.0
- Check: `rs-enumerate-devices`
- OR set `use_realsense=False`

**"Depth model loading failed"**
- Check internet (first download ~400MB)
- Install: `pip install transformers torch`
- Try smaller model: edit `DepthEstimator.__init__()`

**"Import error"**
- Ensure in parent directory (`AW19/`)
- Check `__init__.py` exists
- Use: `import full_python_pipeline.computer_vision_pipeline as cvp`

**"Depth not visible"**
- Check `estimate_depth=True` in request
- Use `tk_debugging_unified.py` (proper normalization)
- Verify depth inference succeeded (check logs)

## Performance

| Mode | Depth | Total Pipeline | GPU |
|------|-------|---------------|-----|
| Transformers | 50-100ms | 60-130ms | Recommended |
| RealSense | 33ms (hw) | 50-80ms | Optional |

## Integration

Used by:
1. **`main_api.py`** - Imports as `import computer_vision_pipeline as cvp`
2. **`screen_capture.py`** - Sends frames to API → pipeline
3. **`PoseOverlayApp`** (VisionOS) - Fetches depth/pose via API

## Architecture

```
computer_vision_pipeline.py
├── Depth Acquisition
│   ├── RealSense (if use_realsense=True)
│   │   └── realsense_adapter_adjusted.py
│   └── Transformers (fallback or primary)
│       └── Depth-Anything-V2 model
├── ArUco Detection & Pose
├── Mask Extraction (HSV)
├── Time Synchronization
└── PNG Encoding
```

## Version History

**v2.0 (2025-11-09)** - Unified Pipeline
- ✅ Merged RealSense + Transformers
- ✅ Time synchronization
- ✅ PNG encoding
- ✅ Automatic fallback
- ✅ Unified debug viewer

**v1.1** - RealSense Adjusted
- AVP coordinate alignment
- Z-buffer reprojection

**v1.0** - Transformers Only
- Basic depth estimation
- ArUco detection

## Documentation

See parent directory:
- `PIPELINE_IMPROVEMENTS.md` - Technical details
- `TESTING_GUIDE.md` - Verification tests
- `IMPLEMENTATION_SUMMARY.md` - Executive summary
