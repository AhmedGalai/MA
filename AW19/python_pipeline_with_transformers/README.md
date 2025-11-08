# AW19 - Optimized AVP Vision Processing System

Complete vision processing system with GPU acceleration, monocular depth estimation, and AVP integration.

## Architecture

```
┌────────────────────────┐
│  screen_capture.py     │
│  (Capture UI)          │
│  - UI with sliders     │
│  - Highlights region   │
└──────────┬─────────────┘
           │ POST /receive_frame
           ↓
┌─────────────────────────────────┐
│  main_api.py                    │
│  (Main API Server)              │
│  - Receives frames              │
│  - Manages models               │
│  - Coordinates pipeline         │
└──────────┬──────────────────────┘
           │ Calls
           ↓
┌─────────────────────────────────┐
│  computer_vision_pipeline.py    │
│  (CV Pipeline - GPU Optimized)  │
│  - ArUco detection              │
│  - Pose estimation              │
│  - ROI mask extraction          │
│  - Monocular Depth Estimation   │
│  - GPU acceleration             │
└─────────────────────────────────┘
           │
           │ Disparity + Mask
           ↓
┌─────────────────────────────────┐
│  Pose (Integrated)              │
│  (in main_api.py)               │
│  - Mock or forward to real API  │
│  - Returns 6DOF pose            │
└─────────────────────────────────┘

    External:
┌─────────────────────────────────┐
│  AVP Device                     │
│  (Apple Vision Pro)             │
│  - Sends RGB + intrinsics       │
│  - Receives pose results        │
└─────────────────────────────────┘
```

## Components

### 1. **screen_capture.py**
- UI with sliders for capture region control
- Visual highlight of capture area
- Forwards frames to main API

### 2. **main_api.py**
- Main API server (port 5000)
- Receives RGB frames
- Coordinates CV pipeline
- Manages .ply models
- Forwards pose requests

### 3. **computer_vision_pipeline.py** ⭐ GPU Optimized
- **GPU Detection**: Automatically detects CUDA and uses GPU when available
- **ArUco Detection**: 3x4 board, DICT_4X4_50
- **Pose Estimation**: solvePnP with IPPE
- **ROI Masking**: HSV-based color segmentation
- **Monocular Depth Estimation**: Transformer model (Depth-Anything-V2)
- **Disparity Generation**: Converts depth to disparity maps
- **Optimizations**:
  - Lazy loading of depth model
  - GPU-accelerated inference
  - Efficient memory management
  - Thread-safe state management

### 4. **Integrated Pose Endpoint**
- `/pose` handled by `main_api.py`
- Mock or forward mode toggled by `use_random_pose` in `app_config.py` or `/config`
- For real forwarding, configure `pose_api.base_url` and `route` in `app_config.py`

### 5. **tk_debugging_client.py**
- Debug viewer for pipeline results
- Displays RGB, intrinsics, pose, mask, depth
- Configuration controls

## Installation

### 1. Install Dependencies

```bash
cd AW19
pip install -r requirements.txt
```

### 2. GPU Support (Optional but Recommended)

For CUDA GPU acceleration:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 3. Verify Models Directory

Ensure `.ply` models are in the `models/` folder:
```bash
ls models/
# Should show: ball.ply, cube.ply, cylinder.ply, etc.
```

## Usage

### Standard Workflow

**Terminal 1: Start Pose API**
```bash
python pose_api.py
```
Runs on http://localhost:9000

**Terminal 2: Start Main API**
```bash
python main_api.py
```
Runs on http://localhost:5000
- Automatically detects GPU
- Loads depth model on first use
- Shows device info on startup

**Terminal 3: Start Screen Capture**
```bash
python screen_capture.py
```
- Adjust sliders for capture region
- Click "Start Capture"
- Red highlight shows area

**Terminal 4: Start Debug Viewer (Optional)**
```bash
python tk_debugging_client.py
```

### AVP Integration Workflow

#### 1. List Available Models
```bash
curl http://localhost:5000/models
```
Response:
```json
{
  "models": [
    {"name": "ball.ply"},
    {"name": "cube.ply"},
    {"name": "cylinder.ply"}
  ]
}
```

#### 2. Select Model
```bash
curl -X POST http://localhost:5000/select_model \
  -H "Content-Type: application/json" \
  -d '{"model_name": "ball.ply"}'
```

#### 3. Request Pose Estimation

**Option A: With provided depth and mask**
```bash
curl -X POST http://localhost:5000/avp_pose \
  -H "Content-Type: application/json" \
  -d '{
    "rgb_frame": "<base64_image>",
    "depth_map": "<base64_depth>",
    "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "mask": "<base64_mask>",
    "model_name": "ball.ply",
    "depthscale": 1000.0
  }'
```

**Option B: Use pipeline disparity and mask**
```bash
curl -X POST http://localhost:5000/avp_pose \
  -H "Content-Type: application/json" \
  -d '{
    "rgb_frame": "<base64_image>",
    "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  }'
```
Will automatically use last disparity and mask from CV pipeline.

## API Endpoints

### Frame Processing
- `POST /receive_frame` - Receive RGB frame
  - Optional: `estimate_depth: true` to run MDE

### Configuration
- `GET /config` - Get CV pipeline config
- `POST /config` - Update HSV settings

### Results
- `GET /intrinsics` - Camera intrinsics
- `GET /pose` - ArUco board pose
- `GET /mask` - ROI mask
- `GET /depth` - Depth map (if MDE run)
- `GET /disparity` - Disparity map
- `GET /rgb_frame` - Raw frame
- `GET /detected_frame` - Frame with markers

### Models
- `GET /models` - List .ply models
- `GET /model?name=X` - Get specific model
- `POST /select_model` - Select model
- `POST /avp_pose` - Pose estimation

### Head Pose
- `POST /head_pose` - Send AVP tracking data
- `GET /head_pose` - Retrieve tracking data

### Status
- `GET /health` - Health check
- `GET /stats` - Pipeline statistics

## GPU Optimization Details

### Automatic GPU Detection
The system automatically detects and uses GPU when available:

```python
# In computer_vision_pipeline.py
DEVICE = detect_gpu()  # 'cuda' or 'cpu'
```

### GPU-Accelerated Components

1. **Depth Estimation** (Major speedup)
   - Transformer model runs on GPU
   - ~10-50x faster than CPU
   - Typical: 50-100ms on GPU vs 2-5s on CPU

2. **Color Conversion** (OpenCV CUDA)
   - If OpenCV built with CUDA support
   - BGR to HSV conversion accelerated

### Performance Benchmarks

**CPU (Intel i7)**:
- ArUco detection: ~10ms
- Pose estimation: ~5ms
- Mask extraction: ~5ms
- Depth estimation: ~2-5 seconds ❌

**GPU (NVIDIA RTX 3060)**:
- ArUco detection: ~10ms
- Pose estimation: ~5ms
- Mask extraction: ~5ms
- Depth estimation: ~50-100ms ✅

## Configuration

### CV Pipeline Settings

Update via API:
```python
import requests
requests.post('http://localhost:5000/config', json={
    "hsv_center": [90, 128, 128],  # H, S, V
    "h_tol": 12,                   # Hue tolerance
    "s_tol": 50,                   # Saturation tolerance
    "v_tol": 50                    # Value tolerance
})
```

### Model Paths

Edit in `main_api.py` and `app_config.py`:
```python
MODELS_DIR = "models"           # .ply files location
# In app_config.py: defaults.model_name and pose_api base_url/route
```

Edit in `computer_vision_pipeline.py`:
```python
# Change depth model
model_name = "depth-anything/Depth-Anything-V2-small-hf"  # Faster
model_name = "depth-anything/Depth-Anything-V2-large-hf"  # More accurate
```

## Troubleshooting

### GPU Not Detected

Check:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False:
1. Install CUDA-enabled PyTorch
2. Verify NVIDIA drivers installed
3. Check CUDA version compatibility

### Depth Model Loading Slow

First load downloads model (~1-2 GB):
- Stored in `~/.cache/huggingface/`
- Subsequent loads are fast
- Use smaller model for faster loading

### Out of Memory (GPU)

Reduce model size:
```python
# In computer_vision_pipeline.py
model_name = "depth-anything/Depth-Anything-V2-small-hf"
```

Or process fewer frames:
```python
# Only estimate depth on demand
estimate_depth = False  # Default in receive_frame
```

## Performance Tips

1. **Use GPU** for depth estimation
2. **Only estimate depth when needed** - set `estimate_depth: true` in `/receive_frame`
3. **Reduce capture FPS** if processing can't keep up
4. **Use smaller depth model** if GPU memory limited
5. **Batch processing** - send frames in groups for better GPU utilization

## Files

- `main_api.py` - Main API server
- `computer_vision_pipeline.py` - CV pipeline with GPU support
- `screen_capture.py` - Capture UI
- `app_config.py` - Central configuration
- `tk_debugging_client.py` - Debug viewer
- `models/` - .ply mesh files
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Future Enhancements

- [ ] Real pose estimation (replace mock)
- [ ] Multi-object tracking
- [ ] Temporal filtering for stability
- [ ] WebSocket streaming for lower latency
- [ ] Batch depth estimation
- [ ] Model caching
- [ ] TensorRT optimization
- [ ] ONNX export for mobile

---

**Version**: AW19
**Author**: Ahmed
**Last Updated**: 2025
