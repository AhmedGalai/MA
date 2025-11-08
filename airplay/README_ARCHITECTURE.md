# AVP Vision Processing System - Architecture

## Overview

The system is separated into **three independent components**:

1. **Screen Capture UI** (`screen_capture.py`) - Captures screen with adjustable parameters via UI
2. **Processing API** (`avp_api.py`) - Receives frames, processes them, and provides model endpoints
3. **Debug Viewer** (`tk_hypercam_2.py`) - Displays pipeline results for debugging

This distributed architecture allows for:
- Independent control of capture parameters
- Remote processing
- Multiple capture sources
- Multiple viewers
- Model selection and pose estimation for AVP
- Scalability and flexibility

---

## Architecture Diagram

```
┌──────────────────────┐
│  screen_capture.py   │
│  (Capture UI)        │
│                      │
│  - UI with sliders   │
│  - Grabs screen      │
│  - Highlights region │
│  - Encodes to JPEG   │
│  - Sends to API      │
└──────────┬───────────┘
           │ POST /receive_frame
           │ (RGB frames)
           ↓
┌──────────────────────────────┐
│    avp_api.py                │
│  (Processing API)            │
│                              │
│  - Receives frames           │
│  - Detects ArUco             │
│  - Solves pose               │
│  - Creates ROI mask          │
│  - Stores results            │
│  - Manages .ply models       │
│  - Forwards pose to external │
└──────────┬───────────────────┘
           │ GET /rgb_frame
           │ GET /pose
           │ GET /mask
           │ GET /models
           │ GET /model
           │ POST /select_model
           │ POST /avp_pose
           ↓
┌──────────────────────┐
│  tk_hypercam_2.py    │
│  (Debug Viewer)      │
│                      │
│  - Fetches results   │
│  - Displays pipeline │
│  - Debug mode only   │
└──────────────────────┘

    External Devices:
┌──────────────────────┐
│   AVP Device         │
│  (Apple Vision Pro)  │
└──────────┬───────────┘
           │ POST /head_pose (tracking data)
           │ POST /select_model (model selection)
           │ POST /avp_pose (final pose request)
           ↓
     (API receives and forwards)
           │
           ↓
┌──────────────────────┐
│  Pose Estimation API │
│  (localhost:9000)    │
│                      │
│  - Receives mesh     │
│  - Calculates pose   │
│  - Returns result    │
└──────────────────────┘
```

---

## Components

### 1. Screen Capture UI (`screen_capture.py`)

**Purpose**: Capture screen region with UI controls and forward RGB frames to API

**Features**:
- **UI with Sliders**: Adjust left, top, width, height, and FPS in real-time
- **Visual Highlight**: Red semi-transparent rectangle shows capture region
- **Live Stats**: Shows frames captured, sent, failed, FPS, and success rate
- **Start/Stop Control**: Start and stop capture with buttons
- **API Configuration**: Configurable API URL
- JPEG encoding for efficient transmission
- Performance statistics

**Usage**:
```bash
# Launch with default settings (934, 100, 812x1080 @ 30 FPS)
python screen_capture.py

# Launch with custom initial values
python screen_capture.py --left 0 --top 0 --width 1920 --height 1080 --fps 60

# Launch with remote API
python screen_capture.py --api http://192.168.1.100:5000
```

**Command-line Arguments** (initial values only, can be changed in UI):
- `--left`: Initial left position (default: 934)
- `--top`: Initial top position (default: 100)
- `--width`: Initial width (default: 812)
- `--height`: Initial height (default: 1080)
- `--fps`: Initial FPS (default: 30)
- `--api`: API base URL (default: http://localhost:5000)

**UI Controls**:
- **Sliders**: Adjust capture region and FPS (disabled during capture)
- **Start/Stop Buttons**: Control capture
- **Show Highlight**: Toggle red rectangle overlay
- **Status**: Real-time capture statistics

---

### 2. Processing API (`avp_api.py`)

**Purpose**: Receive frames, process through vision pipeline, and provide model management for AVP

**Features**:
- **Frame Processing**:
  - ArUco marker detection (DICT_4X4_50, 3x4 board)
  - Board pose estimation (rvec, tvec)
  - ROI mask generation (HSV-based)
  - Camera intrinsics calculation

- **Configuration Management**:
  - HSV color settings
  - Tolerance parameters
  - Runtime updates via API

- **Head Pose Integration**:
  - Receive head pose from AVP
  - Store position, rotation, quaternion
  - Calculate data age/staleness

- **Model Management**:
  - List available .ply models
  - Load and serve specific models
  - Model selection for pose estimation
  - Forward pose requests to external pose API

**Endpoints**:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/receive_frame` | Receive RGB frame for processing |
| GET/POST | `/config` | Get or update processing configuration |
| GET | `/intrinsics` | Camera intrinsics matrix |
| GET | `/pose` | Board pose (rvec, tvec) |
| GET | `/mask` | ROI mask image |
| GET | `/rgb_frame` | Raw RGB frame |
| GET | `/detected_frame` | Frame with ArUco markers drawn |
| POST | `/head_pose` | Send head pose data from AVP |
| GET | `/head_pose` | Get latest head pose data |
| GET | `/stats` | Processing statistics |
| GET | `/health` | Health check |
| **GET** | **/models** | **List available .ply models** |
| **GET** | **/model?name=X** | **Get specific .ply model** |
| **POST** | **/select_model** | **Select model for pose estimation** |
| **POST** | **/avp_pose** | **Final pose endpoint for AVP** |

**Usage**:
```bash
python avp_api.py
```

Server runs on `http://localhost:5000`

**Configuration**:
- `MODELS_DIR`: Path to .ply models folder (default: `../full_project_python/models`)
- `POSE_FORWARD_URL`: External pose API URL (default: `http://localhost:9000/pose`)

---

### 3. Debug Viewer (`tk_hypercam_2.py`)

**Purpose**: Display API pipeline results for debugging

**Features**:
- **Debug-only mode**: No capture control, only visualization
- **Tabbed display interface**:
  - RGB Feed - Raw frames from API
  - Camera Intrinsics - Computed intrinsics matrix
  - Board Pose - ArUco board rvec/tvec
  - ROI Mask - HSV-based mask
  - Detected Markers - Frames with ArUco markers drawn
  - Head Pose - AVP tracking data

- **Configuration controls** (HSV only):
  - HSV color selection for mask
  - Tolerance adjustment
  - UI refresh rate

- **Real-time status display**:
  - Frames processed
  - Pose availability
  - Head pose availability

**Usage**:
```bash
# Start debug viewer (ensure API is running first)
python tk_hypercam_2.py
```

**Note**: This is a debug-only tool. Screen capture parameters are controlled in `screen_capture.py`.

---

## Workflow

### Standard Workflow

1. **Start API Server**:
   ```bash
   python avp_api.py
   ```

2. **Start Screen Capture UI**:
   ```bash
   python screen_capture.py
   ```
   - Adjust sliders for capture region
   - Click "Start Capture"
   - Red highlight shows capture area

3. **Start Debug Viewer** (optional):
   ```bash
   python tk_hypercam_2.py
   ```

4. **Send Head Pose** (optional):
   ```bash
   python send_head_pose_example.py animated
   ```

### AVP Integration Workflow

For Apple Vision Pro integration:

1. **Start API Server** (on machine running processing):
   ```bash
   python avp_api.py
   ```

2. **List Available Models** (from AVP):
   ```bash
   curl http://localhost:5000/models
   ```
   Returns: `{"models": [{"name": "bunny.ply"}, {"name": "cube.ply"}, ...]}`

3. **Select Model** (from AVP):
   ```bash
   curl -X POST http://localhost:5000/select_model \
     -H "Content-Type: application/json" \
     -d '{"model_name": "bunny.ply"}'
   ```

4. **Request Pose Estimation** (from AVP):
   ```bash
   curl -X POST http://localhost:5000/avp_pose \
     -H "Content-Type: application/json" \
     -d '{
       "rgb_frame": "<base64_encoded_image>",
       "depth_map": "",
       "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
       "mask": "<base64_encoded_mask>",
       "depthscale": 1000.0
     }'
   ```

   Returns pose result from external pose API.

5. **Send Head Pose** (from AVP):
   ```bash
   curl -X POST http://localhost:5000/head_pose \
     -H "Content-Type: application/json" \
     -d '{
       "position": [0.0, 1.6, -0.5],
       "rotation": [0.1, 0.0, 0.0],
       "quaternion": [0, 0, 0, 1],
       "confidence": 0.95
     }'
   ```

### Remote Processing Workflow

**Machine A (Capture)**:
```bash
python screen_capture.py --api http://192.168.1.100:5000
```

**Machine B (Processing)**:
```bash
python avp_api.py
```

**Machine C (Viewing)**:
```bash
# Edit tk_hypercam_2.py: API_BASE_URL = "http://192.168.1.100:5000"
python tk_hypercam_2.py
```

---

## Data Flow

### Frame Processing Flow

```
Screen → Capture → [JPEG Encode] → API → [Decode] →
  ↓
[ArUco Detection] → [Pose Estimation] → [ROI Masking] →
  ↓
[Store Results] → [Serve to Clients]
```

### Configuration Flow

```
Client → POST /config → API → [Update Settings] →
  ↓
Screen Capture → GET /config → [Sync Settings]
```

### Head Pose Flow

```
AVP Device → POST /head_pose → API → [Store Pose] →
  ↓
Client → GET /head_pose → [Display]
```

---

## Configuration

### Screen Capture Configuration

Edit `screen_capture.py` or use command-line arguments:

```python
config = CaptureConfig(
    left=934,
    top=100,
    width=812,
    height=1080,
    fps=30,
    api_url="http://localhost:5000"
)
```

### API Configuration

Stored in API state, accessible via `/config` endpoint:

```json
{
  "hsv_center": [90, 128, 128],
  "tolerances": {
    "h": 12,
    "s": 50,
    "v": 50
  }
}
```

### Client Configuration

Edit `tk_hypercam_2.py`:

```python
API_BASE_URL = "http://localhost:5000"
```

---

## Performance Considerations

### Network Bandwidth

At 30 FPS with 812x1080 resolution and JPEG quality 85:
- **Frame size**: ~100-200 KB per frame
- **Bandwidth**: ~3-6 MB/s

Recommendations:
- Use local network for best performance
- Reduce FPS for remote processing
- Adjust JPEG quality if needed

### Processing Load

**CPU Usage**:
- Screen capture: ~2-5% (depends on resolution)
- API processing: ~10-30% (depends on complexity)
- Viewer: ~5-10%

**Memory Usage**:
- Screen capture: ~50 MB
- API: ~200-500 MB
- Viewer: ~100-200 MB

---

## Troubleshooting

### Screen Capture Issues

**"Cannot connect to API"**:
- Ensure API is running: `python avp_api.py`
- Check firewall settings
- Verify API URL is correct

**Low FPS**:
- Reduce target FPS
- Check network latency
- Reduce JPEG quality

### API Issues

**"No frames received"**:
- Ensure screen capture is running
- Check `/health` endpoint
- Verify network connectivity

**High CPU usage**:
- Reduce incoming frame rate
- Optimize ArUco detection parameters

### Viewer Issues

**"No data yet..."**:
- Ensure API is running
- Ensure screen capture is sending frames
- Check `/stats` endpoint

**Stale data**:
- Check network latency
- Verify screen capture FPS matches expectations

---

## Advanced Usage

### Multiple Capture Sources

Run multiple screen captures to the same API:

```bash
# Terminal 1: Capture from monitor 1
python screen_capture.py --left 0 --top 0 --width 1920 --height 1080

# Terminal 2: Capture from monitor 2
python screen_capture.py --left 1920 --top 0 --width 1920 --height 1080
```

(Note: API will process the most recent frame from any source)

### Custom Processing Pipeline

Modify `process_frame()` in `avp_api.py` to add custom processing:

```python
def process_frame(frame_bgr):
    # Your custom processing here
    custom_result = my_custom_detection(frame_bgr)

    # Store in state
    with state.lock:
        state.custom_result = custom_result

    # Existing processing continues...
```

### Headless Operation

Run API and capture without viewer:

```bash
# Start API
python avp_api.py &

# Start capture
python screen_capture.py

# Access results via REST API
curl http://localhost:5000/stats
curl http://localhost:5000/pose
```

---

## Migration from Old Architecture

**Old**: API captured screen directly
**New**: Separate capture program sends frames to API

**Benefits**:
- Distributed processing
- Easier to scale
- More flexible deployment
- Better separation of concerns

**Changes Required**:
- Start `screen_capture.py` separately
- Remove `/capture/start` and `/capture/stop` endpoints
- Client no longer controls capture

---

## Future Enhancements

- [ ] WebSocket support for lower latency
- [ ] Frame queue management
- [ ] Multiple API instances (load balancing)
- [ ] Recording/playback functionality
- [ ] Web-based viewer
- [ ] Docker containerization
- [ ] Authentication/authorization
- [ ] HTTPS support

---

## Files

- `screen_capture.py` - Screen capture forwarder
- `avp_api.py` - Processing API server
- `tk_hypercam_2.py` - Viewer client
- `send_head_pose_example.py` - Head pose sender example
- `requirements_api.txt` - Python dependencies
- `HEAD_POSE_API.md` - Head pose API documentation
- `README_ARCHITECTURE.md` - This file
