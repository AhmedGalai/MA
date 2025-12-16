# AVP (Apple Vision Pro) Integration with RealSense API

This document describes the integration of AVP screen mirroring with the RealSense pose estimation API, enabling coordinate transformation calculation and intrinsics calibration for both cameras.

## Architecture

```
┌──────────────────┐
│  Docker Container│
│     (UxPlay)     │  ← Receives AirPlay from visionOS
└────────┬─────────┘
         │ HTTP POST /receive_frame
         ↓
┌──────────────────┐
│    main_api.py   │  ← Stores latest AVP frame
│   (Port 8000)    │     Calculates intrinsics
└────────┬─────────┘     Transforms coordinates
         │
         ├→ RealSenseClient ─→ Intel RealSense D435i
         ├→ ArUco Detection ─→ Both cameras
         ├→ Intrinsics Calc ─→ Automatic calibration
         └→ Transformation  ─→ T_avp_rs matrix
```

## Features Implemented

### 1. **AVP Frame Receiving** (`/receive_frame`)
- Receives frames from UxPlay Docker container
- Stores most recent frame with timestamp
- Continuously discards old frames
- Metadata: width, height, receive time

**Endpoint:**
```http
POST /receive_frame
Content-Type: application/json

{
  "rgb_frame": "data:image/jpeg;base64,...",
  "timestamp": 1234567890.123,
  "purpose": "general"
}
```

### 2. **AVP Latest Frame** (`/get_avp_latest_frame`)
- Returns most recent AVP frame
- Includes timestamp and age
- Works independently of visionOS connection

**Endpoint:**
```http
GET /get_avp_latest_frame

Response:
{
  "rgb": "data:image/jpeg;base64,...",
  "timestamp": 1234567890.123,
  "age_seconds": 0.5,
  "width": 1280,
  "height": 720
}
```

### 3. **AVP ArUco Detection** (`/get_avp_aruco_frame`)
- Detects ArUco markers in AVP view
- **Automatically calculates AVP intrinsics** when enough samples collected (12+ frames)
- Draws markers with green borders and IDs
- Shows calculation progress

**Endpoint:**
```http
GET /get_avp_aruco_frame

Response:
{
  "rgb": "data:image/jpeg;base64,...",  // Annotated image
  "markers_detected": 3,
  "marker_ids": [0, 1, 2],
  "timestamp": 1234567890.123,
  "intrinsics_calculated": true,
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "samples_collected": 15
}
```

### 4. **RS ArUco Detection** (`/get_aruco_frame`)
- Enhanced to **automatically calculate RS intrinsics**
- Collects samples from detected markers
- Calculates camera matrix when ready (12+ frames)
- Shows notification when intrinsics calculated

**Response includes:**
```json
{
  "rgb": "data:image/jpeg;base64,...",
  "markers_detected": 4,
  "marker_ids": [0, 1, 2, 3],
  "timestamp": 1234567890.123,
  "intrinsics_calculated": true,
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "samples_collected": 20
}
```

### 5. **Intrinsics Retrieval** (`/get_intrinsics`)
- Get calculated intrinsics for both cameras
- Shows calculation status and method
- Includes timestamps

**Endpoint:**
```http
GET /get_intrinsics

Response:
{
  "rs": {
    "K": [[615.3, 0, 320.5], [0, 615.8, 240.2], [0, 0, 1]],
    "calculated": true,
    "method": "aruco_calibration",
    "timestamp": 1234567890.123
  },
  "avp": {
    "K": [[820.1, 0, 640.0], [0, 820.5, 360.0], [0, 0, 1]],
    "calculated": true,
    "method": "aruco_calibration",
    "timestamp": 1234567891.456
  }
}
```

### 6. **Coordinate Transformation** (`/get_transformation`)
- Calculates T_avp_rs transformation matrix
- Requires both cameras calibrated with ArUco board
- Returns transformation and individual calibrations

**Endpoint:**
```http
GET /get_transformation

Response:
{
  "calibrated": true,
  "T_avp_rs": [...],  // 4x4 transformation matrix
  "T_world_rs": [...],
  "T_world_avp": [...],
  "timestamp": 1234567890.123
}
```

## How It Works

### Intrinsics Calibration

Both cameras use the same calibration algorithm:

1. **Detection**: ArUco board detected in camera view
2. **Collection**: 2D image points and 3D object points collected
3. **Buffering**: Last 50 frames stored (rolling buffer)
4. **Ready Check**: When 12+ frames collected, calibration ready
5. **Calculation**: Uses OpenCV's `calibrateCamera()` with:
   - Zero tangential distortion
   - Fixed K3
   - Planar board (Z=0)

**ArUco Board Configuration:**
- Dictionary: DICT_4X4_50
- Grid: 3 rows × 4 columns
- Marker size: 30mm
- Separation: 10mm

### Transformation Calculation

When both cameras detect the same ArUco board:

1. **RS Calibration**: `T_world_rs` = transformation from world (board) to RS camera
2. **AVP Calibration**: `T_world_avp` = transformation from world (board) to AVP camera
3. **Relative Transform**: `T_avp_rs = inv(T_world_avp) @ T_world_rs`

This gives the transformation from RS camera space to AVP camera space.

## Docker Container Setup

The UxPlay container should continuously send frames:

```python
import requests
import cv2
import base64
import time

# Capture from UxPlay stdout
# ... (see app_2.py for full implementation)

while True:
    ret, frame = cap.read()
    if ret:
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Send to API
        requests.post('http://api:8000/receive_frame', json={
            'rgb_frame': f'data:image/jpeg;base64,{frame_b64}',
            'timestamp': time.time()
        })

    time.sleep(0.1)  # ~10 fps
```

## Usage Workflow

### 1. Start Systems
```bash
# Start Docker container with UxPlay
docker-compose up uxplay

# Start API
python3 main_api.py

# Start Debug Viewer
python3 debug_viewer.py
```

### 2. Connect visionOS Device
- Enable screen mirroring on Apple Vision Pro
- Connect to "Kubuntu Backend" (or configured name)
- Frames automatically stream to API

### 3. Calibrate Cameras
**RealSense:**
1. Hold ArUco board in front of RealSense
2. Move board slowly to different positions/angles
3. Watch debug viewer for "RS Intrinsics Calculated!" message
4. Intrinsics saved automatically

**AVP:**
1. Display ArUco board on screen visible in mirroring
2. Debug viewer shows AVP ArUco detection
3. Move board in view
4. Wait for "AVP Intrinsics Calculated!" message

### 4. Calculate Transformation
1. Hold ArUco board visible to BOTH cameras simultaneously
2. API automatically calibrates both cameras
3. Transformation T_avp_rs calculated
4. Debug viewer displays transformation matrix

## API Data Structures

### Intrinsics Storage
```python
{
    'K': np.ndarray,        # 3x3 camera matrix
    'calculated': bool,      # True when calculated
    'method': str,          # 'aruco_calibration'
    'timestamp': float      # Unix timestamp
}
```

### Calibration Buffer
```python
class IntrinsicsCalibBuffer:
    objpoints: List[np.ndarray]  # 3D board points
    imgpoints: List[np.ndarray]  # 2D image points
    img_size: Tuple[int, int]    # Image dimensions
    max_frames: int = 50         # Rolling buffer size

    def ready(min_samples=12) -> bool
    def calibrate() -> Optional[np.ndarray]
```

### Frame Metadata
```python
last_avp_frame_metadata = {
    'width': int,
    'height': int,
    'receive_time': float  # Unix timestamp
}
```

## Performance Notes

- **Frame Rate**: API accepts frames at any rate, keeps only latest
- **Intrinsics Calc**: ~12-20 samples needed, takes <1 second
- **Transformation**: Instant once both cameras calibrated
- **Memory**: Only latest AVP frame stored (~1-2 MB)

## Troubleshooting

### No AVP Frames
- Check Docker container running: `docker ps`
- Check visionOS device connected to UxPlay
- Verify network connectivity: `curl http://localhost:8000/health`

### Intrinsics Not Calculating
- Ensure ArUco board clearly visible
- Move board slowly to different positions
- Check samples collected: Look at `samples_collected` field
- Need 12+ diverse samples

### Transformation Not Available
- Both cameras must have intrinsics calculated
- ArUco board must be visible to BOTH cameras
- Check `/get_intrinsics` endpoint for both calculated

### Frames Too Old
- Check `age_seconds` in response
- If >1 second, Docker container may have stopped
- Restart UxPlay container

## Next Steps

1. **Debug Viewer Integration**: Add AVP panel to debug viewer
2. **Automatic Recalibration**: Reset intrinsics on camera change
3. **Persistent Storage**: Save intrinsics to file
4. **Multi-Board**: Support multiple ArUco boards
5. **Real-time Display**: Show transformation in debug viewer

## File Changes

**main_api.py:**
- Added `IntrinsicsCalibBuffer` class
- Added global storage for AVP frames and intrinsics
- Enhanced `/receive_frame` with metadata
- Added `/get_avp_latest_frame` endpoint
- Added `/get_avp_aruco_frame` endpoint
- Enhanced `/get_aruco_frame` with intrinsics calculation
- Added `/get_intrinsics` endpoint
- Added `/get_transformation` endpoint

**Files needed:**
- `aruco_detector.py` - ArUco detection
- `coordinate_manager.py` - Transformation management
- `config.py` - ArUco board configuration
