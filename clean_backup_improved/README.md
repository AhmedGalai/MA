# RealSense Pose Estimation API with Debug Viewer

This directory contains a complete setup for the RealSense-based pose estimation pipeline with a debug viewer for monitoring the system.

## What Was Done

### Files Copied from `../Abgabe_2/src/Kubuntu/`:
1. **main_api.py** - Flask REST API server
2. **debug_viewer.py** - Tkinter-based debug visualization tool
3. **config.py** - Configuration module (adjusted for model paths)
4. **realsense_client.py** - RealSense camera interface (improved with warm-up)
5. **realsense_manager.py** - RealSense management utilities
6. **aruco_calibration.py** - ArUco board calibration
7. **aruco_detector.py** - ArUco marker detection
8. **coordinate_manager.py** - Coordinate transformation management
9. **coordinate_transformer.py** - Coordinate transformation utilities
10. **mask_transformer.py** - Mask transformation between camera views
11. **foundationpose_client.py** - FoundationPose estimation client
12. **uxplay_capture.py** - UxPlay frame capture utility
13. **requirements.txt** - Python dependencies

### Key Adjustments Made

#### 1. Fixed RealSenseClient Method Name
- **Issue**: `main_api.py` was calling `realsense_client.get_frame()` but the method is named `capture()`
- **Fix**: Updated all 3 occurrences in main_api.py to use `capture()` and properly unpack the returned dictionary

#### 2. Fixed Color Format
- **Issue**: Incorrect color conversion (RGB to BGR) when RealSense already returns BGR
- **Fix**: Removed unnecessary color conversion in get_rgbd_frame endpoint

#### 3. Improved Camera Initialization
- **Issue**: RealSense camera timing out on frame capture ("Frame didn't arrive within 1000")
- **Fixes**:
  - Added 30-frame warm-up period after camera start (~1 second)
  - Increased capture timeout from 1000ms to 5000ms
  - Better error handling during warm-up

#### 4. Configured Model Paths
- **Issue**: Models directory pointed to local directory
- **Fix**: Updated `config.py` to use `../Abgabe/src/Kubuntu/models` for accessing .ply files

#### 5. Created Directory Structure
- Created `extrinsics/` directory for storing calibration data

## Directory Structure

```
/home/ag/Desktop/MA/clean/
├── main_api.py                 # Flask API server
├── debug_viewer.py             # Debug visualization tool
├── config.py                   # Configuration (points to ../Abgabe/.../models)
├── realsense_client.py         # RealSense camera interface
├── realsense_manager.py        # RealSense utilities
├── aruco_calibration.py        # ArUco calibration
├── aruco_detector.py           # ArUco detection
├── coordinate_manager.py       # Coordinate management
├── coordinate_transformer.py   # Coordinate transformations
├── mask_transformer.py         # Mask transformations
├── foundationpose_client.py    # Pose estimation client
├── uxplay_capture.py           # UxPlay capture
├── requirements.txt            # Dependencies
├── extrinsics/                 # Calibration data storage
└── README.md                   # This file
```

## Available Features

### 1. RealSense RGB & Depth Streaming
- **Works independently** without visionOS app connection
- Endpoint: `GET /get_rgbd_frame`
- Returns base64-encoded RGB and depth colormap images
- Accessible via debug viewer

### 2. ArUco Pattern Detection
- Detect ArUco board in RealSense view
- Endpoint: `POST /calibrate_rs`
- Calculate camera pose relative to world frame
- Works independently of visionOS

### 3. Calculated RS Pose
- Get RealSense camera pose in world coordinates
- Available after calibration
- Part of coordinate transformation system

### 4. Debug Viewer Features
- **System Status** panel showing RS connection and calibration state
- **RealSense RGB** view with live camera feed
- **RealSense Depth** view with colormap visualization
- **Statistics** panel with frame timing and success rates
- **Configurable refresh rate** (1-10 Hz)
- **Manual refresh** button
- **UxPlay capture controls** (for ArUco calibration and ROI selection)

### 5. Model Management
- Endpoint: `GET /models`
- Lists available .ply files from `../Abgabe/src/Kubuntu/models/`
- Available models:
  - Banana.ply
  - Football.ply
  - Power Drill-ply.ply
  - Screw.ply
  - Spanner-ply.ply
  - ball.ply
  - cube.ply
  - cylinder.ply
  - rectangle.ply

## Usage

### Starting the API Server

```bash
cd /home/ag/Desktop/MA/clean
python3 main_api.py
```

The API will:
1. Initialize on `http://0.0.0.0:8000`
2. Attempt to connect to RealSense camera
3. Load existing calibration if available
4. Start serving requests

### Starting the Debug Viewer

```bash
python3 debug_viewer.py
```

Or with custom settings:
```bash
python3 debug_viewer.py --api-url http://localhost:8000 --width 1920 --height 1080
```

### Using the Debug Viewer

1. **Connect**: Click "Connect" button to start polling the API
2. **View Streams**: RGB and Depth views update at configured refresh rate
3. **Adjust Refresh Rate**: Use slider (1-10 Hz)
4. **Manual Refresh**: Click "Manual Refresh" for immediate update
5. **Clear Stats**: Reset statistics counters

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Get RGBD Frame
```bash
curl http://localhost:8000/get_rgbd_frame
```

#### List Models
```bash
curl http://localhost:8000/models
```

#### Calibrate RealSense
```bash
curl -X POST http://localhost:8000/calibrate_rs
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Main dependencies:
- Flask & flask-cors (API server)
- pyrealsense2 (RealSense SDK)
- opencv-python (image processing)
- numpy (numerical operations)
- Pillow (image handling for GUI)
- requests (HTTP client for debug viewer)

## Configuration

Edit `config.py` to adjust:
- Network settings (host, port)
- ArUco board parameters
- RealSense resolution and FPS
- File paths
- Processing parameters

## Troubleshooting

### Camera Not Connecting
- Check USB connection
- Verify RealSense is not used by another application
- Check permissions: `sudo usermod -a -G video $USER`

### Frame Timeout Errors
- Camera warm-up period has been extended to 30 frames
- Timeout increased to 5000ms
- If issues persist, check USB bandwidth

### Debug Viewer Not Connecting
- Verify API is running: `curl http://localhost:8000/health`
- Check firewall settings
- Ensure correct API URL in viewer

### Models Not Found
- Verify path: `../Abgabe/src/Kubuntu/models/` exists
- Check .ply files are present
- Verify read permissions

## Notes

- All RealSense features work **independently** of visionOS app connection
- Calibration data is stored in `extrinsics/T_world_rs.json`
- Camera warm-up takes ~1 second on startup
- Debug viewer polls API at configurable rate (default 2 Hz)
- RGB format from RealSense is BGR (OpenCV format)

## Next Steps

1. **Test Camera**: Start API and verify RealSense connects
2. **Test Debug Viewer**: Launch viewer and connect to API
3. **Verify Streams**: Check RGB and Depth views update correctly
4. **Test ArUco Detection**: Place ArUco board in view and calibrate
5. **Verify Models**: Check `/models` endpoint returns .ply files

## Architecture

```
┌─────────────────┐
│  Debug Viewer   │  (Tkinter GUI)
│   (Port N/A)    │
└────────┬────────┘
         │ HTTP Polling
         ↓
┌─────────────────┐
│   main_api.py   │  (Flask Server)
│   (Port 8000)   │
└────────┬────────┘
         │
         ├→ RealSenseClient ─→ Intel RealSense Camera
         ├→ ArUco Detection ─→ Calibration
         ├→ CoordinateManager ─→ Transformations
         └→ Models Directory ─→ ../Abgabe/src/Kubuntu/models/
```
