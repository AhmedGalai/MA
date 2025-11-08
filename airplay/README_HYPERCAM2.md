# HyperCam 2 - API-Based Screen Capture and Processing

## Overview

This system consists of two components that work together:
- **avp_api.py**: Flask API server for ArUco Vision Processing
- **tk_hypercam_2.py**: Tkinter client for screen capture and visualization

## Key Improvements

### Performance Optimizations
1. **Separated capture and display**: Client forwards frames to API and retrieves processed results, preventing UI freezing
2. **JPEG encoding**: Faster frame transmission (quality=85)
3. **Asynchronous API calls**: Non-blocking requests in separate threads
4. **30 FPS default**: Higher frame rate for smoother operation

### Default Parameters
- **Rectangle**: 934 x 1080 @ (934, 100)
- **Capture FPS**: 30
- **UI Refresh**: 30 Hz

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│ tk_hypercam_2   │         │   avp_api.py    │
│                 │         │                 │
│ Capture Thread  │──POST──>│ Frame Processor │
│                 │         │                 │
│ UI Thread       │<──GET───│ State Storage   │
│                 │         │                 │
└─────────────────┘         └─────────────────┘
```

### Client (tk_hypercam_2.py)
- **Capture Loop**: Grabs screen at configured FPS
- **Processing Loop**: Sends frames to API with HSV parameters
- **UI Loop**: Retrieves and displays processed data from API

### Server (avp_api.py)
- **Process frames**: Detect ArUco, solve pose, create ROI masks
- **Store state**: Thread-safe storage of latest results
- **Serve data**: Multiple GET endpoints for different outputs

## API Endpoints

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| POST   | /process_frame    | Submit frame for processing        |
| GET    | /intrinsics       | Camera intrinsics matrix           |
| GET    | /pose             | Board pose (rvec, tvec)            |
| GET    | /mask             | ROI mask image                     |
| GET    | /rgb_frame        | Raw RGB frame                      |
| GET    | /detected_frame   | Frame with ArUco markers drawn     |
| GET    | /stats            | Processing statistics              |
| GET    | /health           | API health check                   |

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements_api.txt
```

### 2. Start API Server
```bash
python avp_api.py
```

Server starts on `http://localhost:5000`

### 3. Start Client
```bash
python tk_hypercam_2.py
```

### 4. Configure and Run
1. Adjust capture region using sliders
2. Click "Apply Region"
3. Set HSV color and tolerances for ROI detection
4. Click "Start" to begin capture

## Display Tabs

1. **RGB Feed**: Raw captured frames (retrieved from API)
2. **Intrinsics**: Camera matrix and parameters
3. **Pose**: Rotation/translation vectors and distance
4. **Mask**: ROI mask based on HSV color selection
5. **Detected Markers**: Frame with ArUco markers highlighted

## ArUco Board Configuration

- **Dictionary**: DICT_4X4_50
- **Layout**: 3 rows × 4 columns
- **Marker Size**: 30mm
- **Separation**: 10mm

## Troubleshooting

### Client Freezes
- Ensure API server is running
- Check network latency to localhost
- Reduce FPS if necessary

### No Markers Detected
- Ensure good lighting
- Check ArUco board visibility
- Verify board configuration matches

### High CPU Usage
- Reduce capture FPS
- Lower UI refresh rate
- Decrease image quality in client

## Technical Notes

- Frame encoding uses JPEG (quality=85) for speed
- All API calls have 2-10 second timeouts
- Thread-safe state management with locks
- Automatic retry on failed API calls
- GC-safe PhotoImage references
