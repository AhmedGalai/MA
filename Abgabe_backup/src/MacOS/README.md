# MacOS Backend Components

This directory contains the Python-based backend components for the AR Pose Estimation system, configured for MacOS environments.

## Overview

The backend provides 6D pose estimation services for the Apple Vision Pro AR application through:
- Flask-based REST API server
- Computer vision pipeline (ROI masking, intrinsics estimation)
- Integration with FoundationPose 6D pose estimation backend
- Optional RealSense depth camera support
- Native MacOS screen capture for AirPlay mirroring

## Directory Structure

```
backend/
├── final_pipeline/           # Current production pipeline
│   ├── main_api.py          # Main Flask API server
│   ├── pipeline_core.py     # Core CV pipeline logic
│   ├── pose_estimator.py    # Pose estimation integration
│   ├── realsense_depth.py   # RealSense camera interface
│   └── requirements.txt     # Python dependencies
├── full_python_pipeline/     # Legacy pipeline (reference)
├── screen_capture.py         # AirPlay frame capture (MacOS)
└── main_api.py              # Standalone API entry point

models/                       # 3D mesh models (.ply files)
docs/                         # Additional documentation
start.sh                      # Automated startup script (MacOS)
```

## Requirements

### System Requirements
- MacOS 12 (Monterey) or later
- Python 3.8+
- Apple Silicon (M1/M2) or Intel Mac
- Intel RealSense D435/D455 (optional, for depth)
- Xcode Command Line Tools

### Python Dependencies
```bash
pip install -r backend/final_pipeline/requirements.txt
```

Core dependencies:
- numpy>=1.24.0
- opencv-python>=4.8.0
- flask>=2.3.0
- flask-cors>=4.0.0
- pyrealsense2>=2.54.0 (for RealSense depth)
- torch>=2.0.0 (for pose estimation backend)

## Installation

1. **Install Homebrew** (if not installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. **Install Python and dependencies**:
```bash
brew install python@3.11
brew install opencv
```

3. **Install Xcode Command Line Tools**:
```bash
xcode-select --install
```

4. **Create virtual environment**:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

5. **Install Python packages**:
```bash
pip install --upgrade pip
pip install -r final_pipeline/requirements.txt
```

6. **Install RealSense SDK** (if using depth camera):
```bash
brew install librealsense
```

## Usage

### Quick Start (Automated)

Use the provided startup script:

```bash
# Start API only
./start.sh

# Start API + screen capture
./start.sh --with-capture

# Start API + debug viewer
./start.sh --with-debug

# Start all components
./start.sh --full

# Show help
./start.sh --help
```

The script will:
- Check Python installation
- Create/activate virtual environment
- Install dependencies
- Display local IP for Vision Pro connection
- Start the API server

### Manual Start

1. **Start the main API server**:
```bash
cd backend/final_pipeline
source ../venv/bin/activate
python3 main_api.py
```

The API will be available at `http://0.0.0.0:5000`

2. **Get local IP** for Vision Pro:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}'
```

3. **Configure Vision Pro app**:
- Open the Vision Pro application
- Enter API endpoint: `http://<MAC_IP>:5000`
- Select a 3D model from the dropdown
- Open immersive space to start pose estimation

### With Screen Capture

To capture AirPlay frames from Vision Pro:

1. **Install UxPlay** (AirPlay receiver):
```bash
brew install uxplay
```

2. **Start UxPlay**:
```bash
uxplay -n "Mac Backend"
```

3. **Mirror Vision Pro** to UxPlay, then start capture:
```bash
./start.sh --with-capture
```

Or manually:
```bash
python3 backend/screen_capture.py
```

### Configuration

Edit `backend/app_config.py` to customize:
- API host and port (default: 0.0.0.0:5000)
- Depth mode (RealSense, monocular, or none)
- ROI parameters
- Model paths
- HSV color thresholds

## API Endpoints

### Health & Status
- `GET /health` - Check server availability
- `GET /stats` - Get processing statistics

### Configuration
- `GET /models` - List available .ply models
- `POST /select_model` - Set active model for pose estimation
- `GET/POST /config` - Read/update configuration

### Frame Processing
- `POST /receive_frame` - Receive AirPlay RGB frame (base64)
- `GET /rgb_frame` - Retrieve latest buffered frame
- `GET /mask` - Get current ROI mask
- `GET /depth` - Get depth map
- `POST /external_depth` - Inject external depth data

### Pose Estimation
- `POST /head_pose` - Receive headset pose from Vision Pro
- `POST /avp_pose` - Main pose estimation endpoint
  - Accepts: ROI circle params, color filter, depth mode, model name
  - Returns: List of 4x4 transformation matrices (OpenCV format)

Example `/avp_pose` request:
```json
{
  "roi_circle": {
    "center_px": [640, 360],
    "radius_px": 150
  },
  "color_filter": {
    "h_min": 80, "h_max": 100,
    "s_min": 50, "s_max": 255,
    "v_min": 50, "v_max": 255
  },
  "depth_mode": "realsense",
  "model_name": "Banana.ply",
  "use_mock_pose": false
}
```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000
# Kill the process
kill -9 <PID>
```

### RealSense Not Detected
```bash
# Check connected devices
rs-enumerate-devices
# Check USB connection (must be USB 3.0)
system_profiler SPUSBDataType | grep Intel
```

### Virtual Environment Issues
```bash
# Remove and recreate venv
rm -rf backend/venv
python3 -m venv backend/venv
source backend/venv/bin/activate
pip install -r backend/final_pipeline/requirements.txt
```

### Network Connection Issues

1. **Check firewall**:
```bash
# MacOS firewall settings
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
```

2. **Verify same network**:
- Ensure Mac and Vision Pro are on the same WiFi network
- Disable VPN if active

3. **Test connectivity**:
```bash
# From Vision Pro, test connection
curl http://<MAC_IP>:5000/health
```

## Platform-Specific Notes

### MacOS vs Kubuntu Differences

1. **Networking**: MacOS typically has more restrictive firewall
   - May need to allow Python in System Preferences > Security & Privacy > Firewall

2. **Screen Capture**: Uses native MacOS APIs
   - Quartz/ScreenCaptureKit for window capture
   - Better performance than X11-based capture

3. **Development Environment**:
   - Optimized for Mac Mini M2 (used in thesis development)
   - Native support for Apple Silicon

4. **Startup Script**: MacOS-specific `start.sh`
   - Uses `ifconfig` for IP detection
   - Uses `lsof` for port checking
   - Includes log rotation

## Development Setup

This was the primary development platform for the thesis project.

**Hardware used**:
- Mac Mini (M2, 8-core CPU, 10-core GPU, 16GB RAM)
- Apple Vision Pro headset
- Intel RealSense D435 (optional)

**Development tools**:
- Xcode 15+ for Vision Pro app development
- Python 3.11 for backend
- PyCharm / VS Code for Python development

## Logs

When using `start.sh`, logs are written to:
```
logs/
├── main_api.log          # Main API server logs
├── screen_capture.log    # AirPlay capture logs
└── debug_viewer.log      # Debug UI logs
```

## Related Documentation

- Main thesis: `../../latex/Masterarbeit/main.pdf`
- System architecture: Section 4 (System Design)
- Backend API details: Section 4.2 (Main API and Pose Backend)
- Development notes: Section 5 (Development)
- VisionOS client: `../VisionOS/README.md`

## References

Based on the Master's Thesis:
"Augmented Reality-Enhanced Programming by Demonstration: 6D Pose Estimation Using Apple Vision Pro for Intuitive Robot Teaching"
by Ahmed Galai, 2025

Developed at: Institute for Anthropomatics and Robotics (IAR), Karlsruhe Institute of Technology (KIT)
