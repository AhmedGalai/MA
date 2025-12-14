# Kubuntu Backend Components

This directory contains the Python-based backend components for the AR Pose Estimation system, configured for Kubuntu/Linux environments.

## Overview

The backend provides 6D pose estimation services for the Apple Vision Pro AR application through:
- Flask-based REST API server
- Computer vision pipeline (ROI masking, intrinsics estimation)
- Integration with FoundationPose 6D pose estimation backend
- Optional RealSense depth camera support
- Screen capture for AirPlay mirroring

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
├── screen_capture.py         # AirPlay frame capture
└── main_api.py              # Standalone API entry point

models/                       # 3D mesh models (.ply files)
docs/                         # Additional documentation
```

## Requirements

### System Requirements
- Ubuntu 20.04+ / Kubuntu 20.04+
- Python 3.8+
- CUDA-capable GPU (recommended for FoundationPose)
- Intel RealSense D435/D455 (optional, for depth)

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

1. **Install system dependencies**:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
sudo apt install libopencv-dev  # OpenCV C++ libraries
```

2. **Create virtual environment**:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

3. **Install Python packages**:
```bash
pip install --upgrade pip
pip install -r final_pipeline/requirements.txt
```

4. **Install RealSense SDK** (if using depth camera):
```bash
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
sudo apt update
sudo apt install librealsense2-devel librealsense2-utils
```

## Usage

### Quick Start

1. **Start the main API server**:
```bash
cd backend/final_pipeline
python3 main_api.py
```

The API will be available at `http://0.0.0.0:5000`

2. **Configure Vision Pro app**:
- Open the Vision Pro application
- Enter API endpoint: `http://<KUBUNTU_IP>:5000`
- Select a 3D model from the dropdown
- Open immersive space to start pose estimation

### With Screen Capture

To capture AirPlay frames from Vision Pro:

1. **Setup AirPlay receiver** (using uxplay):
```bash
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base
git clone https://github.com/FDH2/UxPlay.git
cd UxPlay && mkdir build && cd build
cmake .. && make
sudo make install
```

2. **Start uxplay**:
```bash
uxplay -n "Kubuntu Backend"
```

3. **Mirror Vision Pro to uxplay**, then start capture:
```bash
python3 backend/screen_capture.py
```

### Configuration

Edit `backend/app_config.py` to customize:
- API host and port
- Depth mode (RealSense, monocular, or none)
- ROI parameters
- Model paths

## API Endpoints

### Health & Status
- `GET /health` - Check server availability
- `GET /stats` - Get processing statistics

### Configuration
- `GET /models` - List available .ply models
- `POST /select_model` - Set active model
- `GET/POST /config` - Read/update configuration

### Frame Processing
- `POST /receive_frame` - Receive AirPlay RGB frame
- `GET /rgb_frame` - Retrieve latest frame
- `GET /mask` - Get current ROI mask
- `GET /depth` - Get depth map

### Pose Estimation
- `POST /head_pose` - Receive headset pose from Vision Pro
- `POST /avp_pose` - Main pose estimation endpoint
  - Accepts: ROI circle, color filter, depth mode, model name
  - Returns: List of 4x4 transformation matrices

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 5000
sudo lsof -i :5000
# Kill the process
sudo kill -9 <PID>
```

### RealSense Not Detected
```bash
# Check connected devices
rs-enumerate-devices
# Check USB connection (must be USB 3.0)
lsusb | grep Intel
```

### OpenCV Import Errors
```bash
# Install system OpenCV
sudo apt install python3-opencv
# Or reinstall in venv
pip uninstall opencv-python
pip install opencv-python-headless
```

## Platform-Specific Notes

### Kubuntu vs MacOS Differences

1. **Networking**: Kubuntu may require firewall configuration
```bash
sudo ufw allow 5000/tcp
```

2. **Screen Capture**: Different window capture mechanisms
- Kubuntu: Uses X11/Wayland screen capture
- MacOS: Uses Quartz/ScreenCaptureKit

3. **RealSense**: Better Linux support, more stable drivers

## Related Documentation

- Main thesis: `../../latex/Masterarbeit/main.pdf`
- System architecture: Section 4 (System Design)
- Backend API details: Section 4.2 (Main API and Pose Backend)
- Development notes: Section 5 (Development)

## References

Based on the Master's Thesis:
"Augmented Reality-Enhanced Programming by Demonstration: 6D Pose Estimation Using Apple Vision Pro for Intuitive Robot Teaching"
by Ahmed Galai, 2025
