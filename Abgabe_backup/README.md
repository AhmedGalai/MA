# Master's Thesis Submission

**Title**: Augmented Reality-Enhanced Programming by Demonstration: 6D Pose Estimation Using Apple Vision Pro for Intuitive Robot Teaching

**Author**: Ahmed Galai
**Matriculation Number**: 10007404
**Institution**: Karlsruhe Institute of Technology (KIT)
**Institute**: Institute for Anthropomatics and Robotics (IAR)
**Thesis Type**: Master's Thesis
**Date**: November 2025

---

## Abstract

This thesis develops a proof-of-concept augmented reality (AR) application on the Apple Vision Pro for low-latency six-degree-of-freedom (6D) pose estimation, serving as a basis for intuitive AR-based Programming by Demonstration (PbD) systems. The system enables natural interaction through gaze and gestures, offloads pose estimation to an AI-based backend, and provides real-time visual feedback through AR overlays. The work demonstrates that modern mixed reality headsets can act as intuitive front-ends to perception backends, bridging human demonstrations and machine-readable object poses.

**Keywords**: Augmented Reality, Programming by Demonstration, 6D Pose Estimation, Apple Vision Pro, Human-Robot Collaboration, visionOS

---

## Repository Structure

```
Abgabe/
│
├── README.md                      # This file - submission overview
├── Todo.md                        # Project status, known issues, next steps
│
├── latex/                         # LaTeX thesis source and PDF
│   └── Masterarbeit/
│       ├── main.tex              # Main LaTeX document
│       ├── main.pdf              # Compiled thesis (if present)
│       ├── content/              # Chapter files
│       │   ├── introduction.tex
│       │   ├── research.tex      # State of the Art
│       │   ├── objective.tex     # Objectives & Methodology
│       │   ├── design.tex        # System Design
│       │   ├── development.tex   # Implementation
│       │   ├── evaluation.tex    # Evaluation Results
│       │   └── conclusion.tex    # Conclusions & Outlook
│       ├── figures/              # Images, diagrams, screenshots
│       ├── sources.bib           # Bibliography
│       └── format/               # LaTeX formatting templates
│
└── src/                          # Source code implementations
    │
    ├── Kubuntu/                  # Linux/Kubuntu backend
    │   ├── backend/             # Python Flask API server
    │   ├── models/              # 3D mesh models (.ply files)
    │   ├── docs/                # Additional documentation
    │   └── README.md            # Kubuntu-specific setup guide
    │
    ├── MacOS/                    # macOS backend (primary development)
    │   ├── backend/             # Python Flask API server
    │   ├── models/              # 3D mesh models (.ply files)
    │   ├── docs/                # Additional documentation
    │   ├── start.sh             # Automated startup script
    │   └── README.md            # MacOS-specific setup guide
    │
    └── VisionOS/                 # Apple Vision Pro application
        ├── visionos/            # Swift/visionOS project
        │   └── PoseOverlayApp/  # Xcode project
        │       ├── PoseOverlayApp/        # Main app code
        │       ├── PoseOverlayApp.xcodeproj  # Xcode project file
        │       └── Packages/              # Swift packages
        └── README.md            # VisionOS development guide
```

---

## System Overview

The system consists of four main components:

### 1. Apple Vision Pro Client (VisionOS)
- **Location**: `src/VisionOS/`
- **Technology**: Swift, SwiftUI, ARKit, RealityKit
- **Purpose**:
  - Gaze and gesture-based region-of-interest (ROI) selection
  - Configuration UI (model selection, depth mode)
  - 6D pose visualization as AR overlays
  - Headset pose streaming to backend

### 2. Python Backend API
- **Location**: `src/MacOS/backend/` or `src/Kubuntu/backend/`
- **Technology**: Python, Flask, OpenCV, NumPy
- **Purpose**:
  - AirPlay frame reception and processing
  - ROI mask reconstruction from UI parameters
  - Intrinsics estimation (ArUco markers)
  - Depth acquisition (RealSense or monocular)
  - Coordinate transformations
  - Pose request coordination

### 3. FoundationPose 6D Estimation Backend
- **External dependency** (collaborator's Docker container)
- **Purpose**: Deep learning-based 6D object pose estimation
- **Reference**: Cited in thesis Section 4.2

### 4. AirPlay Screen Mirroring
- **External tool**: UxPlay or similar AirPlay receiver
- **Purpose**: Capture RGB frames from Vision Pro (workaround for ADP limitations)
- **Reference**: Thesis Section 5.4

---

## Quick Start

### Prerequisites

- **Backend**: Mac (M1/M2) or Linux (Kubuntu 20.04+)
- **Vision Pro**: Apple Vision Pro headset
- **Network**: Both devices on same WiFi
- **Development**: Xcode 15+ (for VisionOS app)

### 1. Start Backend Server

**On MacOS**:
```bash
cd src/MacOS
./start.sh
# Note the displayed IP address (e.g., http://192.168.1.10:5000)
```

**On Kubuntu/Linux**:
```bash
cd src/Kubuntu
# Follow setup instructions in README.md
python3 backend/final_pipeline/main_api.py
```

### 2. Build and Deploy Vision Pro App

```bash
cd src/VisionOS/visionos/PoseOverlayApp
open PoseOverlayApp.xcodeproj
# In Xcode:
# 1. Select Apple Vision Pro device/simulator
# 2. Configure signing with your Apple Developer account
# 3. Build and run (⌘R)
```

### 3. Use the Application

1. **Launch app** on Vision Pro
2. **Enter backend URL**: `http://<BACKEND_IP>:5000`
3. **Select 3D model** from dropdown (e.g., "Banana.ply")
4. **Open immersive space**
5. **Look at target object** - ROI circle follows gaze
6. **Start pose polling** - 3D arrow overlays appear

For detailed instructions, see platform-specific READMEs.

---

## Key Features

### Implemented Functionality

✅ **Gaze-based ROI Selection**: Circular region-of-interest controlled by natural gaze direction
✅ **Gesture Interaction**: Pinch and tap gestures for UI control
✅ **Multi-window Spatial UI**: Configuration, ROI, debug, and logs windows
✅ **Low-latency Pose Estimation**: Interactive update rates (~1-3 Hz)
✅ **AR Pose Visualization**: 3D arrows rendered at estimated object locations
✅ **Multiple Depth Modes**: RealSense, monocular depth estimation, or none
✅ **Model Selection**: Runtime switching between 3D mesh models
✅ **Headset Pose Streaming**: Continuous pose correction using AVP tracking
✅ **Debug Logging**: Real-time diagnostics and error reporting

### Technical Highlights

- **Coordinate System Handling**: OpenCV ↔ RealityKit transformations
- **ROI Mask Reconstruction**: Backend reconstructs masks from UI parameters (no image upload needed)
- **AirPlay Workaround**: Bypasses ADP camera access limitations
- **ArUco Calibration**: Camera intrinsics estimation from marker boards
- **Modular Architecture**: Clean separation between frontend, backend, and pose estimation

---

## Technology Stack

### Vision Pro Application (Frontend)
- **Language**: Swift 5.9+
- **Frameworks**: SwiftUI, ARKit, RealityKit
- **Platform**: visionOS 1.0+
- **Concurrency**: Swift async/await, Combine

### Backend API (Server)
- **Language**: Python 3.8+
- **Framework**: Flask, Flask-CORS
- **Computer Vision**: OpenCV, NumPy
- **Depth**: pyrealsense2 (optional), transformers (MDE)
- **Deployment**: Standalone server or Docker

### Development Environment
- **Hardware**: Mac Mini M2, Apple Vision Pro
- **IDE**: Xcode 15.2 (VisionOS), PyCharm/VS Code (Python)
- **Version Control**: Git

---

## Thesis Chapters Reference

| Chapter | Title | Implementation |
|---------|-------|----------------|
| 1 | Introduction | N/A (motivation, overview) |
| 2 | State of the Art | N/A (literature review) |
| 3 | Objective and Methodology | System requirements definition |
| 4 | System Design | Architecture in `src/` directories |
| 5 | Development | VisionOS app in `src/VisionOS/` |
| 6 | Evaluation | Results documented in thesis |
| 7 | Conclusion | N/A (summary, outlook) |

### Mapping Sections to Code

- **Section 4.2** (Main API Backend) → `src/MacOS/backend/final_pipeline/main_api.py`
- **Section 4.3** (CV Pipeline) → `src/MacOS/backend/final_pipeline/pipeline_core.py`
- **Section 4.4** (Python Prototype) → `src/MacOS/backend/full_python_pipeline/`
- **Section 4.5** (AR Interface Design) → Described, implemented in VisionOS app
- **Section 5.3** (UI Components) → `src/VisionOS/visionos/PoseOverlayApp/PoseOverlayApp/`
- **Section 5.4** (AirPlay Capture) → `src/MacOS/backend/screen_capture.py`

---

## External Dependencies

The following components are referenced in the thesis but **not included** in this submission (external/collaborative):

### 1. FoundationPose 6D Pose Estimation API
- **Description**: Deep learning model for category-level 6D pose estimation
- **Status**: Collaborator's Docker container (cited in thesis)
- **Reference**: [matchcow_pose_api] in thesis bibliography
- **Integration**: Main API calls this via HTTP POST

### 2. Depth Estimation Models
- **AnyDepth v2**: Monocular depth transformer (HuggingFace endpoint)
- **ZoeDepth**: Alternative monocular depth model
- **Status**: Cloud-based inference or separate installation
- **Reference**: Thesis Section 4.3.3

### 3. AirPlay Receiver (UxPlay)
- **Description**: Open-source AirPlay mirroring receiver
- **Installation**: `brew install uxplay` (macOS) or from source (Linux)
- **Repository**: https://github.com/FDH2/UxPlay
- **Reference**: Thesis Section 5.4

### 4. 3D Model Acquisition
- **Models**: Banana, Power Drill, Spanner, Football, etc.
- **Format**: PLY (polygon mesh)
- **Source**: Various 3D model repositories
- **Location**: Included in `src/*/models/` directories

---

## Known Limitations

### Apple Developer Program (ADP) Restrictions

This implementation uses **standard ADP**, not Apple Developer Enterprise Program (ADEP):

- ❌ No direct access to raw camera frames
- ❌ No direct access to depth sensors
- ❌ No low-level IMU data

**Workarounds implemented**:
- ✅ AirPlay mirroring for RGB frames (captured externally)
- ✅ ARKit world tracking for headset pose
- ✅ External RealSense camera for depth
- ✅ System-managed passthrough for AR background

These limitations and workarounds are documented in **Thesis Section 4** (System Design).

### Other Limitations

- **Latency**: 200-800ms total (depends on network and backend)
- **Update Rate**: 1-3 Hz pose updates (interactive but not real-time)
- **Occlusion Handling**: Basic (advanced filtering described but not fully integrated)
- **Multi-object Tracking**: Single object at a time

---

## File Statistics

- **Python files**: 94 (backend implementation)
- **Swift files**: 18 (VisionOS application)
- **3D models**: 90 (.ply files across platforms)
- **LaTeX files**: 10 chapters + bibliography
- **Documentation**: 4 README files + this overview + Todo.md

**Total estimated size**: ~30-50 MB (excluding build artifacts)

---

## Building the Thesis PDF

To compile the LaTeX thesis:

```bash
cd latex/Masterarbeit
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `latex/Masterarbeit/main.pdf`

---

## Testing the Implementation

### Backend Health Check

```bash
curl http://localhost:5000/health
# Expected: {"status": "ok"}
```

### List Available Models

```bash
curl http://localhost:5000/models
# Expected: ["Banana.ply", "Power Drill-ply.ply", ...]
```

### Vision Pro Connection Test

1. Ensure backend running
2. Get backend IP: `ifconfig | grep "inet "`
3. In Vision Pro app, enter: `http://<IP>:5000`
4. Check debug logs window for connection status

---

## Platform-Specific Notes

### MacOS (Primary Development Platform)

- **Hardware used**: Mac Mini M2, 8-core CPU, 10-core GPU, 16GB RAM
- **Optimizations**: Native Apple Silicon support
- **Startup**: Automated via `start.sh` script
- **Screen capture**: Uses Quartz/ScreenCaptureKit

See: `src/MacOS/README.md`

### Kubuntu/Linux (Alternative Deployment)

- **Purpose**: Platform independence demonstration
- **Differences**: X11/Wayland screen capture, different networking setup
- **RealSense**: Better Linux driver support
- **Firewall**: May require UFW configuration

See: `src/Kubuntu/README.md`

### VisionOS (Vision Pro Client)

- **Target**: visionOS 1.0+
- **Development**: Requires Xcode 15+ and Apple Developer account
- **Simulator**: Limited functionality (no real ARKit)
- **Deployment**: USB-C connection to Mac required

See: `src/VisionOS/README.md`

---

## Troubleshooting

### Common Issues

**"Backend not reachable"**
- Check both devices on same WiFi
- Verify firewall allows port 5000
- Test with `curl http://<IP>:5000/health`

**"No models available"**
- Ensure `.ply` files exist in `backend/../models/`
- Check backend logs for loading errors

**"Pose overlay not appearing"**
- Verify immersive space is open
- Check polling is started
- Look at object (gaze determines ROI center)
- Review debug logs for errors

**"Build errors in Xcode"**
- Update to Xcode 15+
- Select valid signing team
- Download visionOS platform in Xcode settings

For detailed troubleshooting, see platform-specific README files.

---

## Project Timeline

- **Literature Review**: Research on PbD, AR, 6D pose estimation
- **Prototyping**: Web prototype, Python desktop prototype
- **Backend Development**: Flask API, CV pipeline, depth integration
- **VisionOS Development**: UI components, networking, rendering
- **Integration**: End-to-end system testing
- **Evaluation**: Performance metrics, usability testing
- **Documentation**: Thesis writing, code documentation

---

## Future Work

Potential improvements identified in thesis **Section 7** (Conclusion):

1. **Direct Camera Access**: Apply for ADEP to use Vision Pro sensors directly
2. **On-device Depth**: Utilize Vision Pro's depth sensors
3. **Advanced Filtering**: Integrate Kalman/particle filters for pose smoothing
4. **Multi-object Tracking**: Extend to simultaneous multiple objects
5. **Robot Integration**: Connect to robot controllers for full PbD workflow
6. **Trajectory Recording**: Capture gesture-based motion paths
7. **Real-time Performance**: Optimize for <50ms latency

---

## License and Usage

This work is submitted as part of a Master's Thesis at Karlsruhe Institute of Technology (KIT).

- **Academic use**: Permitted with proper citation
- **Commercial use**: Contact author/university
- **Reproduction**: Cite thesis and acknowledge IAR/KIT

**Citation**:
```
Galai, A. (2025). Augmented Reality-Enhanced Programming by Demonstration:
6D Pose Estimation Using Apple Vision Pro for Intuitive Robot Teaching.
Master's Thesis, Karlsruhe Institute of Technology (KIT).
```

---

## Contact Information

**Author**: Ahmed Galai
**Email**: (See thesis cover page)
**Institution**: Karlsruhe Institute of Technology (KIT)
**Institute**: Institute for Anthropomatics and Robotics (IAR)

For questions about:
- **Thesis content**: Contact author or supervisors
- **Code implementation**: See inline documentation and README files
- **External dependencies**: Refer to original projects (FoundationPose, UxPlay)

---

## Acknowledgments

- Supervisors and advisors at IAR/KIT
- Collaborators providing FoundationPose API
- Open-source communities (OpenCV, RealityKit, UxPlay)
- Apple Developer ecosystem and documentation

---

## Related Files

- **Todo.md**: Detailed project status, known issues, next steps
- **src/MacOS/README.md**: MacOS backend setup and usage
- **src/Kubuntu/README.md**: Linux backend setup and usage
- **src/VisionOS/README.md**: Vision Pro app development guide
- **latex/Masterarbeit/main.pdf**: Full thesis document

---

**Last Updated**: November 25, 2025
**Version**: 1.0 (Submission)
**Status**: Ready for submission
