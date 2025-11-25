# VisionOS Application

This directory contains the native Apple Vision Pro application built with visionOS, SwiftUI, ARKit, and RealityKit.

## Overview

The VisionOS client is a mixed reality application that provides:
- **Gaze-based interaction** for object/region selection
- **Natural gesture controls** (pinch, tap) for ROI manipulation
- **Real-time 6D pose visualization** as AR overlays
- **Spatial UI windows** for configuration and debugging
- **Immersive space** for pose arrow rendering

The app acts as a thin front-end client that:
1. Captures user interaction (gaze, gestures)
2. Sends configuration to backend API
3. Receives 6D pose estimates
4. Renders pose overlays in mixed reality

## Project Structure

```
visionos/PoseOverlayApp/
├── PoseOverlayApp/              # Main application
│   ├── PoseOverlayAppApp.swift  # App entry point
│   ├── AppModel.swift           # Main app state
│   ├── ContentView.swift        # Main window UI
│   ├── ImmersiveSpaceView.swift # Immersive AR view
│   ├── ROIOverlayView.swift     # ROI window UI
│   ├── LogsView.swift           # Debug logs window
│   ├── SensorMonitorView.swift  # Sensor data window
│   │
│   ├── Services/                # Backend communication
│   │   ├── PoseService.swift    # Pose estimation API
│   │   ├── ModelService.swift   # Model selection API
│   │   └── HeadPoseService.swift # Headset pose streaming
│   │
│   ├── Models/                  # Data models
│   │   └── PoseResponse.swift   # API response types
│   │
│   ├── Utilities/               # Helper functions
│   │   └── MatrixUtils.swift    # Matrix conversions
│   │
│   └── Systems/                 # RealityKit systems
│       └── ArrowFactory.swift   # 3D arrow generation
│
├── PoseOverlayApp.xcodeproj     # Xcode project
├── Packages/                    # Swift packages
│   └── RealityKitContent/       # 3D assets package
└── PoseOverlayAppTests/         # Unit tests
```

## Requirements

### Hardware
- **Apple Vision Pro** headset
- **Mac** with Apple Silicon (M1/M2/M3) or Intel
  - Minimum: Mac Mini M2 (used in thesis)
  - Recommended: 16GB+ RAM

### Software
- **macOS** 14 (Sonoma) or later
- **Xcode** 15.0 or later
- **visionOS SDK** 1.0 or later
- **Apple Developer Account** (standard or enterprise)

### Network
- Vision Pro and backend server on **same WiFi network**
- Backend API running (see `../MacOS/README.md` or `../Kubuntu/README.md`)

## Installation

### 1. Open Project in Xcode

```bash
cd visionos/PoseOverlayApp
open PoseOverlayApp.xcodeproj
```

### 2. Configure Signing

1. Select **PoseOverlayApp** target
2. Go to **Signing & Capabilities** tab
3. Select your **Team** (Apple Developer Account)
4. Xcode will automatically handle provisioning

### 3. Set Bundle Identifier

If needed, change bundle identifier to match your team:
```
com.yourteam.PoseOverlayApp
```

### 4. Required Entitlements

Already configured in project (no changes needed):
- **ARKit World Tracking** - for headset pose
- **Network Access** - for backend communication

### 5. Info.plist Usage Descriptions

Already included:
```xml
<key>NSLocalNetworkUsageDescription</key>
<string>Required to connect to pose estimation backend API</string>

<key>NSMotionUsageDescription</key>
<string>Required for ARKit world tracking and headset pose</string>
```

## Building and Running

### On Vision Pro Simulator

1. Select **Apple Vision Pro** simulator
2. Click **Run** (⌘R)
3. App will launch in visionOS simulator

Note: Simulator has limitations:
- No real ARKit tracking
- No actual gaze/gesture input
- Mock backend connection only

### On Physical Vision Pro

1. **Connect Vision Pro** to Mac via USB-C
2. **Enable Developer Mode** on Vision Pro:
   - Settings > Privacy & Security > Developer Mode > ON
   - Restart device
3. **Trust computer** when prompted on Vision Pro
4. Select **Vision Pro** device in Xcode
5. Click **Run** (⌘R)
6. App installs and launches on headset

## Usage

### 1. Start Backend Server

On Mac or Kubuntu machine:
```bash
cd ../MacOS  # or ../Kubuntu
./start.sh
```

Note the displayed IP address (e.g., `http://192.168.1.10:5000`)

### 2. Launch Vision Pro App

1. Open **PoseOverlayApp** on Vision Pro
2. Main window appears with configuration options

### 3. Configure Connection

1. **API URL field**: Enter backend IP
   - Example: `http://192.168.1.10:5000`
2. Tap **Connect** or press Return
3. Status indicator turns green when connected

### 4. Select Model

1. **Model Picker**: Dropdown populated from backend
2. Select a `.ply` model (e.g., "Banana.ply")
3. Model is automatically set on backend

### 5. Configure ROI (Region of Interest)

1. Open **ROI window** (button in main window)
2. **Radius slider**: Adjust circle size (pixels)
3. **Color picker**: Set ROI outline color
   - Default: Cyan (#00FFFF)
   - Must be distinctive from target object
4. **Enable toggle**: Turn ROI masking on/off

### 6. Open Immersive Space

1. Tap **Open Immersive Space** button
2. Passthrough AR view activates
3. Physical environment visible with virtual overlays

### 7. Request Pose Estimation

1. **Look at target object** using gaze
   - ROI circle follows your gaze
   - Position circle over object
2. Tap **Start Polling** in main window
3. App continuously requests poses (1-3 Hz)
4. **3D arrow** appears at estimated object pose

### 8. Monitor Debug Info

1. Open **Logs window**
2. View:
   - API request/response times
   - Pose matrices (4×4)
   - Error messages
   - Connection status

## UI Components

### Main Window (Spatial)

Located in `ContentView.swift`

Controls:
- **API URL** text field
- **Model selector** dropdown
- **Depth mode** segmented control (RealSense / MDE / None)
- **Connect** / **Start Polling** / **Stop Polling** buttons
- **Open Immersive Space** button
- **Open ROI Window** button
- **Open Logs** button

### ROI Window (Spatial)

Located in `ROIOverlayView.swift` and `ROIWindowView.swift`

Controls:
- **Radius slider** (0-300 pixels)
- **Color picker** (HSV adjustable)
- **Enable/Disable toggle**
- Live preview of ROI circle parameters

### Logs Window (Spatial)

Located in `LogsView.swift`

Displays:
- Timestamped log entries
- Last received pose matrix
- API latency measurements
- Error messages with HTTP codes
- Scrollable list (newest at bottom)

### Immersive View

Located in `ImmersiveSpaceView.swift`

Renders:
- **3D pose arrow** at estimated object location
  - Red shaft pointing along object +X axis
  - Cone head indicating direction
- **ROI circle** overlay (when enabled)
  - Gaze-anchored at fixed depth
  - Thin outline with low opacity
- **Headset pose axes** (optional debug)

## Architecture

### Data Flow

```
User Interaction (Gaze + Gestures)
  ↓
[AppModel] - State management
  ↓
[Services] - API calls
  ├─ PoseService.swift      → POST /avp_pose
  ├─ ModelService.swift     → GET /models, POST /select_model
  └─ HeadPoseService.swift  → POST /head_pose
  ↓
Backend API (MacOS/Kubuntu)
  ↓
FoundationPose 6D Estimation
  ↓
Backend API Response (4×4 matrix)
  ↓
[MatrixUtils] - OpenCV → RealityKit conversion
  ↓
[ImmersiveSpaceView] - RealityKit rendering
  ↓
3D Arrow Overlay in AR Space
```

### Coordinate System Conversions

Located in `MatrixUtils.swift`

**OpenCV (Backend) coordinates**:
- +X: Right
- +Y: Down
- +Z: Forward (into scene)

**RealityKit (VisionOS) coordinates**:
- +X: Right
- +Y: Up
- +Z: Backward (out of screen)

Conversion applied:
```swift
let conversionMatrix = simd_float4x4(
    simd_float4(1,  0,  0, 0),
    simd_float4(0, -1,  0, 0),  // Flip Y
    simd_float4(0,  0, -1, 0),  // Flip Z
    simd_float4(0,  0,  0, 1)
)
```

### Concurrency Model

Uses Swift structured concurrency:

- **@MainActor**: UI updates and RealityKit scene changes
- **Background tasks**: API networking (async/await)
- **@Published properties**: Reactive state updates
- **Actor isolation**: Thread-safe service classes

Example:
```swift
@MainActor
class AppModel: ObservableObject {
    @Published var currentPose: simd_float4x4?

    func fetchPose() async {
        // Runs in background
        let pose = await poseService.requestPose(...)

        // Update UI on main actor
        await MainActor.run {
            self.currentPose = pose
        }
    }
}
```

## API Integration

### Endpoints Used

**Health Check**:
```swift
GET /health
Response: { "status": "ok" }
```

**List Models**:
```swift
GET /models
Response: ["Banana.ply", "Power Drill-ply.ply", ...]
```

**Select Model**:
```swift
POST /select_model
Body: { "model_name": "Banana.ply" }
```

**Stream Headset Pose**:
```swift
POST /head_pose
Body: {
    "position": [x, y, z],
    "quaternion": [qx, qy, qz, qw],
    "timestamp": 1699876543.21
}
```

**Request 6D Pose**:
```swift
POST /avp_pose
Body: {
    "roi_circle": {
        "center_px": [640, 360],
        "radius_px": 120
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

Response: {
    "poses": [
        {
            "matrix": [[...], [...], [...], [...]],  // 4×4
            "confidence": 0.95
        }
    ],
    "timestamp": 1699876543.21
}
```

## Limitations (ADP vs ADEP)

This app uses **Apple Developer Program (ADP)**, not ADEP.

### What's NOT Available:
- ❌ Direct raw camera frames
- ❌ Direct depth sensor access
- ❌ Low-level IMU readings

### Workarounds Implemented:
- ✅ AirPlay mirroring for RGB frames (captured on backend)
- ✅ ARKit world tracking for headset pose
- ✅ External RealSense for depth
- ✅ System-managed passthrough for background

These limitations are documented in thesis **Section 4** (System Design).

## Troubleshooting

### "Failed to Connect to Backend"

1. **Check network**:
   ```bash
   # On Vision Pro, test connectivity
   ping 192.168.1.10
   ```

2. **Verify backend running**:
   ```bash
   # On Mac/Kubuntu
   curl http://localhost:5000/health
   ```

3. **Check firewall** (on backend machine)

### "No Models Available"

Backend `/models` endpoint returned empty list:
- Ensure `.ply` files exist in `backend/../models/`
- Check backend logs for errors

### "Pose Arrow Not Appearing"

1. **Check immersive space** is open
2. **Verify polling** is started
3. **Look at object** - gaze determines ROI center
4. **Check logs window** for error messages
5. **Adjust ROI** radius and color if object not detected

### "Jittery Pose Overlay"

- **Increase polling interval** (reduce request frequency)
- **Improve lighting** conditions
- **Reduce object motion** during capture
- **Check backend latency** in logs

### Build Errors

**"No matching provisioning profiles"**:
- Select your team in Signing & Capabilities
- Ensure valid Apple Developer account

**"Missing visionOS SDK"**:
- Update Xcode to 15.0+
- Download visionOS simulator in Xcode > Settings > Platforms

## Development Notes

### Developed On

- **Hardware**: Mac Mini M2, 8-core CPU, 10-core GPU, 16GB RAM
- **Xcode**: Version 15.2
- **visionOS**: Version 1.0
- **Testing**: Physical Apple Vision Pro device

### Deployment Target

- **Minimum visionOS**: 1.0
- **Swift**: 5.9+
- **Architecture**: arm64 (Apple Silicon)

### Code Style

- SwiftUI for declarative UI
- Async/await for networking
- Combine for reactive state
- RealityKit for 3D rendering
- No external Swift package dependencies (except RealityKitContent)

## Performance

Typical metrics on Mac Mini M2 + Vision Pro:

- **Pose request latency**: 200-800ms (depends on backend)
- **ROI update rate**: 60 Hz (gaze tracking)
- **Render frame rate**: 90 Hz (VisionOS standard)
- **Network overhead**: <10ms (local WiFi)
- **Polling interval**: 1-3 Hz (configurable)

## Related Documentation

- Main thesis: `../../latex/Masterarbeit/main.pdf`
- VisionOS development: Section 5 (Development)
- System design: Section 4 (System Design)
- Backend APIs: `../MacOS/README.md`, `../Kubuntu/README.md`

## Future Enhancements

Potential improvements (see thesis Section 7 - Conclusion):

1. **Direct camera access** with ADEP
2. **On-device depth** using Vision Pro sensors
3. **Pose filtering** (Kalman, particle filter)
4. **Multi-object tracking**
5. **Robot controller integration** for PbD
6. **Gesture-based trajectory recording**

## References

Based on the Master's Thesis:
"Augmented Reality-Enhanced Programming by Demonstration: 6D Pose Estimation Using Apple Vision Pro for Intuitive Robot Teaching"
by Ahmed Galai, 2025

Developed at: Institute for Anthropomatics and Robotics (IAR), Karlsruhe Institute of Technology (KIT)

## License

Part of Master's Thesis submission. Academic use only.
