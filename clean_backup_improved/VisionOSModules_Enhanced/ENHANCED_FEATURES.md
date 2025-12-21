# VisionOSModules Enhanced - Changes Summary

## Overview

This enhanced version of VisionOSModules fixes the coordinate transformation issue between the AVP camera frame and the AVP world frame, enabling accurate 3D pose overlays in immersive space. It also adds continuous tracking so poses persist even when ArUco markers are temporarily occluded.

## Key Changes

### 1. Fixed Coordinate Transformation Bug

**Problem**: ArUco poses were detected in the AVP camera frame but rendered directly in world space without the necessary camera-to-world transformation. This caused large position and rotation errors.

**Solution**: Implemented proper transformation chain:
```
T_world_aruco = T_world_device × T_device_camera × T_camera_aruco
```

Where:
- `T_camera_aruco`: ArUco pose in camera frame (from detection)
- `T_device_camera`: Physical offset from device anchor to camera (~4cm forward, ~1cm down)
- `T_world_device`: Device/head pose in world (from ARKit DeviceAnchor)

### 2. Added Continuous Tracking

**Feature**: The app now maintains pose tracking even when the ArUco marker is temporarily not visible.

**How it works**:
1. When ArUco is detected, the app computes and stores the device-to-ArUco spatial relationship
2. When ArUco is not visible, the app uses the current device pose and stored relationship to estimate where the ArUco marker should be
3. The 3D visualization continues to display at the correct world location

**Benefits**:
- Smooth tracking without flickering when markers are briefly occluded
- Enables tracking during partial occlusions
- Better user experience for interactive applications

### 3. Removed Manual Gizmo Calibration

**Changed**: Removed the manual yellow gizmo anchor and slider-based calibration system.

**Replaced with**: Camera offset calibration sliders in the Anchor Setup window that fine-tune the estimated camera-to-device offset.

**Rationale**: The root cause was incorrect coordinate transformations, not calibration. With proper transforms, manual gizmo alignment is no longer necessary.

## New Files

### `/Utilities/CameraTransformUtils.swift`

Utility class for camera-to-world transformations:

- `estimatedCameraOffset`: Tunable camera offset (default: 0, -0.01, 0.04 meters)
- `cameraToWorldTransform(from:)`: Computes camera pose in world space
- `arucoPoseToWorld(cameraPose:deviceAnchor:)`: Transforms ArUco from camera to world
- `computeArucoToDeviceTransform(...)`: Stores device-relative transform for continuous tracking
- `estimateArucoWorldPose(...)`: Estimates pose when marker not visible

## Modified Files

### `/ArucoStreamModel.swift`

**Added**:
- `deviceToArucoTransform`: Stores device-relative transform for continuous tracking
- `isTracking`: Boolean indicating whether we're actively tracking (detection or estimation)

### `/SensorDataModel.swift`

**Changed**:
- Made `worldTracking` provider public (was private) to enable device anchor queries

### `/ImmersiveSpaceView.swift`

**Added**:
- ARKit import
- Enhanced `updateBoardAxes()` with camera-to-world transformation
- Continuous tracking logic (uses stored transform when marker not visible)
- Frame counter for periodic logging

**Removed**:
- Gizmo entity creation and rendering
- Slider-based translation/rotation controls
- Calibration buttons
- `updateWorldAnchorFromSliders()` method
- `sliderTranslation` and `sliderRotation` computed properties
- `makeAnchorGizmo()` function

### `/Views/AnchorSetupView.swift`

**Replaced**: Entire UI redesigned for camera offset calibration

**New features**:
- Camera offset sliders (X, Y, Z in meters)
- Real-time tracking status display
- Continuous tracking status indicator
- Reset to default button

## How to Use

### Initial Setup

1. Launch the app and connect to your Python backend
2. Open the Anchor Setup window
3. Start the 3D immersive view
4. Point the AVP at an ArUco marker

### Calibrating Camera Offset

If the 3D pose doesn't align perfectly with the physical marker:

1. Open the Anchor Setup window
2. Adjust the camera offset sliders:
   - **Z (forward)**: If pose is too far forward/back
   - **Y (down)**: If pose is too high/low (more negative = lower)
   - **X (right)**: If pose is offset left/right

3. Observe the 3D visualization in real-time
4. Fine-tune until alignment is accurate
5. Use "Reset to Default" to restore original values if needed

### Continuous Tracking

Once ArUco is detected:
- The "Continuous Tracking" status will show "Enabled"
- Move your head or temporarily occlude the marker
- The 3D pose will remain anchored to the physical location
- If you completely lose tracking, detect the marker again to re-establish

## Testing Checklist

### ✓ Basic Transform Verification
1. Place ArUco marker on flat surface
2. View from 1m away, directly facing marker
3. Verify 3D axes appear at marker location (not offset)

### ✓ Head Movement Tracking
1. Keep marker stationary
2. Walk around, tilt head, rotate
3. Verify 3D axes stay anchored to physical marker

### ✓ Distance/Scale Verification
1. Move closer/farther from marker
2. Verify distance in 3D matches physical distance

### ✓ Rotation Alignment
1. Rotate physical marker 90°
2. Verify 3D axes rotate exactly 90° in same direction

### ✓ Continuous Tracking
1. Detect marker (tracking status shows "Active")
2. Cover marker with hand
3. Verify pose remains visible and stable
4. Uncover marker and verify tracking resumes

## Technical Details

### Coordinate Systems

**OpenCV Camera** (from detection):
- X: right
- Y: down
- Z: forward

**RealityKit Camera**:
- X: right
- Y: up
- Z: backward

**Conversion**: Applied via `MatrixUtils.convertOpenCVToRealityKit()`

### ARKit Integration

The app queries `WorldTrackingProvider` for the device anchor at each frame:
```swift
let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: CACurrentMediaTime())
let T_world_device = deviceAnchor.originFromAnchorTransform
```

### Camera Offset Estimation

Default offset values (tunable via UI):
- X: 0.0m (no lateral offset)
- Y: -0.01m (1cm down from device anchor)
- Z: 0.04m (4cm forward from device anchor)

These are estimates based on AVP physical design. Fine-tune for your specific use case.

## Known Limitations

1. **No Enterprise API Access**: This implementation uses estimated camera offset rather than actual camera extrinsics (which require ADEP/Enterprise API)

2. **Tracking Drift**: Continuous tracking may accumulate small drift over time if the device anchor tracking drifts. Re-detect the marker periodically for best accuracy.

3. **Single Marker**: Currently tracks one ArUco board. Multiple markers would require separate tracking states.

## Comparison: Original vs Enhanced

| Feature | Original | Enhanced |
|---------|----------|----------|
| Coordinate transformation | ❌ Missing T_world_camera | ✅ Full transformation chain |
| 3D pose accuracy | ❌ Large errors | ✅ Accurate spatial alignment |
| Continuous tracking | ❌ Disabled when not visible | ✅ Persists during occlusion |
| Calibration method | Manual gizmo + sliders | Camera offset sliders |
| User experience | Frustrating alignment | Smooth, stable tracking |

## Future Enhancements

Potential improvements:
1. **Adaptive offset calibration**: Auto-calibrate camera offset using known marker positions
2. **Persistent storage**: Save calibrated camera offset across app launches
3. **Multi-marker tracking**: Track multiple ArUco markers simultaneously
4. **Confidence indicators**: Visual feedback on tracking quality
5. **Enterprise API integration**: Use actual camera extrinsics when ADEP available

## Troubleshooting

**Problem**: 3D pose still has offset after applying fix

**Solution**: Adjust camera offset in Anchor Setup window

---

**Problem**: Continuous tracking not working

**Solution**: Ensure ArUco marker is detected first (tracking status shows "Active"), then device-to-ArUco transform will be stored

---

**Problem**: Pose jumps when re-detecting marker

**Solution**: Normal behavior if tracking drifted. The new detection corrects to accurate position.

---

**Problem**: "Cannot query device anchor" errors in logs

**Solution**: Ensure ARKit permissions granted and immersive space is open

## Credits

Based on analysis of visionOS coordinate systems and ARKit documentation:
- Apple Developer Documentation: DeviceAnchor, WorldTrackingProvider
- Community research on AVP camera positioning
- RealityKit coordinate system conventions

---

**Version**: Enhanced v1.0
**Date**: 2024-12-21
**Original App**: VisionOSModules/PoseOverlayWithStreaming
