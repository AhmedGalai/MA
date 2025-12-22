# Quick Start Guide - VisionOSModules Enhanced

## Overview

This is an enhanced version of the VisionOSModules app with fixed coordinate transformations and continuous ArUco tracking.

## What's New

✅ **Fixed 3D pose accuracy** - ArUco markers now appear at correct world locations
✅ **Continuous tracking** - Poses persist even when markers are temporarily occluded
✅ **Camera offset calibration** - Fine-tune camera positioning in real-time
✅ **Removed manual gizmo** - Simpler, more accurate workflow

## Building the App

### Prerequisites

- Xcode 15.2 or later
- visionOS SDK
- Apple Vision Pro (physical device or simulator)
- Python backend running (for ArUco detection)

### Build Steps

1. Open the project in Xcode:
   ```bash
   cd VisionOSModules_Enhanced/PoseOverlayWithStreaming
   open PoseOverlayWithStreaming.xcodeproj
   # or if using Xcode command line:
   xed .
   ```

2. Select your target device (Vision Pro)

3. Build and run (⌘R)

## First Run

### 1. Connect to Backend

- Enter your Python backend IP address
- Click "Connect"
- Verify "Connected" status appears

### 2. Open Immersive View

- Click "Show 3D View" button
- Grant ARKit permissions if prompted
- The immersive space will open

### 3. Detect ArUco Marker

- Point your AVP at an ArUco marker board
- The app will detect and display:
  - 2D overlay in the camera view
  - 3D axes in immersive space

### 4. Verify Alignment

- The 3D axes should align with the physical marker
- Walk around the marker - it should stay anchored in place
- Cover the marker briefly - tracking should continue (continuous tracking)

## Camera Offset Calibration

If the 3D pose doesn't align perfectly:

1. **Open Anchor Setup window** (from main window)

2. **Adjust sliders**:
   - **Z (forward)**: Most impactful - adjust if pose is in front/behind marker
   - **Y (down)**: Adjust if pose is above/below marker
   - **X (right)**: Adjust if pose is left/right of marker

3. **Fine-tune in real-time** while viewing the 3D immersive space

4. **Reset to default** if you want to start over

### Recommended Calibration Process

1. Place marker on flat surface (e.g., table)
2. Stand 1 meter away, directly facing marker
3. Observe 3D axes alignment
4. Adjust Z-offset first (usually between 0.03-0.06m)
5. Adjust Y-offset second (usually between -0.02 to 0.00m)
6. Fine-tune X-offset last (usually close to 0.0m)

## Usage Tips

### Best Practices

- **Good lighting**: Ensure ArUco marker is well-lit for reliable detection
- **Steady viewing**: Keep marker in view for 1-2 seconds for initial detection
- **Periodic re-detection**: If continuous tracking drifts, re-detect marker
- **Known distances**: Use markers at known positions to verify calibration

### Tracking Status

Monitor tracking in the Anchor Setup window:

- **ArUco Tracking: Active** - Currently detecting marker
- **ArUco Tracking: Inactive** - No detection
- **Continuous Tracking: Enabled** - Device-to-marker relationship stored
- **Continuous Tracking: Waiting** - Need to detect marker first

### Debugging

Enable detailed logging in ImmersiveSpaceView.swift (already included):
- Position logs every 60 frames (1 second at 60Hz)
- Continuous tracking status every 120 frames

Check Xcode console for:
```
ArUco world pos: [x, y, z]
📍 Continuous tracking (ArUco not visible)
⚠️ Cannot query device anchor
```

## Testing Scenarios

### Scenario 1: Static Marker
- Place marker on table
- View from different angles
- **Expected**: Pose stays anchored to marker position

### Scenario 2: Head Movement
- Keep marker stationary
- Walk around, rotate head, tilt
- **Expected**: Marker stays in correct world location

### Scenario 3: Occlusion
- Detect marker
- Cover with hand or object
- Move head while occluded
- Uncover marker
- **Expected**: Pose remains visible and stable during occlusion

### Scenario 4: Multiple Detections
- Detect marker
- Look away (marker leaves camera view)
- Look back at marker
- **Expected**: Detection resumes, pose updates to accurate position

## Common Issues

### Issue: "Cannot query device anchor" errors

**Cause**: ARKit not initialized or immersive space not open

**Fix**:
1. Ensure you clicked "Show 3D View"
2. Grant ARKit permissions if prompted
3. Wait 2-3 seconds for ARKit to initialize

---

### Issue: 3D pose has constant offset from marker

**Cause**: Camera offset needs calibration

**Fix**: Adjust camera offset in Anchor Setup window (see calibration section above)

---

### Issue: Pose jumps when moving head

**Cause**: Normal behavior if marker leaves camera view and returns

**Fix**: This is expected - the new detection corrects position. For smooth tracking, keep marker in view.

---

### Issue: Continuous tracking drifts over time

**Cause**: ARKit device tracking accumulates small errors

**Fix**: Periodically re-detect the marker to correct drift

---

### Issue: No ArUco detection at all

**Cause**: Backend not running or marker not visible

**Fix**:
1. Verify Python backend is running
2. Check network connection
3. Ensure marker is in camera view
4. Check lighting conditions

## Differences from Original App

| Feature | Original | Enhanced |
|---------|----------|----------|
| Gizmo anchor | ✅ Yellow sphere + axes | ❌ Removed |
| Manual sliders | ✅ Translation + rotation | ✅ Camera offset only |
| Calibration button | ✅ Save to UserDefaults | ❌ Removed |
| Continuous tracking | ❌ No | ✅ Yes |
| 3D accuracy | ❌ Large errors | ✅ Correct alignment |

## File Structure

```
VisionOSModules_Enhanced/
├── PoseOverlayWithStreaming/
│   ├── Utilities/
│   │   ├── CameraTransformUtils.swift  ← NEW: Camera transformations
│   │   └── MatrixUtils.swift
│   ├── Views/
│   │   └── AnchorSetupView.swift      ← MODIFIED: Camera calibration UI
│   ├── ArucoStreamModel.swift         ← MODIFIED: Added tracking state
│   ├── SensorDataModel.swift          ← MODIFIED: Exposed worldTracking
│   └── ImmersiveSpaceView.swift       ← MODIFIED: Enhanced transforms
├── ENHANCED_FEATURES.md               ← Technical details
└── QUICK_START.md                     ← This file
```

## Next Steps

After successful calibration:

1. **Document your offset values** - Note the calibrated X, Y, Z values for your device
2. **Test in your use case** - Verify accuracy for your specific application
3. **Consider hardcoding** - If you find consistent offset values, update `CameraTransformUtils.estimatedCameraOffset` default

## Support

For issues or questions:
- Check ENHANCED_FEATURES.md for technical details
- Review Xcode console logs
- Verify backend is running and accessible
- Test with well-lit, clearly visible ArUco markers

---

**Ready to use!** Start detecting ArUco markers and enjoy accurate 3D pose tracking in immersive space.
