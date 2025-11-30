# Debugging Window Guide

## Overview

The `tk_debugging.py` provides a streamlined debugging interface for the Final Pipeline, removing unnecessary options from the original debugging window and focusing only on what's needed for RealSense-based depth estimation.

## Changes from Original `tk_debugging_unified.py`

### **Removed** ❌
- ❌ Transformers depth mode toggle
- ❌ Dual-mode switching (RealSense/Transformers)
- ❌ HSV color picker and tolerance sliders (mask comes from AVP)
- ❌ "Apply ROI Settings" button
- ❌ "Reverse RealSense Mode" toggle
- ❌ "Random Pose" toggle
- ❌ ArUco detection feed (handled by pose manager)
- ❌ Clean ROI mask feed (not needed)
- ❌ Disparity feed (focusing on depth only)

### **Kept** ✅
- ✅ RealSense RGB feed
- ✅ RealSense Depth feed (metric)
- ✅ Connection controls (host/port)
- ✅ UI refresh rate control
- ✅ **Save Next Frame** (fully working)
- ✅ Pause/Resume updates
- ✅ Pipeline statistics display

### **Added** 🆕
- 🆕 AVP Mask display
- 🆕 6D Pose Overlay display
- 🆕 **Test Pose API** button (sends test request)
- 🆕 Success rate calculation
- 🆕 Calibration status indicator
- 🆕 Direct RealSense camera access for visualization

## Running the Debugging Window

### Prerequisites

1. **Start Final Pipeline API**
   ```bash
   cd final_pipeline
   python -m pipeline_api
   # Server runs on http://localhost:5001
   ```

2. **Connect RealSense Camera**
   - Intel RealSense D435/D455 via USB 3.0
   - Verify: `rs-enumerate-devices`

3. **Calibrate Pipeline** (if not already done)
   ```python
   from final_pipeline import FinalPipeline
   pipeline = FinalPipeline()
   pipeline.calibrate_with_aruco(headset_image, K, dist)
   ```

### Start Debugging Window

```bash
cd final_pipeline
python tk_debugging.py
```

## Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  API Connection: [Host] [Port] [Connect] ● Status          │
│  UI Refresh: [1-60 Hz] [Pause] [Refresh Now]               │
│  [Save Next Frame] [Test Pose API]                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┬──────────────┐                           │
│  │ RealSense RGB│ RealSense    │                           │
│  │              │ Depth        │                           │
│  └──────────────┴──────────────┘                           │
│  ┌──────────────┬──────────────┐                           │
│  │ AVP Mask     │ 6D Pose      │                           │
│  │              │ Overlay      │                           │
│  └──────────────┴──────────────┘                           │
├─────────────────────────────────────────────────────────────┤
│  Pipeline Info:                                             │
│  • Calibrated: True/False                                  │
│  • Frames Processed: 123                                   │
│  • Successful Poses: 120                                   │
│  • Success Rate: 97%                                       │
│  • Avg Processing Time: 45.2 ms                            │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 1. API Connection

**Connect to Final Pipeline API:**
- Default: `localhost:5001`
- Status indicator:
  - 🟢 Green = Connected
  - 🔴 Red = Disconnected
  - ⚪ Gray = Unknown

**Change Connection:**
```
Host: [localhost]  Port: [5001]  [Connect]
```

### 2. Video Feeds

#### **RealSense RGB**
- Live RGB feed from RealSense camera
- Resolution: 640×480 @ 30fps
- Color-aligned depth frame

#### **RealSense Depth**
- Metric depth in millimeters
- Colored using JET colormap
- Red = close, Blue = far

#### **AVP Mask** (Coming from AVP)
- Binary mask from Apple Vision Pro
- Shows object region of interest
- Used for pose estimation

#### **6D Pose Overlay**
- RGB + 3D axes overlay
- Red axis = X
- Green axis = Y
- Blue axis = Z
- White dot = origin
- Confidence score displayed

### 3. Save Next Frame

**How to Use:**
1. Click **"Save Next Frame"** button
2. Confirmation popup appears
3. Next frame is automatically saved to `saved_frames/`

**Saved Files:**
```
saved_frames/
├── rgb_20250116_230145.png           # RGB image
├── depth_20250116_230145.npy         # Depth array (uint16, mm)
├── depth_viz_20250116_230145.png     # Depth visualization
└── intrinsics_20250116_230145.json   # Camera intrinsics
```

**Loading Saved Depth:**
```python
import numpy as np

# Load depth array
depth = np.load('saved_frames/depth_20250116_230145.npy')
print(f"Depth shape: {depth.shape}")
print(f"Depth range: {depth.min()} - {depth.max()} mm")

# Convert to meters
depth_meters = depth / 1000.0
```

### 4. Test Pose API

**What it does:**
- Creates test mask (100×100 region)
- Sends test pose request to API
- Shows result in popup

**How to Use:**
1. Click **"Test Pose API"** button
2. Pipeline processes test data
3. Result popup shows:
   - Success/Failure
   - Confidence score
   - Processing time
   - Error message (if failed)

**Example Result:**
```
Success!
Confidence: 0.923
Processing time: 42.3ms
```

### 5. Pipeline Statistics

**Real-Time Stats:**
- **Calibrated**: Pipeline calibration status
- **RealSense Available**: Camera connection status
- **Frames Processed**: Total frames
- **Successful Poses**: Valid pose estimates
- **Failed Poses**: Failed estimations
- **Avg Processing Time**: Mean latency
- **Success Rate**: Percentage of successful poses

**Auto-Updates:** Every frame (1-60 Hz configurable)

### 6. UI Controls

**Refresh Rate Slider:**
- Range: 1-60 Hz
- Controls UI update frequency
- Lower = less CPU, higher = smoother

**Pause/Resume:**
- Pause auto-updates
- Resume updates
- Manual refresh still works

**Refresh Now:**
- Force immediate update
- Works even when paused

## Workflow

### Standard Operation

1. **Start API Server**
   ```bash
   python -m pipeline_api
   ```

2. **Start Debugging Window**
   ```bash
   python tk_debugging.py
   ```

3. **Verify Connection**
   - Check status indicator (should be green)
   - Stats should update automatically

4. **Monitor Feeds**
   - RGB: Live video from RealSense
   - Depth: Metric depth visualization
   - Mask: Object region (when sent from AVP)
   - Pose: 6D orientation (when available)

5. **Test API**
   - Click "Test Pose API"
   - Verify successful response

6. **Save Data**
   - Click "Save Next Frame"
   - Check `saved_frames/` directory

### Debugging Issues

**No RGB/Depth Feed:**
- Check RealSense connection
- Run: `rs-enumerate-devices`
- Verify USB 3.0 connection

**API Not Connected:**
- Check API server is running
- Verify host/port settings
- Check firewall settings

**Low Success Rate:**
- Check calibration status
- Verify mask quality
- Ensure object in RealSense range (0.3-3m)

**High Processing Time:**
- Normal range: 40-60ms
- If >100ms, check system load
- Ensure RealSense drivers updated

## Keyboard Shortcuts

- **Space**: Pause/Resume updates
- **S**: Save next frame
- **T**: Test pose API
- **R**: Refresh now
- **Q**: Quit application

## Troubleshooting

### RealSense Access Error

**Problem**: "RealSense direct access failed"

**Solution**:
```bash
# Check RealSense
rs-enumerate-devices

# Reinstall
pip uninstall pyrealsense2
pip install pyrealsense2

# Restart debugging window
```

### API Connection Failed

**Problem**: Red status indicator

**Solution**:
1. Check API server is running:
   ```bash
   curl http://localhost:5001/health
   ```

2. Restart API server:
   ```bash
   python -m pipeline_api
   ```

3. Update connection in debugging window

### Save Frame Not Working

**Problem**: No files in `saved_frames/`

**Solution**:
1. Check write permissions
2. Verify RealSense is capturing
3. Check console for errors
4. Try saving to different directory

### Pose Overlay Not Showing

**Problem**: No axes on RGB feed

**Solution**:
1. Ensure pipeline is calibrated
2. Verify pose estimation succeeds
3. Check Test Pose API response
4. Review intrinsics accuracy

## Advanced Usage

### Custom Test Mask

Edit `test_pose_api()` in `tk_debugging.py`:

```python
def test_pose_api(self):
    # Create custom mask
    test_mask = cv.imread('your_mask.png', cv.IMREAD_GRAYSCALE)

    # Encode and send
    mask_base64 = encode_image_to_base64(test_mask)
    result = self.client.process_frame(mask_base64, headset_pose=test_pose)
```

### Monitor Specific Pose

```python
def fetch_and_display_data(self):
    # ... existing code ...

    # Monitor specific pose
    if self.last_stats:
        latest_pose = self.last_stats.get('last_pose')
        if latest_pose:
            print(f"Position: {latest_pose.get('tvec')}")
            print(f"Rotation: {latest_pose.get('rvec')}")
```

### Export Statistics

```python
def save_stats_to_file(self):
    if self.last_stats:
        with open('stats_export.json', 'w') as f:
            json.dump(self.last_stats, f, indent=2)
```

## Comparison with Original Debugging Window

| Feature | Original (tk_debugging_unified.py) | New (tk_debugging.py) |
|---------|-----------------------------------|---------------------|
| **Lines of Code** | ~1100 | ~500 |
| **Depth Modes** | Transformers + RealSense | RealSense only |
| **Image Feeds** | 6 panels | 4 panels |
| **HSV Controls** | Yes (color picker + sliders) | No (mask from AVP) |
| **API Endpoints** | /config, /stats, /data_batch | /stats, /process, /health |
| **Complexity** | High (many options) | Low (focused) |
| **Save Frame** | Partial | **Fully Working** ✅ |
| **Pose API Test** | No | **Yes** ✅ |
| **Success Rate** | No | **Yes** ✅ |

## Summary

The new debugging window is:
- ✅ **Simpler** - Removed unnecessary options
- ✅ **Focused** - RealSense-only approach
- ✅ **Functional** - Save frame fully working
- ✅ **Integrated** - Direct pose API testing
- ✅ **Informative** - Better statistics display

Perfect for debugging the Final Pipeline!
