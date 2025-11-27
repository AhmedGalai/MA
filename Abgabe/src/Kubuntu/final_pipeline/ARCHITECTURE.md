# Final Pipeline Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     FINAL PIPELINE SYSTEM                        │
│                                                                  │
│  Input: AVP Mask + Headset Pose                                 │
│  Output: 6D Pose in AVP Coordinates                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

```
┌────────────────────┐
│  pipeline_api.py   │  Flask REST API Server
│  (280 lines)       │  - /process endpoint
│                    │  - /calibrate endpoint
│                    │  - /update_pose endpoint
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ pipeline_core.py   │  Main Integration Logic
│ (350 lines)        │  - Orchestrates pipeline flow
│                    │  - Error handling
│                    │  - Statistics tracking
└────┬───┬───┬───┬───┘
     │   │   │   │
     ▼   ▼   ▼   ▼
┌────┴───┴───┴───┴────────────────────────────────────────┐
│                  Core Components                         │
├──────────────────┬────────────────┬──────────────────────┤
│                  │                │                      │
│ realsense_depth  │  pose_manager  │ coordinate_transform │
│   (300 lines)    │  (350 lines)   │    (280 lines)       │
│                  │                │                      │
│ - Depth capture  │ - ArUco calib  │ - Kalman filter     │
│ - Intrinsics     │ - Pose stream  │ - Transformations   │
│ - Deprojection   │ - Pose history │ - Mask warping      │
│                  │                │                      │
└──────────────────┴────────────────┴──────────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │pose_estimator│
                  │  (270 lines) │
                  │              │
                  │ - 6D pose    │
                  │ - PCA/PnP    │
                  │ - Confidence │
                  └──────────────┘
```

## Data Flow Diagram

```
[AVP System]                [Final Pipeline]              [RealSense Camera]
     │                             │                             │
     │ 1. Mask (binary image)      │                             │
     ├────────────────────────────▶│                             │
     │                             │                             │
     │ 2. Headset Pose             │                             │
     ├────────────────────────────▶│                             │
     │    (position + rotation)    │                             │
     │                             │                             │
     │                             │ 3. Request depth frame      │
     │                             ├────────────────────────────▶│
     │                             │                             │
     │                             │ 4. RGB + Depth aligned      │
     │                             │◀────────────────────────────┤
     │                             │                             │
     │                             │ [PROCESSING]                │
     │                             │                             │
     │                             │ A. Update Kalman filter     │
     │                             │ B. Transform mask           │
     │                             │ C. Estimate pose            │
     │                             │ D. Transform to AVP         │
     │                             │                             │
     │ 5. Final 6D Pose            │                             │
     │◀────────────────────────────┤                             │
     │    (AVP coordinates)        │                             │
     │    + Confidence             │                             │
     │                             │                             │
```

## Coordinate Frame Transformations

```
        [World Frame]
      (ArUco Board Origin)
              │
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼
   [Headset]   [RealSense]
   T_w_h       T_w_rs
   (One-time   (One-time
    ArUco       ArUco
    calib)      calib)
        │           │
        │           │
        │     Derive: T_rs_h = T_w_rs^-1 * T_w_h
        │           │
        └─────┬─────┘
              │
              ▼
        [AVP ↔ RealSense]
         Transformation
              │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
[Mask_AVP]          [Pose_RS]
    │                   │
    │ Transform         │ Estimate
    │ using T_rs_h      │ using depth+mask
    │                   │
    ▼                   ▼
[Mask_RS]           [Object_RS]
    │                   │
    └─────────┬─────────┘
              │
              │ Transform back
              │ using T_h_rs = T_rs_h^-1
              │
              ▼
         [Object_AVP]
         Final Result
```

## Processing Pipeline Flow

```
┌──────────────────────────────────────────────────────────┐
│ Step 1: RealSense Depth Capture                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  realsense_depth.capture_frame()                         │
│  ├─ Wait for frames (timeout: 1000ms)                   │
│  ├─ Align depth to color                                │
│  ├─ Convert to numpy arrays                             │
│  └─ Return {rgb, depth, intrinsics, timestamp}          │
│                                                          │
│  Output: depth_map (480×640 uint16 millimeters)         │
│         intrinsics_K (3×3 camera matrix)                │
│                                                          │
│  Time: ~33ms @ 30fps                                     │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: Headset Pose Update (if provided)               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  pose_manager.update_headset_pose(pose)                  │
│  ├─ Add to pose history buffer (max 30)                 │
│  └─ Update last_headset_update timestamp                │
│                                                          │
│  transformer.update_pose_with_correction(pose, dt)       │
│  ├─ Kalman predict step (dt seconds)                    │
│  ├─ Convert rotation to quaternion                      │
│  ├─ Kalman update step (measurement)                    │
│  └─ Return corrected pose                               │
│                                                          │
│  Output: corrected_pose {position, rotation, quaternion}│
│                                                          │
│  Time: <1ms                                              │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Step 3: Transform Mask (AVP → RealSense)                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  transformer.transform_mask_avp_to_realsense()           │
│  ├─ Get calibrated T_rs_avp transformation              │
│  ├─ Extract rotation R and translation t                │
│  ├─ Compute homography H = K_rs * R * K_avp^-1          │
│  └─ Warp mask using cv.warpPerspective()                │
│                                                          │
│  Input:  mask_avp (H_avp × W_avp binary)                │
│  Output: mask_rs (480 × 640 binary)                     │
│                                                          │
│  Time: ~5ms                                              │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Step 4: Estimate 6D Pose (in RealSense view)            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  pose_estimator.estimate_pose_from_depth_and_mask()      │
│  ├─ Extract 3D points from depth using mask             │
│  │  └─ Deproject pixels: (u,v,d) → (x,y,z)             │
│  ├─ Compute centroid (position)                         │
│  ├─ PCA for orientation (rotation matrix)               │
│  │  └─ Eigendecomposition of covariance                │
│  ├─ Convert rotation matrix to rvec                     │
│  └─ Compute confidence score                            │
│                                                          │
│  Output: {rvec, tvec, confidence, num_points}           │
│                                                          │
│  Time: ~8ms                                              │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Step 5: Transform Pose (RealSense → AVP)                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  _transform_pose_rs_to_avp()                             │
│  ├─ Get T_rs_avp (calibrated transform)                 │
│  ├─ Invert: T_avp_rs = T_rs_avp^-1                      │
│  ├─ Construct T_rs_obj from rvec, tvec                  │
│  ├─ Transform: T_avp_obj = T_avp_rs * T_rs_obj          │
│  └─ Extract final rvec_avp, tvec_avp                    │
│                                                          │
│  Output: pose_avp {rvec, tvec, confidence}              │
│                                                          │
│  Time: ~1ms                                              │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ FINAL RESULT  │
                 │               │
                 │ Pose in AVP   │
                 │ coordinates   │
                 │ + Confidence  │
                 └───────────────┘
```

## Kalman Filter State

```
State Vector (13D):
┌───────────────────────────────────────┐
│ Position:          [x, y, z]         │  (3)
│ Quaternion:        [qw, qx, qy, qz]  │  (4)
│ Linear Velocity:   [vx, vy, vz]      │  (3)
│ Angular Velocity:  [wx, wy, wz]      │  (3)
└───────────────────────────────────────┘

Prediction:
  x_pred = F * x_curr
  P_pred = F * P * F^T + Q

Update:
  K = P_pred * H^T * (H * P_pred * H^T + R)^-1
  x_new = x_pred + K * (z - H * x_pred)
  P_new = (I - K * H) * P_pred

Where:
  F = State transition (constant velocity model)
  Q = Process noise covariance
  H = Measurement matrix (observe pos + quat only)
  R = Measurement noise covariance
  K = Kalman gain
```

## Calibration Process

```
┌────────────────────────────────────────────────┐
│              ArUco Calibration                 │
├────────────────────────────────────────────────┤
│                                                │
│  Setup:                                        │
│  ┌──────────────┐         ┌──────────────┐   │
│  │   Headset    │         │  RealSense   │   │
│  │  (AVP view)  │         │  (RS view)   │   │
│  │              │         │              │   │
│  │  Sees:       │         │  Sees:       │   │
│  │  ArUco ID 0  │         │  ArUco Board │   │
│  │  (on head)   │         │  (3×4)       │   │
│  └──────────────┘         └──────────────┘   │
│                                                │
│  Step 1: Detect headset marker in AVP view    │
│          → T_world_headset                     │
│                                                │
│  Step 2: Detect board in RealSense view       │
│          → T_realsense_world                   │
│                                                │
│  Step 3: Derive transformation                 │
│          T_realsense_avp =                     │
│            T_realsense_world * T_world_headset │
│                                                │
│  Save:                                         │
│  ✓ calibration/headset_to_world.json         │
│  ✓ calibration/realsense_to_world.json       │
│  ✓ calibration/avp_to_realsense.json         │
│                                                │
└────────────────────────────────────────────────┘
```

## Error Handling Flow

```
process_frame()
    │
    ├─ No mask provided?
    │  └─> Return {"success": False, "error": "No mask"}
    │
    ├─ RealSense unavailable?
    │  └─> Return {"success": False, "error": "RealSense not available"}
    │
    ├─ Not calibrated?
    │  └─> Return {"success": False, "error": "Not calibrated"}
    │
    ├─ RealSense capture fails?
    │  └─> Return {"success": False, "error": "Depth capture failed"}
    │
    ├─ Mask transformation fails?
    │  └─> Return {"success": False, "error": "Mask transform failed"}
    │
    ├─ Pose estimation fails?
    │  └─> Return {"success": False, "error": "Pose estimation failed"}
    │      └─> stats["failed_poses"]++
    │
    └─ Success!
       └─> Return {"success": True, "pose_avp_view": {...}}
           └─> stats["successful_poses"]++
```

## Module Dependencies

```
pipeline_api.py
    └─> pipeline_core.py
            ├─> realsense_depth.py
            │       └─> pyrealsense2
            │       └─> numpy, cv2
            │
            ├─> pose_manager.py
            │       └─> numpy, cv2
            │       └─> collections.deque
            │       └─> config.py
            │
            ├─> coordinate_transformer.py
            │       └─> scipy.spatial.transform
            │       └─> numpy, cv2
            │       └─> pose_manager.py
            │       └─> config.py
            │
            └─> pose_estimator.py
                    └─> numpy, cv2
                    └─> config.py
```

## Performance Breakdown

```
Component                    Time (ms)   % of Total
──────────────────────────────────────────────────
RealSense depth capture      33          66%
Headset pose update          <1          <1%
  └─ Kalman predict          <1
  └─ Kalman update           <1
Mask transformation          5           10%
  └─ Homography compute      <1
  └─ warpPerspective         5
Pose estimation              8           16%
  └─ 3D point extraction     3
  └─ PCA computation         4
  └─ Confidence scoring      1
Pose transformation          1           2%
Overhead/bookkeeping         3           6%
──────────────────────────────────────────────────
TOTAL                        ~50         100%
```

## Thread Safety

All components are designed for single-threaded operation. For multi-threaded use:

```python
# Recommended pattern for concurrent requests
import threading

pipeline_lock = threading.Lock()

def process_request(mask, pose):
    with pipeline_lock:
        result = pipeline.process_frame(
            avp_mask=mask,
            headset_pose=pose
        )
    return result
```

## Memory Usage

```
Component              Memory
─────────────────────────────────
RealSense pipeline     ~50 MB
Depth frame (uint16)   ~0.6 MB (640×480×2)
RGB frame (uint8)      ~0.9 MB (640×480×3)
Kalman filter state    ~2 KB
Pose history (30)      ~5 KB
Calibration data       ~1 KB
─────────────────────────────────
TOTAL                  ~52 MB
```

## Configuration Files

```
config.py
  ├─ REALSENSE_CONFIG
  │    └─ width, height, fps, formats
  │
  ├─ ARUCO_CONFIG
  │    └─ dictionary, marker sizes, board layout
  │
  ├─ KALMAN_CONFIG
  │    └─ process/measurement noise, uncertainty
  │
  ├─ POSE_ESTIMATION_CONFIG
  │    └─ min_points, RANSAC params, threshold
  │
  ├─ CALIBRATION_FILES
  │    └─ Paths to saved calibrations
  │
  └─ API_CONFIG
       └─ host, port, debug flag
```

This architecture provides a clean, modular, and maintainable pipeline for real-time 6D pose estimation with probabilistic correction.
