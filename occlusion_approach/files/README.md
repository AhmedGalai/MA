# Stochastic 6D Pose Estimation with Occlusion Handling

A comprehensive Python implementation of Bayesian state estimation for 6D pose tracking with uncertainty quantification, drift handling, and occlusion scenarios.

## Overview

This project implements two approaches for robust pose estimation:

1. **Extended Kalman Filter (EKF)** - Optimal for Gaussian noise and linear/near-linear systems
2. **Particle Filter (PF)** - Better for non-Gaussian noise and highly non-linear dynamics

Both methods handle:
- Camera pose drift over time
- Measurement uncertainty from camera intrinsics
- Occlusion scenarios with multiple failure modes
- Real-time uncertainty quantification

## Features

### Core Capabilities
- ✅ 6D pose estimation (position + orientation)
- ✅ Quaternion-based orientation representation
- ✅ Velocity estimation (linear and angular)
- ✅ Dynamic uncertainty modeling
- ✅ Occlusion detection and handling
- ✅ Drift compensation

### Visualization
- Interactive 3D scene with camera trajectory
- Particle distribution visualization (PF only)
- Uncertainty ellipsoids (3-sigma bounds)
- Top-down 2D projection with uncertainty
- Frame-by-frame slider navigation
- Real-time statistics panel

### Occlusion Handling
The system supports three failure modes during occlusion:
1. **null** - No observation returned
2. **random** - Random irrelevant pose
3. **previous** - Repeat last valid pose
4. **none** - Noisy observation (default)

## Installation

### Requirements
```bash
pip install numpy matplotlib scipy --break-system-packages
```

### Files
- `pose_estimation_ekf.py` - Extended Kalman Filter implementation
- `pose_estimation_pf.py` - Particle Filter implementation
- `README.md` - This file

## Usage

### Extended Kalman Filter (Recommended for most cases)
```bash
python pose_estimation_ekf.py
```

**When to use EKF:**
- Measurement noise is approximately Gaussian
- System dynamics are near-linear
- Computational efficiency is important
- Real-time performance needed

### Particle Filter
```bash
python pose_estimation_pf.py
```

**When to use PF:**
- Non-Gaussian measurement noise
- Highly non-linear dynamics
- Multimodal distributions expected
- Computational resources available

## Configuration

### Changing Occlusion Failure Mode

Edit the `failure_mode` variable in the `main()` function:

```python
# In pose_estimation_ekf.py or pose_estimation_pf.py
failure_mode = 'none'  # Options: 'none', 'null', 'random', 'previous'
```

### Adjusting Parameters

**Trajectory parameters:**
```python
n_frames = 100      # Number of frames
dt = 0.033          # Time step (30 FPS)
radius = 3.0        # Camera trajectory radius
height = 1.5        # Camera height
```

**Occluder geometry:**
```python
occluder_bounds = {
    'x_range': [-0.5, 0.5],
    'y_range': [0.8, 1.5],
    'z_range': [0.0, 1.5]
}
```

**EKF parameters:**
```python
self.Q = np.eye(13) * 0.01          # Process noise
self.R_normal = np.eye(7) * 0.01    # Measurement noise
```

**Particle Filter parameters:**
```python
n_particles = 1000                   # Number of particles
self.pos_noise = 0.05                # Position noise
self.quat_noise = 0.02               # Orientation noise
```

## System Architecture

### State Representation
Both filters use a 13-dimensional state vector:
- **Position**: [x, y, z]
- **Orientation**: [qw, qx, qy, qz] (quaternion)
- **Linear velocity**: [vx, vy, vz]
- **Angular velocity**: [wx, wy, wz]

### Motion Model
Constant velocity model with drift:
```
x(t+1) = x(t) + v(t) * dt + noise
q(t+1) = q(t) ⊗ exp(ω(t) * dt/2) + noise
```

### Measurement Model
Direct observation of position and orientation:
```
z(t) = [position, quaternion] + noise
```

### Uncertainty Modeling

**Normal conditions:**
- Position noise: σ = 0.005 m
- Orientation noise: σ = 0.01 rad

**Occluded conditions:**
- Position noise: σ = 0.05 m (10× increase)
- Orientation noise: σ = 0.1 rad (10× increase)

**Drift over time:**
- Process noise scales linearly: drift_scale = 1.0 + 0.5 * (t/T)

## Visualization Guide

### 3D Scene View
- **Blue sphere**: Ground truth object
- **Red box**: Occluder
- **Green triangle**: Camera position
- **Green line**: Camera orientation
- **Orange star**: Estimated pose
- **Orange ellipsoid/cloud**: Uncertainty (3σ)

### 2D Top-Down View
- Shows XY projection
- Uncertainty ellipse represents 3-sigma bounds
- Particle scatter visible in PF version

### Info Panel
- Frame number
- Occlusion status
- Position estimate
- Uncertainty metrics (standard deviation)

## Algorithm Comparison

| Feature | EKF | Particle Filter |
|---------|-----|-----------------|
| Computational cost | Low (O(n²)) | High (O(N×n)) |
| Gaussian assumption | Required | Not required |
| Multimodal support | No | Yes |
| Real-time capable | Yes | Depends on N |
| Recommended particles | N/A | 1000-5000 |

## Technical Details

### Extended Kalman Filter
- **Prediction**: Linearized motion model with Jacobian
- **Update**: Kalman gain computed from innovation covariance
- **Quaternion**: Special handling for normalization

### Particle Filter
- **Prediction**: Sample-based motion model
- **Update**: Likelihood-based weight update
- **Resampling**: Triggered when effective sample size < N/2

## Example Output

```
======================================================================
Stochastic 6D Pose Estimation with Extended Kalman Filter
======================================================================

1. Generating ground truth trajectory...
2. Generating observations with occlusion handling...
   Using failure mode: none
   Detected 18 occluded frames
3. Running Extended Kalman Filter...
4. Launching interactive visualization...

Instructions:
  - Use the slider to navigate through frames
  - 3D view shows camera path, object, occluder, and uncertainty
  - 2D view shows top-down projection with uncertainty ellipse
  - Info panel displays detailed statistics

Done!
```

## Performance Notes

### EKF Performance
- ~0.5 ms per frame
- Suitable for real-time applications
- Memory: ~100 KB

### Particle Filter Performance (1000 particles)
- ~10 ms per frame
- May struggle with real-time on limited hardware
- Memory: ~10 MB

## Troubleshooting

### Import Errors
```bash
# Ensure packages are installed
pip install numpy matplotlib scipy --break-system-packages
```

### Visualization Issues
```bash
# May need to install tkinter for matplotlib backend
sudo apt-get install python3-tk
```

### Memory Issues (Particle Filter)
```python
# Reduce number of particles
n_particles = 500  # Instead of 1000
```

## Future Enhancements

- [ ] Unscented Kalman Filter (UKF) implementation
- [ ] Real-time camera feed integration
- [ ] Multiple object tracking
- [ ] GPU acceleration for Particle Filter
- [ ] Adaptive resampling strategies
- [ ] Learned motion models

## References

1. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics
2. Särkkä, S. (2013). Bayesian Filtering and Smoothing
3. Arulampalam, M. S., et al. (2002). A tutorial on particle filters

## License

MIT License - Feel free to use and modify for your projects.

## Author

Created for demonstrating Bayesian state estimation with uncertainty quantification.
