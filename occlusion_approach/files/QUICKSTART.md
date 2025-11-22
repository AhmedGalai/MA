# Quick Start Guide

## Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install numpy matplotlib scipy --break-system-packages
```

### Step 2: Run the Program
Choose one of the following:

#### Option A: Extended Kalman Filter (Recommended)
```bash
python pose_estimation_ekf.py
```
- Best for most applications
- Fast and efficient
- Good for Gaussian noise

#### Option B: Particle Filter
```bash
python pose_estimation_pf.py
```
- Better for non-Gaussian noise
- Shows particle distribution
- More computational intensive

#### Option C: Compare Both Methods
```bash
python compare_methods.py
```
- Side-by-side comparison
- Performance metrics
- Recommended for evaluation

### Step 3: Interact with the Visualization
- **Slider**: Navigate through frames (0-99)
- **3D View**: Shows camera, object, occluder, and uncertainty
- **2D View**: Top-down projection with uncertainty ellipse
- **Info Panel**: Real-time statistics

## Understanding the Visualization

### Colors
- 🔵 **Blue sphere**: Ground truth object position
- 🟥 **Red box**: Occluder (blocks line of sight)
- 🟢 **Green triangle**: Camera position
- 🟠 **Orange star**: Estimated object position
- 🟠 **Orange cloud/ellipse**: Uncertainty region (3-sigma)

### What to Look For

1. **Normal Frames** (No Occlusion):
   - Small, tight uncertainty
   - Estimate very close to true position
   - Orange star overlaps blue sphere

2. **Occluded Frames**:
   - Larger uncertainty cloud
   - Estimate may drift
   - Red occluder blocks camera-to-object line

3. **Drift Over Time**:
   - Uncertainty grows gradually
   - Process noise increases
   - More visible in later frames

## Customization

### Change Occlusion Behavior
Edit line ~220 in the main scripts:

```python
failure_mode = 'none'  # Options: 'none', 'null', 'random', 'previous'
```

- `'none'`: Noisy observation during occlusion
- `'null'`: No observation
- `'random'`: Random incorrect pose
- `'previous'`: Repeat last valid pose

### Adjust Trajectory
Edit the parameters:

```python
n_frames = 100      # Total frames
radius = 3.0        # Camera orbit radius
height = 1.5        # Camera height
```

### Tune Filter Parameters

**EKF (Extended Kalman Filter):**
```python
# In __init__ method
self.Q = np.eye(13) * 0.01          # Process noise (drift)
self.R_normal = np.eye(7) * 0.01    # Measurement noise
```

**Particle Filter:**
```python
n_particles = 1000          # Number of particles (500-5000)
self.pos_noise = 0.05       # Position process noise
self.quat_noise = 0.02      # Orientation process noise
```

## Typical Results

### Extended Kalman Filter
- **Speed**: ~0.5 ms/frame
- **Memory**: ~100 KB
- **Mean Error**: 0.02-0.05 m
- **Peak Error**: 0.1-0.3 m (during occlusion)

### Particle Filter (1000 particles)
- **Speed**: ~10 ms/frame
- **Memory**: ~10 MB
- **Mean Error**: 0.02-0.06 m
- **Peak Error**: 0.1-0.4 m (during occlusion)

## Troubleshooting

### "No module named matplotlib"
```bash
pip install matplotlib --break-system-packages
```

### Visualization window doesn't appear
```bash
# Install tkinter
sudo apt-get install python3-tk

# Or use different backend
export MPLBACKEND=TkAgg
```

### Slow performance (Particle Filter)
Reduce particles:
```python
n_particles = 500  # Default is 1000
```

### High memory usage
Use EKF instead of Particle Filter, or reduce particles.

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Modify parameters** to see how they affect tracking
3. **Try different occlusion modes** to test robustness
4. **Run compare_methods.py** to evaluate both approaches

## Key Concepts

### State Vector (13D)
- Position: [x, y, z]
- Orientation: [qw, qx, qy, qz] quaternion
- Linear velocity: [vx, vy, vz]
- Angular velocity: [wx, wy, wz]

### Uncertainty
- **Small ellipse**: Confident estimate
- **Large ellipse**: Uncertain estimate
- **Growth**: Due to process noise and drift
- **Shrink**: When good measurements arrive

### Occlusion Effects
- Uncertainty increases 10×
- Estimate relies on motion model
- May drift without measurements
- Recovers when object visible again

## Performance Tips

1. **For real-time applications**: Use EKF
2. **For high accuracy**: Increase particles (PF)
3. **For multimodal uncertainty**: Use PF
4. **For computational efficiency**: Use EKF

## Support

For issues or questions:
1. Check the README.md
2. Review the code comments
3. Experiment with parameters
4. Compare both methods

## Summary

This implementation provides robust 6D pose estimation with:
- ✅ Bayesian uncertainty quantification
- ✅ Drift compensation
- ✅ Occlusion handling
- ✅ Interactive visualization
- ✅ Multiple failure modes
- ✅ Real-time capable (EKF)

Enjoy exploring probabilistic robotics!
