"""
Stochastic Pose Estimation with Extended Kalman Filter
Handles pose uncertainty, drift, and occlusion scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Set random seed for reproducibility
np.random.seed(42)

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for 6D pose estimation
    State: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    - Position (x, y, z)
    - Orientation as quaternion (qw, qx, qy, qz)
    - Linear velocity (vx, vy, vz)
    - Angular velocity (wx, wy, wz)
    """
    
    def __init__(self, dt=0.033):
        self.dt = dt
        self.state_dim = 13  # 3 pos + 4 quat + 3 lin_vel + 3 ang_vel
        
        # Initial state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        self.x = np.zeros(self.state_dim)
        self.x[3] = 1.0  # Initialize quaternion to identity
        
        # Initial covariance
        self.P = np.eye(self.state_dim) * 0.1
        self.P[3:7, 3:7] = np.eye(4) * 0.01  # Lower uncertainty for orientation
        
        # Process noise (motion model uncertainty + drift)
        self.Q = np.eye(self.state_dim) * 0.01
        self.Q[0:3, 0:3] *= 0.05  # Position drift
        self.Q[3:7, 3:7] *= 0.02  # Orientation drift
        self.Q[7:10, 7:10] *= 0.1  # Velocity uncertainty
        self.Q[10:13, 10:13] *= 0.1  # Angular velocity uncertainty
        
        # Measurement noise (observation uncertainty)
        self.R_normal = np.eye(7) * 0.01  # Normal observation
        self.R_normal[0:3, 0:3] *= 0.005  # Position measurement noise
        self.R_normal[3:7, 3:7] *= 0.01   # Orientation measurement noise
        
    def normalize_quaternion(self):
        """Normalize quaternion part of state"""
        q = self.x[3:7]
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        self.x[3:7] = q_norm
        
    def predict(self, drift_scale=1.0):
        """Prediction step with motion model"""
        # Extract state components
        pos = self.x[0:3]
        q = self.x[3:7]
        lin_vel = self.x[7:10]
        ang_vel = self.x[10:13]
        
        # Predict position using velocity
        pos_new = pos + lin_vel * self.dt
        
        # Predict orientation using angular velocity
        # Convert angular velocity to quaternion derivative
        ang_vel_norm = np.linalg.norm(ang_vel)
        if ang_vel_norm > 1e-6:
            axis = ang_vel / ang_vel_norm
            angle = ang_vel_norm * self.dt
            delta_q = np.array([
                np.cos(angle/2),
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2)
            ])
            # Quaternion multiplication
            q_new = self.quaternion_multiply(q, delta_q)
        else:
            q_new = q
            
        # Update state
        self.x[0:3] = pos_new
        self.x[3:7] = q_new
        self.normalize_quaternion()
        
        # Jacobian of motion model (simplified)
        F = np.eye(self.state_dim)
        F[0:3, 7:10] = np.eye(3) * self.dt
        
        # Update covariance with drift
        Q_drift = self.Q * drift_scale
        self.P = F @ self.P @ F.T + Q_drift
        
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        
    def update(self, measurement, occlusion_factor=1.0):
        """
        Update step with measurement
        measurement: [x, y, z, qw, qx, qy, qz]
        occlusion_factor: 1.0 (no occlusion) to 10+ (heavy occlusion)
        """
        if measurement is None:
            return
            
        # Measurement function H (observe position and orientation)
        H = np.zeros((7, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # Observe position
        H[3:7, 3:7] = np.eye(4)  # Observe orientation
        
        # Measurement noise increases with occlusion
        R = self.R_normal * occlusion_factor
        
        # Innovation
        z = measurement
        z_pred = H @ self.x
        y = z - z_pred
        
        # Handle quaternion wrap-around (ensure shortest path)
        if np.dot(z[3:7], z_pred[3:7]) < 0:
            y[3:7] = z + z_pred - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        self.normalize_quaternion()
        
        # Update covariance
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
    def get_pose(self):
        """Get current pose estimate"""
        return self.x[0:3], self.x[3:7]
        
    def get_covariance(self):
        """Get position and orientation covariance"""
        pos_cov = self.P[0:3, 0:3]
        quat_cov = self.P[3:7, 3:7]
        return pos_cov, quat_cov

def generate_ground_truth_trajectory(n_frames=100):
    """Generate a circular trajectory around an object"""
    t = np.linspace(0, 2*np.pi, n_frames)
    radius = 3.0
    height = 1.5
    
    # Camera positions (circular path)
    camera_positions = np.column_stack([
        radius * np.cos(t),
        radius * np.sin(t),
        np.ones(n_frames) * height
    ])
    
    # Camera orientations (looking at origin)
    camera_orientations = []
    for i in range(n_frames):
        # Look-at matrix
        forward = -camera_positions[i] / np.linalg.norm(camera_positions[i])
        right = np.cross([0, 0, 1], forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        rot_matrix = np.column_stack([right, up, forward])
        quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]
        camera_orientations.append(quat_wxyz)
    
    return camera_positions, np.array(camera_orientations)

def check_occlusion(camera_pos, object_pos, occluder_bounds):
    """
    Check if object is occluded by a rectangle
    occluder_bounds: dict with 'x_range', 'y_range', 'z_range'
    """
    # Ray from camera to object
    ray_dir = object_pos - camera_pos
    t_vals = []
    
    # Check intersection with each face of the occluder box
    for axis in range(3):
        for bound in ['min', 'max']:
            if axis == 0:  # X
                plane_val = occluder_bounds['x_range'][0 if bound == 'min' else 1]
            elif axis == 1:  # Y
                plane_val = occluder_bounds['y_range'][0 if bound == 'min' else 1]
            else:  # Z
                plane_val = occluder_bounds['z_range'][0 if bound == 'min' else 1]
            
            if abs(ray_dir[axis]) > 1e-6:
                t = (plane_val - camera_pos[axis]) / ray_dir[axis]
                if 0 < t < 1:  # Intersection between camera and object
                    intersect = camera_pos + t * ray_dir
                    # Check if intersection is within box bounds
                    if (occluder_bounds['x_range'][0] <= intersect[0] <= occluder_bounds['x_range'][1] and
                        occluder_bounds['y_range'][0] <= intersect[1] <= occluder_bounds['y_range'][1] and
                        occluder_bounds['z_range'][0] <= intersect[2] <= occluder_bounds['z_range'][1]):
                        return True
    return False

def generate_observations(camera_positions, camera_orientations, object_pos, 
                         occluder_bounds, failure_mode='none'):
    """
    Generate observations with occlusion handling
    failure_mode: 'none', 'null', 'random', 'previous'
    """
    n_frames = len(camera_positions)
    observations = []
    occlusion_factors = []
    
    prev_obs = None
    
    for i in range(n_frames):
        is_occluded = check_occlusion(camera_positions[i], object_pos, occluder_bounds)
        
        if is_occluded:
            occlusion_factor = 10.0  # High uncertainty
            
            if failure_mode == 'null':
                obs = None
            elif failure_mode == 'random':
                # Random irrelevant pose
                obs = np.concatenate([
                    np.random.randn(3) * 2.0,
                    R.random().as_quat()[[3, 0, 1, 2]]  # Convert to [w, x, y, z]
                ])
            elif failure_mode == 'previous':
                obs = prev_obs if prev_obs is not None else None
            else:  # 'none' - still provide noisy observation
                obs = np.concatenate([
                    object_pos + np.random.randn(3) * 0.3,  # Heavy noise
                    camera_orientations[i] + np.random.randn(4) * 0.2
                ])
                obs[3:7] = obs[3:7] / np.linalg.norm(obs[3:7])  # Normalize quaternion
        else:
            occlusion_factor = 1.0  # Normal uncertainty
            # Add small measurement noise
            obs = np.concatenate([
                object_pos + np.random.randn(3) * 0.02,
                camera_orientations[i] + np.random.randn(4) * 0.01
            ])
            obs[3:7] = obs[3:7] / np.linalg.norm(obs[3:7])  # Normalize quaternion
            prev_obs = obs
        
        observations.append(obs)
        occlusion_factors.append(occlusion_factor)
    
    return observations, occlusion_factors

def visualize_results(camera_positions, observations, estimated_poses, 
                     estimated_covs, object_pos, occluder_bounds, occlusion_factors):
    """Interactive visualization with slider"""
    n_frames = len(camera_positions)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 3D view
    ax_3d = fig.add_subplot(121, projection='3d')
    
    # 2D uncertainty view
    ax_2d = fig.add_subplot(222)
    ax_info = fig.add_subplot(224)
    
    plt.subplots_adjust(bottom=0.15, hspace=0.3)
    
    # Slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, n_frames-1, valinit=0, valstep=1)
    
    def update(frame):
        frame = int(frame)
        
        # Clear axes
        ax_3d.clear()
        ax_2d.clear()
        ax_info.clear()
        
        # --- 3D View ---
        # Draw ground truth object (sphere at origin)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_sphere = 0.2 * np.cos(u) * np.sin(v) + object_pos[0]
        y_sphere = 0.2 * np.sin(u) * np.sin(v) + object_pos[1]
        z_sphere = 0.2 * np.cos(v) + object_pos[2]
        ax_3d.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.6, color='blue', label='Object')
        
        # Draw occluder (rectangle)
        x_range = occluder_bounds['x_range']
        y_range = occluder_bounds['y_range']
        z_range = occluder_bounds['z_range']
        
        # Draw box faces
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        vertices = [
            [x_range[0], y_range[0], z_range[0]],
            [x_range[1], y_range[0], z_range[0]],
            [x_range[1], y_range[1], z_range[0]],
            [x_range[0], y_range[1], z_range[0]],
            [x_range[0], y_range[0], z_range[1]],
            [x_range[1], y_range[0], z_range[1]],
            [x_range[1], y_range[1], z_range[1]],
            [x_range[0], y_range[1], z_range[1]]
        ]
        
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[7], vertices[6], vertices[2], vertices[3]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]
        
        ax_3d.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor='red', 
                                                edgecolor='darkred', linewidths=1))
        
        # Draw camera trajectory
        ax_3d.plot(camera_positions[:frame+1, 0], 
                  camera_positions[:frame+1, 1], 
                  camera_positions[:frame+1, 2], 
                  'g-', alpha=0.3, linewidth=1)
        
        # Current camera position
        cam_pos = camera_positions[frame]
        ax_3d.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]], 
                     c='green', s=100, marker='^', label='Camera')
        
        # Draw camera orientation
        cam_quat = estimated_poses[frame][1]
        rot = R.from_quat([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]])
        forward = rot.apply([0, 0, 0.5])
        ax_3d.plot([cam_pos[0], cam_pos[0] + forward[0]], 
                  [cam_pos[1], cam_pos[1] + forward[1]], 
                  [cam_pos[2], cam_pos[2] + forward[2]], 'g-', linewidth=2)
        
        # Estimated object pose
        est_pos, est_quat = estimated_poses[frame]
        ax_3d.scatter([est_pos[0]], [est_pos[1]], [est_pos[2]], 
                     c='orange', s=150, marker='*', 
                     label='Estimated Object', zorder=10)
        
        # Uncertainty ellipsoid
        pos_cov = estimated_covs[frame][0]
        
        # Eigenvalue decomposition for ellipsoid
        eigenvalues, eigenvectors = np.linalg.eig(pos_cov)
        radii = np.sqrt(eigenvalues) * 3  # 3-sigma
        
        # Draw uncertainty ellipsoid
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_ell = radii[0] * np.cos(u) * np.sin(v)
        y_ell = radii[1] * np.sin(u) * np.sin(v)
        z_ell = radii[2] * np.cos(v)
        
        for i in range(len(x_ell)):
            for j in range(len(x_ell[0])):
                point = np.array([x_ell[i,j], y_ell[i,j], z_ell[i,j]])
                rotated = eigenvectors @ point
                x_ell[i,j] = rotated[0] + est_pos[0]
                y_ell[i,j] = rotated[1] + est_pos[1]
                z_ell[i,j] = rotated[2] + est_pos[2]
        
        ax_3d.plot_surface(x_ell, y_ell, z_ell, alpha=0.2, color='orange')
        
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title(f'3D Scene - Frame {frame}')
        ax_3d.legend()
        ax_3d.set_xlim([-4, 4])
        ax_3d.set_ylim([-4, 4])
        ax_3d.set_zlim([-1, 3])
        
        # --- 2D Uncertainty View (Top-down XY) ---
        # Ground truth
        circle = plt.Circle((object_pos[0], object_pos[1]), 0.2, 
                           color='blue', alpha=0.6, label='True Object')
        ax_2d.add_patch(circle)
        
        # Occluder
        rect = plt.Rectangle((x_range[0], y_range[0]), 
                            x_range[1] - x_range[0], 
                            y_range[1] - y_range[0],
                            color='red', alpha=0.3, label='Occluder')
        ax_2d.add_patch(rect)
        
        # Camera
        ax_2d.scatter(cam_pos[0], cam_pos[1], c='green', s=100, 
                     marker='^', label='Camera', zorder=5)
        
        # Estimated position with uncertainty
        est_pos_2d = est_pos[:2]
        pos_cov_2d = pos_cov[:2, :2]
        
        # Eigenvalue decomposition for 2D ellipse
        eigenvalues_2d, eigenvectors_2d = np.linalg.eig(pos_cov_2d)
        angle = np.degrees(np.arctan2(eigenvectors_2d[1, 0], eigenvectors_2d[0, 0]))
        width, height = 2 * 3 * np.sqrt(eigenvalues_2d)  # 3-sigma
        
        ellipse = mpatches.Ellipse(est_pos_2d, width, height, angle=angle,
                                   alpha=0.3, color='orange', label='Uncertainty (3σ)')
        ax_2d.add_patch(ellipse)
        
        ax_2d.scatter(est_pos_2d[0], est_pos_2d[1], c='orange', s=150, 
                     marker='*', label='Estimated', zorder=10)
        
        ax_2d.set_xlabel('X')
        ax_2d.set_ylabel('Y')
        ax_2d.set_title('Top-Down View (XY) with Uncertainty')
        ax_2d.legend(loc='upper right', fontsize=8)
        ax_2d.set_xlim([-4, 4])
        ax_2d.set_ylim([-4, 4])
        ax_2d.set_aspect('equal')
        ax_2d.grid(True, alpha=0.3)
        
        # --- Info Panel ---
        ax_info.axis('off')
        
        info_text = f"Frame: {frame}/{n_frames-1}\n\n"
        info_text += f"Occlusion: {'YES' if occlusion_factors[frame] > 1 else 'NO'}\n"
        info_text += f"Occlusion Factor: {occlusion_factors[frame]:.2f}\n\n"
        
        info_text += f"Camera Position:\n"
        info_text += f"  [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]\n\n"
        
        info_text += f"Estimated Object Position:\n"
        info_text += f"  [{est_pos[0]:.3f}, {est_pos[1]:.3f}, {est_pos[2]:.3f}]\n\n"
        
        info_text += f"Position Uncertainty (σ):\n"
        pos_std = np.sqrt(np.diag(pos_cov))
        info_text += f"  [{pos_std[0]:.4f}, {pos_std[1]:.4f}, {pos_std[2]:.4f}]\n\n"
        
        quat_cov = estimated_covs[frame][1]
        quat_std = np.sqrt(np.diag(quat_cov))
        info_text += f"Orientation Uncertainty (σ):\n"
        info_text += f"  [{quat_std[0]:.4f}, {quat_std[1]:.4f}, {quat_std[2]:.4f}, {quat_std[3]:.4f}]\n"
        
        ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    update(0)
    
    plt.show()

def main():
    print("=" * 70)
    print("Stochastic 6D Pose Estimation with Extended Kalman Filter")
    print("=" * 70)
    print()
    
    # Parameters
    n_frames = 100
    dt = 0.033  # 30 FPS
    
    # Object position (sphere at origin)
    object_pos = np.array([0.0, 0.0, 0.5])
    
    # Occluder bounds (rectangular box)
    occluder_bounds = {
        'x_range': [-0.5, 0.5],
        'y_range': [0.8, 1.5],
        'z_range': [0.0, 1.5]
    }
    
    print("1. Generating ground truth trajectory...")
    camera_positions, camera_orientations = generate_ground_truth_trajectory(n_frames)
    
    print("2. Generating observations with occlusion handling...")
    print("   Choose occlusion failure mode:")
    print("   1 - Return None (no observation)")
    print("   2 - Return random irrelevant pose")
    print("   3 - Return previous pose")
    print("   4 - Return noisy observation (default)")
    
    failure_mode = 'none'  # Change to 'null', 'random', 'previous' to test
    
    observations, occlusion_factors = generate_observations(
        camera_positions, camera_orientations, object_pos, 
        occluder_bounds, failure_mode
    )
    
    print(f"   Using failure mode: {failure_mode}")
    print(f"   Detected {sum(1 for f in occlusion_factors if f > 1)} occluded frames")
    
    print("3. Running Extended Kalman Filter...")
    ekf = ExtendedKalmanFilter(dt=dt)
    
    # Initialize with first observation
    if observations[0] is not None:
        ekf.x[0:3] = observations[0][0:3]
        ekf.x[3:7] = observations[0][3:7]
    
    estimated_poses = []
    estimated_covs = []
    
    for i in range(n_frames):
        # Predict with drift consideration
        drift_scale = 1.0 + 0.5 * (i / n_frames)  # Increasing drift over time
        ekf.predict(drift_scale=drift_scale)
        
        # Update with observation (if available)
        if observations[i] is not None:
            ekf.update(observations[i], occlusion_factor=occlusion_factors[i])
        
        # Store results
        pos, quat = ekf.get_pose()
        pos_cov, quat_cov = ekf.get_covariance()
        estimated_poses.append((pos.copy(), quat.copy()))
        estimated_covs.append((pos_cov.copy(), quat_cov.copy()))
    
    print("4. Launching interactive visualization...")
    print()
    print("Instructions:")
    print("  - Use the slider to navigate through frames")
    print("  - 3D view shows camera path, object, occluder, and uncertainty")
    print("  - 2D view shows top-down projection with uncertainty ellipse")
    print("  - Info panel displays detailed statistics")
    print()
    
    visualize_results(camera_positions, observations, estimated_poses, 
                     estimated_covs, object_pos, occluder_bounds, occlusion_factors)
    
    print("Done!")

if __name__ == "__main__":
    main()
