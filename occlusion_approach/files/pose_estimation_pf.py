"""
Stochastic Pose Estimation with Particle Filter
Handles pose uncertainty, drift, and occlusion scenarios
Better for non-Gaussian uncertainty and multimodal distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

np.random.seed(42)

class ParticleFilter:
    """
    Particle Filter for 6D pose estimation
    Better handles non-linear dynamics and non-Gaussian noise
    """
    
    def __init__(self, n_particles=1000, dt=0.033):
        self.n_particles = n_particles
        self.dt = dt
        
        # Particles: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        self.particles = np.zeros((n_particles, 13))
        self.particles[:, 3] = 1.0  # Initialize quaternions to identity
        
        # Weights
        self.weights = np.ones(n_particles) / n_particles
        
        # Process noise parameters
        self.pos_noise = 0.05
        self.quat_noise = 0.02
        self.vel_noise = 0.1
        
    def initialize(self, initial_pose, initial_cov=None):
        """Initialize particles around initial pose"""
        if initial_cov is None:
            initial_cov = np.eye(13) * 0.1
        
        # Sample particles from Gaussian around initial state
        for i in range(self.n_particles):
            self.particles[i, :] = initial_pose + np.random.multivariate_normal(
                np.zeros(13), initial_cov
            )
            # Normalize quaternion
            q_norm = np.linalg.norm(self.particles[i, 3:7])
            if q_norm > 0:
                self.particles[i, 3:7] /= q_norm
            else:
                self.particles[i, 3:7] = np.array([1, 0, 0, 0])
    
    def predict(self, drift_scale=1.0):
        """Prediction step with motion model and drift"""
        for i in range(self.n_particles):
            # Extract components
            pos = self.particles[i, 0:3]
            q = self.particles[i, 3:7]
            lin_vel = self.particles[i, 7:10]
            ang_vel = self.particles[i, 10:13]
            
            # Predict position
            pos_new = pos + lin_vel * self.dt
            pos_new += np.random.randn(3) * self.pos_noise * drift_scale
            
            # Predict orientation
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
                q_new = self.quaternion_multiply(q, delta_q)
            else:
                q_new = q
            
            # Add rotation noise
            noise_axis = np.random.randn(3)
            noise_axis_norm = np.linalg.norm(noise_axis)
            if noise_axis_norm > 1e-6:
                noise_axis /= noise_axis_norm
                noise_angle = np.random.randn() * self.quat_noise * drift_scale
                noise_q = np.array([
                    np.cos(noise_angle/2),
                    noise_axis[0] * np.sin(noise_angle/2),
                    noise_axis[1] * np.sin(noise_angle/2),
                    noise_axis[2] * np.sin(noise_angle/2)
                ])
                q_new = self.quaternion_multiply(q_new, noise_q)
            
            # Update particle
            self.particles[i, 0:3] = pos_new
            self.particles[i, 3:7] = q_new / (np.linalg.norm(q_new) + 1e-8)
            
            # Add velocity noise
            self.particles[i, 7:10] += np.random.randn(3) * self.vel_noise * drift_scale
            self.particles[i, 10:13] += np.random.randn(3) * self.vel_noise * drift_scale
    
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
    
    def quaternion_distance(self, q1, q2):
        """Compute angular distance between quaternions"""
        dot = np.abs(np.sum(q1 * q2))
        dot = np.clip(dot, -1, 1)
        return 2 * np.arccos(dot)
    
    def update(self, measurement, occlusion_factor=1.0):
        """Update step with measurement"""
        if measurement is None:
            # No measurement - uniform weights (pure prediction)
            self.weights = np.ones(self.n_particles) / self.n_particles
            return
        
        # Compute likelihood for each particle
        pos_obs = measurement[0:3]
        quat_obs = measurement[3:7]
        
        # Measurement noise (increases with occlusion)
        pos_std = 0.02 * occlusion_factor
        quat_std = 0.01 * occlusion_factor
        
        for i in range(self.n_particles):
            pos_particle = self.particles[i, 0:3]
            quat_particle = self.particles[i, 3:7]
            
            # Position likelihood (Gaussian)
            pos_diff = pos_particle - pos_obs
            pos_likelihood = np.exp(-0.5 * np.sum(pos_diff**2) / pos_std**2)
            
            # Orientation likelihood (angular distance)
            quat_dist = self.quaternion_distance(quat_particle, quat_obs)
            quat_likelihood = np.exp(-0.5 * quat_dist**2 / quat_std**2)
            
            # Combined likelihood
            self.weights[i] = pos_likelihood * quat_likelihood
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-10:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Resample if effective sample size is low
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.n_particles / 2:
            self.resample()
    
    def resample(self):
        """Resample particles based on weights"""
        indices = np.random.choice(
            self.n_particles, 
            size=self.n_particles, 
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def get_estimate(self):
        """Get weighted mean estimate"""
        # Weighted mean for position
        pos_mean = np.sum(self.particles[:, 0:3] * self.weights[:, np.newaxis], axis=0)
        
        # Weighted mean for quaternion (approximate)
        quat_mean = np.sum(self.particles[:, 3:7] * self.weights[:, np.newaxis], axis=0)
        quat_mean /= (np.linalg.norm(quat_mean) + 1e-8)
        
        return pos_mean, quat_mean
    
    def get_covariance(self):
        """Get covariance estimates"""
        pos_mean, quat_mean = self.get_estimate()
        
        # Position covariance
        pos_diff = self.particles[:, 0:3] - pos_mean
        pos_cov = np.zeros((3, 3))
        for i in range(self.n_particles):
            pos_cov += self.weights[i] * np.outer(pos_diff[i], pos_diff[i])
        
        # Quaternion covariance
        quat_diff = self.particles[:, 3:7] - quat_mean
        quat_cov = np.zeros((4, 4))
        for i in range(self.n_particles):
            quat_cov += self.weights[i] * np.outer(quat_diff[i], quat_diff[i])
        
        return pos_cov, quat_cov

# Import helper functions from EKF version
def generate_ground_truth_trajectory(n_frames=100):
    """Generate a circular trajectory around an object"""
    t = np.linspace(0, 2*np.pi, n_frames)
    radius = 3.0
    height = 1.5
    
    camera_positions = np.column_stack([
        radius * np.cos(t),
        radius * np.sin(t),
        np.ones(n_frames) * height
    ])
    
    camera_orientations = []
    for i in range(n_frames):
        forward = -camera_positions[i] / np.linalg.norm(camera_positions[i])
        right = np.cross([0, 0, 1], forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        rot_matrix = np.column_stack([right, up, forward])
        quat = R.from_matrix(rot_matrix).as_quat()
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        camera_orientations.append(quat_wxyz)
    
    return camera_positions, np.array(camera_orientations)

def check_occlusion(camera_pos, object_pos, occluder_bounds):
    """Check if object is occluded by a rectangle"""
    ray_dir = object_pos - camera_pos
    
    for axis in range(3):
        for bound_idx in [0, 1]:
            if axis == 0:
                plane_val = occluder_bounds['x_range'][bound_idx]
            elif axis == 1:
                plane_val = occluder_bounds['y_range'][bound_idx]
            else:
                plane_val = occluder_bounds['z_range'][bound_idx]
            
            if abs(ray_dir[axis]) > 1e-6:
                t = (plane_val - camera_pos[axis]) / ray_dir[axis]
                if 0 < t < 1:
                    intersect = camera_pos + t * ray_dir
                    if (occluder_bounds['x_range'][0] <= intersect[0] <= occluder_bounds['x_range'][1] and
                        occluder_bounds['y_range'][0] <= intersect[1] <= occluder_bounds['y_range'][1] and
                        occluder_bounds['z_range'][0] <= intersect[2] <= occluder_bounds['z_range'][1]):
                        return True
    return False

def generate_observations(camera_positions, camera_orientations, object_pos, 
                         occluder_bounds, failure_mode='none'):
    """Generate observations with occlusion handling"""
    n_frames = len(camera_positions)
    observations = []
    occlusion_factors = []
    prev_obs = None
    
    for i in range(n_frames):
        is_occluded = check_occlusion(camera_positions[i], object_pos, occluder_bounds)
        
        if is_occluded:
            occlusion_factor = 10.0
            
            if failure_mode == 'null':
                obs = None
            elif failure_mode == 'random':
                obs = np.concatenate([
                    np.random.randn(3) * 2.0,
                    R.random().as_quat()[[3, 0, 1, 2]]
                ])
            elif failure_mode == 'previous':
                obs = prev_obs if prev_obs is not None else None
            else:
                obs = np.concatenate([
                    object_pos + np.random.randn(3) * 0.3,
                    camera_orientations[i] + np.random.randn(4) * 0.2
                ])
                obs[3:7] = obs[3:7] / np.linalg.norm(obs[3:7])
        else:
            occlusion_factor = 1.0
            obs = np.concatenate([
                object_pos + np.random.randn(3) * 0.02,
                camera_orientations[i] + np.random.randn(4) * 0.01
            ])
            obs[3:7] = obs[3:7] / np.linalg.norm(obs[3:7])
            prev_obs = obs
        
        observations.append(obs)
        occlusion_factors.append(occlusion_factor)
    
    return observations, occlusion_factors

def visualize_results_pf(camera_positions, observations, estimated_poses, 
                        estimated_covs, particles_history, object_pos, 
                        occluder_bounds, occlusion_factors):
    """Interactive visualization for Particle Filter with particle cloud"""
    n_frames = len(camera_positions)
    
    fig = plt.figure(figsize=(18, 10))
    
    ax_3d = fig.add_subplot(131, projection='3d')
    ax_particles = fig.add_subplot(132, projection='3d')
    ax_2d = fig.add_subplot(233)
    ax_info = fig.add_subplot(236)
    
    plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.3)
    
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, n_frames-1, valinit=0, valstep=1)
    
    def update(frame):
        frame = int(frame)
        
        ax_3d.clear()
        ax_particles.clear()
        ax_2d.clear()
        ax_info.clear()
        
        # --- 3D Scene View ---
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_sphere = 0.2 * np.cos(u) * np.sin(v) + object_pos[0]
        y_sphere = 0.2 * np.sin(u) * np.sin(v) + object_pos[1]
        z_sphere = 0.2 * np.cos(v) + object_pos[2]
        ax_3d.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.6, color='blue')
        
        # Occluder
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        x_range = occluder_bounds['x_range']
        y_range = occluder_bounds['y_range']
        z_range = occluder_bounds['z_range']
        
        vertices = [
            [x_range[0], y_range[0], z_range[0]], [x_range[1], y_range[0], z_range[0]],
            [x_range[1], y_range[1], z_range[0]], [x_range[0], y_range[1], z_range[0]],
            [x_range[0], y_range[0], z_range[1]], [x_range[1], y_range[0], z_range[1]],
            [x_range[1], y_range[1], z_range[1]], [x_range[0], y_range[1], z_range[1]]
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
                                                edgecolor='darkred'))
        
        # Camera and estimate
        cam_pos = camera_positions[frame]
        ax_3d.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]], 
                     c='green', s=100, marker='^', label='Camera')
        
        est_pos = estimated_poses[frame][0]
        ax_3d.scatter([est_pos[0]], [est_pos[1]], [est_pos[2]], 
                     c='orange', s=150, marker='*', label='Estimate', zorder=10)
        
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title(f'3D Scene - Frame {frame}')
        ax_3d.legend()
        ax_3d.set_xlim([-4, 4])
        ax_3d.set_ylim([-4, 4])
        ax_3d.set_zlim([-1, 3])
        
        # --- Particle Cloud View ---
        particles = particles_history[frame]
        ax_particles.scatter(particles[:, 0], particles[:, 1], particles[:, 2],
                           c='cyan', alpha=0.1, s=1)
        
        # Ground truth object
        ax_particles.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.6, color='blue')
        
        # Estimate
        ax_particles.scatter([est_pos[0]], [est_pos[1]], [est_pos[2]], 
                           c='orange', s=200, marker='*', label='Estimate')
        
        ax_particles.set_xlabel('X')
        ax_particles.set_ylabel('Y')
        ax_particles.set_zlabel('Z')
        ax_particles.set_title(f'Particle Distribution (1000 particles)')
        ax_particles.legend()
        ax_particles.set_xlim([-2, 2])
        ax_particles.set_ylim([-2, 2])
        ax_particles.set_zlim([-1, 2])
        
        # --- 2D Top-Down View ---
        circle = plt.Circle((object_pos[0], object_pos[1]), 0.2, 
                           color='blue', alpha=0.6, label='True Object')
        ax_2d.add_patch(circle)
        
        rect = plt.Rectangle((x_range[0], y_range[0]), 
                            x_range[1] - x_range[0], y_range[1] - y_range[0],
                            color='red', alpha=0.3, label='Occluder')
        ax_2d.add_patch(rect)
        
        # Particle scatter
        ax_2d.scatter(particles[:, 0], particles[:, 1], c='cyan', 
                     alpha=0.05, s=5, label='Particles')
        
        ax_2d.scatter(cam_pos[0], cam_pos[1], c='green', s=100, 
                     marker='^', label='Camera', zorder=5)
        
        est_pos_2d = est_pos[:2]
        pos_cov_2d = estimated_covs[frame][0][:2, :2]
        
        eigenvalues_2d, eigenvectors_2d = np.linalg.eig(pos_cov_2d)
        angle = np.degrees(np.arctan2(eigenvectors_2d[1, 0], eigenvectors_2d[0, 0]))
        width, height = 2 * 3 * np.sqrt(eigenvalues_2d)
        
        ellipse = mpatches.Ellipse(est_pos_2d, width, height, angle=angle,
                                   alpha=0.3, color='orange', label='Uncertainty (3σ)')
        ax_2d.add_patch(ellipse)
        
        ax_2d.scatter(est_pos_2d[0], est_pos_2d[1], c='orange', s=150, 
                     marker='*', label='Estimate', zorder=10)
        
        ax_2d.set_xlabel('X')
        ax_2d.set_ylabel('Y')
        ax_2d.set_title('Top-Down View with Particles')
        ax_2d.legend(loc='upper right', fontsize=7)
        ax_2d.set_xlim([-4, 4])
        ax_2d.set_ylim([-4, 4])
        ax_2d.set_aspect('equal')
        ax_2d.grid(True, alpha=0.3)
        
        # --- Info Panel ---
        ax_info.axis('off')
        
        info_text = f"Frame: {frame}/{n_frames-1}\n\n"
        info_text += f"Method: Particle Filter (1000 particles)\n\n"
        info_text += f"Occlusion: {'YES' if occlusion_factors[frame] > 1 else 'NO'}\n"
        info_text += f"Occlusion Factor: {occlusion_factors[frame]:.2f}\n\n"
        
        info_text += f"Estimated Position:\n"
        info_text += f"  [{est_pos[0]:.3f}, {est_pos[1]:.3f}, {est_pos[2]:.3f}]\n\n"
        
        pos_cov = estimated_covs[frame][0]
        pos_std = np.sqrt(np.diag(pos_cov))
        info_text += f"Position Uncertainty (σ):\n"
        info_text += f"  [{pos_std[0]:.4f}, {pos_std[1]:.4f}, {pos_std[2]:.4f}]\n\n"
        
        info_text += f"Particle spread represents\n"
        info_text += f"multimodal uncertainty distribution"
        
        ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    update(0)
    plt.show()

def main():
    print("=" * 70)
    print("Stochastic 6D Pose Estimation with Particle Filter")
    print("=" * 70)
    print()
    
    n_frames = 100
    dt = 0.033
    n_particles = 1000
    
    object_pos = np.array([0.0, 0.0, 0.5])
    
    occluder_bounds = {
        'x_range': [-0.5, 0.5],
        'y_range': [0.8, 1.5],
        'z_range': [0.0, 1.5]
    }
    
    print("1. Generating ground truth trajectory...")
    camera_positions, camera_orientations = generate_ground_truth_trajectory(n_frames)
    
    print("2. Generating observations...")
    failure_mode = 'none'
    observations, occlusion_factors = generate_observations(
        camera_positions, camera_orientations, object_pos, 
        occluder_bounds, failure_mode
    )
    
    print(f"   Detected {sum(1 for f in occlusion_factors if f > 1)} occluded frames")
    
    print(f"3. Running Particle Filter with {n_particles} particles...")
    pf = ParticleFilter(n_particles=n_particles, dt=dt)
    
    # Initialize
    if observations[0] is not None:
        initial_state = np.zeros(13)
        initial_state[0:3] = observations[0][0:3]
        initial_state[3:7] = observations[0][3:7]
        pf.initialize(initial_state)
    
    estimated_poses = []
    estimated_covs = []
    particles_history = []
    
    for i in range(n_frames):
        drift_scale = 1.0 + 0.5 * (i / n_frames)
        pf.predict(drift_scale=drift_scale)
        
        if observations[i] is not None:
            pf.update(observations[i], occlusion_factor=occlusion_factors[i])
        
        pos, quat = pf.get_estimate()
        pos_cov, quat_cov = pf.get_covariance()
        estimated_poses.append((pos.copy(), quat.copy()))
        estimated_covs.append((pos_cov.copy(), quat_cov.copy()))
        particles_history.append(pf.particles.copy())
    
    print("4. Launching interactive visualization...")
    visualize_results_pf(camera_positions, observations, estimated_poses, 
                        estimated_covs, particles_history, object_pos, 
                        occluder_bounds, occlusion_factors)
    
    print("Done!")

if __name__ == "__main__":
    main()
