"""
Comparison of EKF vs Particle Filter for 6D Pose Estimation
Shows both methods side-by-side with performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

# Import both filter classes
import sys
sys.path.insert(0, '/home/claude')

from pose_estimation_ekf import ExtendedKalmanFilter, generate_ground_truth_trajectory
from pose_estimation_ekf import check_occlusion, generate_observations
from pose_estimation_pf import ParticleFilter

np.random.seed(42)

def compute_metrics(true_pos, est_pos):
    """Compute tracking metrics"""
    position_error = np.linalg.norm(true_pos - est_pos)
    return position_error

def run_comparison():
    print("=" * 80)
    print("EKF vs Particle Filter Comparison for 6D Pose Estimation")
    print("=" * 80)
    print()
    
    # Parameters
    n_frames = 100
    dt = 0.033
    n_particles = 1000
    
    object_pos = np.array([0.0, 0.0, 0.5])
    
    occluder_bounds = {
        'x_range': [-0.5, 0.5],
        'y_range': [0.8, 1.5],
        'z_range': [0.0, 1.5]
    }
    
    print("1. Generating test scenario...")
    camera_positions, camera_orientations = generate_ground_truth_trajectory(n_frames)
    
    failure_mode = 'none'
    observations, occlusion_factors = generate_observations(
        camera_positions, camera_orientations, object_pos, 
        occluder_bounds, failure_mode
    )
    
    n_occluded = sum(1 for f in occlusion_factors if f > 1)
    print(f"   Frames: {n_frames}")
    print(f"   Occluded frames: {n_occluded}")
    print()
    
    # Run EKF
    print("2. Running Extended Kalman Filter...")
    ekf = ExtendedKalmanFilter(dt=dt)
    
    if observations[0] is not None:
        ekf.x[0:3] = observations[0][0:3]
        ekf.x[3:7] = observations[0][3:7]
    
    ekf_poses = []
    ekf_covs = []
    ekf_times = []
    
    for i in range(n_frames):
        t_start = time.time()
        
        drift_scale = 1.0 + 0.5 * (i / n_frames)
        ekf.predict(drift_scale=drift_scale)
        
        if observations[i] is not None:
            ekf.update(observations[i], occlusion_factor=occlusion_factors[i])
        
        pos, quat = ekf.get_pose()
        pos_cov, quat_cov = ekf.get_covariance()
        
        ekf_poses.append((pos.copy(), quat.copy()))
        ekf_covs.append((pos_cov.copy(), quat_cov.copy()))
        ekf_times.append(time.time() - t_start)
    
    print(f"   Average time per frame: {np.mean(ekf_times)*1000:.2f} ms")
    print()
    
    # Run Particle Filter
    print(f"3. Running Particle Filter ({n_particles} particles)...")
    pf = ParticleFilter(n_particles=n_particles, dt=dt)
    
    if observations[0] is not None:
        initial_state = np.zeros(13)
        initial_state[0:3] = observations[0][0:3]
        initial_state[3:7] = observations[0][3:7]
        pf.initialize(initial_state)
    
    pf_poses = []
    pf_covs = []
    pf_times = []
    
    for i in range(n_frames):
        t_start = time.time()
        
        drift_scale = 1.0 + 0.5 * (i / n_frames)
        pf.predict(drift_scale=drift_scale)
        
        if observations[i] is not None:
            pf.update(observations[i], occlusion_factor=occlusion_factors[i])
        
        pos, quat = pf.get_estimate()
        pos_cov, quat_cov = pf.get_covariance()
        
        pf_poses.append((pos.copy(), quat.copy()))
        pf_covs.append((pos_cov.copy(), quat_cov.copy()))
        pf_times.append(time.time() - t_start)
    
    print(f"   Average time per frame: {np.mean(pf_times)*1000:.2f} ms")
    print()
    
    # Compute metrics
    print("4. Computing performance metrics...")
    
    ekf_errors = []
    pf_errors = []
    
    for i in range(n_frames):
        ekf_pos = ekf_poses[i][0]
        pf_pos = pf_poses[i][0]
        
        ekf_error = compute_metrics(object_pos, ekf_pos)
        pf_error = compute_metrics(object_pos, pf_pos)
        
        ekf_errors.append(ekf_error)
        pf_errors.append(pf_error)
    
    print(f"   EKF Mean Error: {np.mean(ekf_errors):.4f} m")
    print(f"   PF Mean Error:  {np.mean(pf_errors):.4f} m")
    print(f"   EKF Max Error:  {np.max(ekf_errors):.4f} m")
    print(f"   PF Max Error:   {np.max(pf_errors):.4f} m")
    print()
    
    # Visualization
    print("5. Creating comparison visualization...")
    visualize_comparison(camera_positions, ekf_poses, pf_poses, ekf_covs, pf_covs,
                        ekf_errors, pf_errors, ekf_times, pf_times,
                        object_pos, occluder_bounds, occlusion_factors)

def visualize_comparison(camera_positions, ekf_poses, pf_poses, ekf_covs, pf_covs,
                        ekf_errors, pf_errors, ekf_times, pf_times,
                        object_pos, occluder_bounds, occlusion_factors):
    """Side-by-side comparison visualization"""
    n_frames = len(camera_positions)
    
    fig = plt.figure(figsize=(20, 12))
    
    # EKF 3D view
    ax_ekf_3d = fig.add_subplot(2, 3, 1, projection='3d')
    # PF 3D view
    ax_pf_3d = fig.add_subplot(2, 3, 2, projection='3d')
    # Metrics view
    ax_metrics = fig.add_subplot(2, 3, 3)
    
    # EKF 2D view
    ax_ekf_2d = fig.add_subplot(2, 3, 4)
    # PF 2D view
    ax_pf_2d = fig.add_subplot(2, 3, 5)
    # Performance view
    ax_perf = fig.add_subplot(2, 3, 6)
    
    plt.subplots_adjust(bottom=0.08, hspace=0.25, wspace=0.3)
    
    # Slider
    ax_slider = plt.axes([0.15, 0.02, 0.7, 0.02])
    slider = Slider(ax_slider, 'Frame', 0, n_frames-1, valinit=0, valstep=1)
    
    def draw_3d_scene(ax, est_pos, pos_cov, title, frame):
        ax.clear()
        
        # Ground truth object
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_sphere = 0.2 * np.cos(u) * np.sin(v) + object_pos[0]
        y_sphere = 0.2 * np.sin(u) * np.sin(v) + object_pos[1]
        z_sphere = 0.2 * np.cos(v) + object_pos[2]
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.6, color='blue')
        
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
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor='red'))
        
        # Camera
        cam_pos = camera_positions[frame]
        ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]], 
                  c='green', s=80, marker='^')
        
        # Estimate
        ax.scatter([est_pos[0]], [est_pos[1]], [est_pos[2]], 
                  c='orange', s=150, marker='*', zorder=10)
        
        # Uncertainty ellipsoid
        eigenvalues, eigenvectors = np.linalg.eig(pos_cov)
        radii = np.sqrt(np.abs(eigenvalues)) * 3
        
        u_ell, v_ell = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
        x_ell = radii[0] * np.cos(u_ell) * np.sin(v_ell)
        y_ell = radii[1] * np.sin(u_ell) * np.sin(v_ell)
        z_ell = radii[2] * np.cos(v_ell)
        
        for i in range(len(x_ell)):
            for j in range(len(x_ell[0])):
                point = np.array([x_ell[i,j], y_ell[i,j], z_ell[i,j]])
                rotated = eigenvectors @ point
                x_ell[i,j] = rotated[0] + est_pos[0]
                y_ell[i,j] = rotated[1] + est_pos[1]
                z_ell[i,j] = rotated[2] + est_pos[2]
        
        ax.plot_surface(x_ell, y_ell, z_ell, alpha=0.15, color='orange')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-0.5, 2])
        ax.view_init(elev=20, azim=45)
    
    def update(frame):
        frame = int(frame)
        
        # EKF 3D
        ekf_pos = ekf_poses[frame][0]
        ekf_pos_cov = ekf_covs[frame][0]
        draw_3d_scene(ax_ekf_3d, ekf_pos, ekf_pos_cov, 'Extended Kalman Filter', frame)
        
        # PF 3D
        pf_pos = pf_poses[frame][0]
        pf_pos_cov = pf_covs[frame][0]
        draw_3d_scene(ax_pf_3d, pf_pos, pf_pos_cov, 'Particle Filter', frame)
        
        # Error metrics over time
        ax_metrics.clear()
        frames = np.arange(n_frames)
        ax_metrics.plot(frames, ekf_errors, 'b-', label='EKF', linewidth=2)
        ax_metrics.plot(frames, pf_errors, 'r-', label='PF', linewidth=2)
        ax_metrics.axvline(frame, color='gray', linestyle='--', alpha=0.5)
        ax_metrics.fill_between(frames, 0, 1, 
                               where=[f > 1 for f in occlusion_factors],
                               alpha=0.2, color='red', label='Occluded')
        ax_metrics.set_xlabel('Frame')
        ax_metrics.set_ylabel('Position Error (m)')
        ax_metrics.set_title('Position Error Over Time')
        ax_metrics.legend()
        ax_metrics.grid(True, alpha=0.3)
        ax_metrics.set_ylim([0, max(max(ekf_errors), max(pf_errors)) * 1.1])
        
        # EKF 2D
        ax_ekf_2d.clear()
        ax_ekf_2d.scatter(object_pos[0], object_pos[1], c='blue', 
                         s=200, marker='o', label='True', zorder=5)
        ax_ekf_2d.scatter(ekf_pos[0], ekf_pos[1], c='orange', 
                         s=150, marker='*', label='EKF Est.', zorder=10)
        
        eigenvalues_2d, eigenvectors_2d = np.linalg.eig(ekf_pos_cov[:2, :2])
        angle = np.degrees(np.arctan2(eigenvectors_2d[1, 0], eigenvectors_2d[0, 0]))
        width, height = 2 * 3 * np.sqrt(np.abs(eigenvalues_2d))
        
        import matplotlib.patches as mpatches
        ellipse = mpatches.Ellipse(ekf_pos[:2], width, height, angle=angle,
                                   alpha=0.3, color='orange')
        ax_ekf_2d.add_patch(ellipse)
        
        ax_ekf_2d.set_xlabel('X')
        ax_ekf_2d.set_ylabel('Y')
        ax_ekf_2d.set_title(f'EKF Top View (Error: {ekf_errors[frame]:.4f}m)')
        ax_ekf_2d.set_xlim([-1.5, 1.5])
        ax_ekf_2d.set_ylim([-1.5, 1.5])
        ax_ekf_2d.set_aspect('equal')
        ax_ekf_2d.grid(True, alpha=0.3)
        ax_ekf_2d.legend()
        
        # PF 2D
        ax_pf_2d.clear()
        ax_pf_2d.scatter(object_pos[0], object_pos[1], c='blue', 
                        s=200, marker='o', label='True', zorder=5)
        ax_pf_2d.scatter(pf_pos[0], pf_pos[1], c='red', 
                        s=150, marker='*', label='PF Est.', zorder=10)
        
        eigenvalues_2d, eigenvectors_2d = np.linalg.eig(pf_pos_cov[:2, :2])
        angle = np.degrees(np.arctan2(eigenvectors_2d[1, 0], eigenvectors_2d[0, 0]))
        width, height = 2 * 3 * np.sqrt(np.abs(eigenvalues_2d))
        
        ellipse = mpatches.Ellipse(pf_pos[:2], width, height, angle=angle,
                                   alpha=0.3, color='red')
        ax_pf_2d.add_patch(ellipse)
        
        ax_pf_2d.set_xlabel('X')
        ax_pf_2d.set_ylabel('Y')
        ax_pf_2d.set_title(f'PF Top View (Error: {pf_errors[frame]:.4f}m)')
        ax_pf_2d.set_xlim([-1.5, 1.5])
        ax_pf_2d.set_ylim([-1.5, 1.5])
        ax_pf_2d.set_aspect('equal')
        ax_pf_2d.grid(True, alpha=0.3)
        ax_pf_2d.legend()
        
        # Performance metrics
        ax_perf.clear()
        
        methods = ['EKF', 'PF']
        mean_errors = [np.mean(ekf_errors), np.mean(pf_errors)]
        mean_times = [np.mean(ekf_times) * 1000, np.mean(pf_times) * 1000]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax_perf_twin = ax_perf.twinx()
        
        bars1 = ax_perf.bar(x - width/2, mean_errors, width, 
                           label='Mean Error (m)', color='skyblue')
        bars2 = ax_perf_twin.bar(x + width/2, mean_times, width, 
                                label='Avg Time (ms)', color='lightcoral')
        
        ax_perf.set_ylabel('Mean Error (m)', color='skyblue')
        ax_perf_twin.set_ylabel('Avg Time per Frame (ms)', color='lightcoral')
        ax_perf.set_title('Performance Comparison')
        ax_perf.set_xticks(x)
        ax_perf.set_xticklabels(methods)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax_perf.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax_perf_twin.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        fig.suptitle(f'EKF vs Particle Filter - Frame {frame}/{n_frames-1}', 
                    fontsize=14, fontweight='bold')
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    update(0)
    
    plt.show()

if __name__ == "__main__":
    run_comparison()
