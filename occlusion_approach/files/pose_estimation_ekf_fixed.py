"""
Stochastic Pose Estimation with Extended Kalman Filter - FIXED VERSION
Handles pose uncertainty, drift, and occlusion scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

# Set random seed for reproducibility
np.random.seed(42)

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for 6D pose estimation
    State: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    """
    
    def __init__(self, dt=0.033):
        self.dt = dt
        self.state_dim = 13
        
        # Initial state
        self.x = np.zeros(self.state_dim)
        self.x[3] = 1.0  # Initialize quaternion to identity
        
        # Initial covariance
        self.P = np.eye(self.state_dim) * 0.1
        self.P[3:7, 3:7] = np.eye(4) * 0.01
        
        # Process noise (motion model uncertainty + drift)
        self.Q = np.eye(self.state_dim) * 0.01
        self.Q[0:3, 0:3] *= 0.05
        self.Q[3:7, 3:7] *= 0.02
        self.Q[7:10, 7:10] *= 0.1
        self.Q[10:13, 10:13] *= 0.1
        
        # Measurement noise
        self.R_normal = np.eye(7) * 0.01
        self.R_normal[0:3, 0:3] *= 0.005
        self.R_normal[3:7, 3:7] *= 0.01
        
    def normalize_quaternion(self):
        """Normalize quaternion part of state"""
        q = self.x[3:7]
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        self.x[3:7] = q_norm
        
    def predict(self, drift_scale=1.0):
        """Prediction step with motion model"""
        pos = self.x[0:3]
        q = self.x[3:7]
        lin_vel = self.x[7:10]
        ang_vel = self.x[10:13]
        
        # Predict position using velocity
        pos_new = pos + lin_vel * self.dt
        
        # Predict orientation using angular velocity
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
            
        # Update state
        self.x[0:3] = pos_new
        self.x[3:7] = q_new
        self.normalize_quaternion()
        
        # Jacobian of motion model
        F = np.eye(self.state_dim)
        F[0:3, 7:10] = np.eye(3) * self.dt
        
        # Update covariance with drift
        Q_drift = self.Q * drift_scale
        self.P = F @ self.P @ F.T + Q_drift
        
        # Ensure symmetry
        self.P = (self.P + self.P.T) / 2
        
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
        """Update step with measurement"""
        if measurement is None:
            return
            
        # Measurement function H
        H = np.zeros((7, self.state_dim))
        H[0:3, 0:3] = np.eye(3)
        H[3:7, 3:7] = np.eye(4)
        
        # Measurement noise increases with occlusion
        R = self.R_normal * occlusion_factor
        
        # Innovation
        z = measurement
        z_pred = H @ self.x
        y = z - z_pred
        
        # Handle quaternion wrap-around
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
        
        # Ensure symmetry
        self.P = (self.P + self.P.T) / 2
        
    def get_pose(self):
        """Get current pose estimate"""
        return self.x[0:3], self.x[3:7]
        
    def get_covariance(self):
        """Get position and orientation covariance"""
        pos_cov = self.P[0:3, 0:3]
        quat_cov = self.P[3:7, 3:7]
        return pos_cov, quat_cov

def load_ply_mesh(filepath):
    """Load a PLY mesh file and return vertices and faces"""
    if not os.path.exists(filepath):
        return None, None
    
    try:
        vertices = []
        faces = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Parse header
        header_end = 0
        n_vertices = 0
        n_faces = 0
        
        for i, line in enumerate(lines):
            if 'element vertex' in line:
                n_vertices = int(line.split()[-1])
            elif 'element face' in line:
                n_faces = int(line.split()[-1])
            elif 'end_header' in line:
                header_end = i + 1
                break
        
        # Parse vertices
        for i in range(header_end, header_end + n_vertices):
            parts = lines[i].split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        # Parse faces
        for i in range(header_end + n_vertices, header_end + n_vertices + n_faces):
            parts = lines[i].split()
            n_verts = int(parts[0])
            face = [int(parts[j+1]) for j in range(n_verts)]
            faces.append(face)
        
        return np.array(vertices), faces
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return None, None

def create_default_mesh(shape='bunny'):
    """Create a default mesh if PLY file not available"""
    if shape == 'bunny':
        # Simple bunny-like shape
        theta = np.linspace(0, 2*np.pi, 20)
        phi = np.linspace(0, np.pi, 15)
        
        vertices = []
        for p in phi:
            for t in theta:
                # Body (ellipsoid)
                x = 0.15 * np.sin(p) * np.cos(t)
                y = 0.12 * np.sin(p) * np.sin(t)
                z = 0.18 * np.cos(p)
                vertices.append([x, y, z])
        
        vertices = np.array(vertices)
        
        # Generate faces (triangulation)
        n_theta = len(theta)
        n_phi = len(phi)
        faces = []
        
        for i in range(n_phi - 1):
            for j in range(n_theta - 1):
                v1 = i * n_theta + j
                v2 = i * n_theta + (j + 1)
                v3 = (i + 1) * n_theta + j
                v4 = (i + 1) * n_theta + (j + 1)
                
                faces.append([v1, v2, v4])
                faces.append([v1, v4, v3])
        
        return vertices, faces
    
    elif shape == 'teapot':
        # Simple teapot-like shape (simplified)
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        
        vertices = []
        for vi in v:
            for ui in u:
                # Body
                r = 0.15 + 0.05 * np.sin(2*vi)
                x = r * np.sin(vi) * np.cos(ui)
                y = r * np.sin(vi) * np.sin(ui)
                z = 0.2 * np.cos(vi)
                vertices.append([x, y, z])
        
        vertices = np.array(vertices)
        
        n_u = len(u)
        n_v = len(v)
        faces = []
        
        for i in range(n_v - 1):
            for j in range(n_u - 1):
                v1 = i * n_u + j
                v2 = i * n_u + (j + 1)
                v3 = (i + 1) * n_u + j
                v4 = (i + 1) * n_u + (j + 1)
                
                faces.append([v1, v2, v4])
                faces.append([v1, v4, v3])
        
        return vertices, faces

def transform_mesh(vertices, position, quaternion):
    """Transform mesh vertices by position and orientation"""
    # Convert quaternion to rotation matrix
    rot = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rot_matrix = rot.as_matrix()
    
    # Apply rotation and translation
    transformed = (rot_matrix @ vertices.T).T + position
    return transformed

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
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(forward, right)
        
        rot_matrix = np.column_stack([right, up, forward])
        quat = R.from_matrix(rot_matrix).as_quat()
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        camera_orientations.append(quat_wxyz)
    
    return camera_positions, np.array(camera_orientations)

def check_occlusion(camera_pos, object_pos, occluder_bounds):
    """Check if object is occluded by a rectangle"""
    ray_dir = object_pos - camera_pos
    t_vals = []
    
    for axis in range(3):
        for bound in ['min', 'max']:
            if axis == 0:
                plane_val = occluder_bounds['x_range'][0 if bound == 'min' else 1]
            elif axis == 1:
                plane_val = occluder_bounds['y_range'][0 if bound == 'min' else 1]
            else:
                plane_val = occluder_bounds['z_range'][0 if bound == 'min' else 1]
            
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
            else:  # 'none'
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

def visualize_results(camera_positions, observations, estimated_poses, 
                     estimated_covs, object_pos, object_quat, occluder_bounds, 
                     occlusion_factors, mesh_vertices, mesh_faces):
    """Interactive visualization with slider"""
    n_frames = len(camera_positions)
    
    fig = plt.figure(figsize=(16, 10))
    
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_2d = fig.add_subplot(222)
    ax_info = fig.add_subplot(224)
    
    plt.subplots_adjust(bottom=0.15, hspace=0.3)
    
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, n_frames-1, valinit=0, valstep=1)
    
    def update(frame):
        frame = int(frame)
        
        ax_3d.clear()
        ax_2d.clear()
        ax_info.clear()
        
        # --- 3D View ---
        # Draw ground truth object mesh
        if mesh_vertices is not None and mesh_faces is not None:
            true_transformed = transform_mesh(mesh_vertices, object_pos, object_quat)
            
            poly_collection = []
            for face in mesh_faces:
                face_verts = [true_transformed[i] for i in face]
                poly_collection.append(face_verts)
            
            poly3d = Poly3DCollection(poly_collection, alpha=0.7, 
                                     facecolor='blue', edgecolor='darkblue', linewidths=0.5)
            ax_3d.add_collection3d(poly3d)
        
        # Draw estimated object mesh (semi-transparent)
        if mesh_vertices is not None and mesh_faces is not None:
            est_pos, est_quat = estimated_poses[frame]
            est_transformed = transform_mesh(mesh_vertices, est_pos, est_quat)
            
            poly_collection_est = []
            for face in mesh_faces:
                face_verts = [est_transformed[i] for i in face]
                poly_collection_est.append(face_verts)
            
            poly3d_est = Poly3DCollection(poly_collection_est, alpha=0.3, 
                                         facecolor='orange', edgecolor='darkorange', linewidths=0.5)
            ax_3d.add_collection3d(poly3d_est)
        
        # Draw occluder
        x_range = occluder_bounds['x_range']
        y_range = occluder_bounds['y_range']
        z_range = occluder_bounds['z_range']
        
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
        
        # Current camera
        cam_pos = camera_positions[frame]
        ax_3d.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]], 
                     c='green', s=100, marker='^', label='Camera')
        
        # Estimated pose marker
        est_pos, est_quat = estimated_poses[frame]
        ax_3d.scatter([est_pos[0]], [est_pos[1]], [est_pos[2]], 
                     c='orange', s=150, marker='*', 
                     label='Estimated Center', zorder=10)
        
        # Uncertainty ellipsoid
        pos_cov = estimated_covs[frame][0]
        eigenvalues, eigenvectors = np.linalg.eig(pos_cov)
        radii = np.sqrt(np.abs(eigenvalues)) * 3  # 3-sigma
        
        # Draw ellipsoid
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
        
        ax_3d.plot_surface(x_ell, y_ell, z_ell, alpha=0.15, color='orange')
        
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title(f'3D Scene - Frame {frame}')
        ax_3d.legend()
        ax_3d.set_xlim([-4, 4])
        ax_3d.set_ylim([-4, 4])
        ax_3d.set_zlim([-1, 3])
        
        # --- 2D View ---
        # True object projection
        if mesh_vertices is not None:
            true_transformed = transform_mesh(mesh_vertices, object_pos, object_quat)
            ax_2d.scatter(true_transformed[:, 0], true_transformed[:, 1], 
                         c='blue', alpha=0.3, s=5, label='True Object')
        
        # Estimated object projection
        if mesh_vertices is not None:
            est_pos, est_quat = estimated_poses[frame]
            est_transformed = transform_mesh(mesh_vertices, est_pos, est_quat)
            ax_2d.scatter(est_transformed[:, 0], est_transformed[:, 1], 
                         c='orange', alpha=0.2, s=5, label='Estimated')
        
        # Occluder
        rect = plt.Rectangle((x_range[0], y_range[0]), 
                            x_range[1] - x_range[0], 
                            y_range[1] - y_range[0],
                            color='red', alpha=0.3, label='Occluder')
        ax_2d.add_patch(rect)
        
        # Camera
        ax_2d.scatter(cam_pos[0], cam_pos[1], c='green', s=100, 
                     marker='^', label='Camera', zorder=5)
        
        # Uncertainty ellipse
        est_pos_2d = est_pos[:2]
        pos_cov_2d = pos_cov[:2, :2]
        
        eigenvalues_2d, eigenvectors_2d = np.linalg.eig(pos_cov_2d)
        angle = np.degrees(np.arctan2(eigenvectors_2d[1, 0], eigenvectors_2d[0, 0]))
        width, height = 2 * 3 * np.sqrt(np.abs(eigenvalues_2d))
        
        ellipse = mpatches.Ellipse(est_pos_2d, width, height, angle=angle,
                                   alpha=0.3, color='orange', label='Uncertainty (3σ)')
        ax_2d.add_patch(ellipse)
        
        ax_2d.set_xlabel('X')
        ax_2d.set_ylabel('Y')
        ax_2d.set_title('Top-Down View (XY)')
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
        
        info_text += f"True Object Position:\n"
        info_text += f"  [{object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}]\n\n"
        
        info_text += f"Estimated Position:\n"
        info_text += f"  [{est_pos[0]:.3f}, {est_pos[1]:.3f}, {est_pos[2]:.3f}]\n\n"
        
        info_text += f"Position Error:\n"
        error = np.linalg.norm(est_pos - object_pos)
        info_text += f"  {error:.4f} m\n\n"
        
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
    print("Stochastic 6D Pose Estimation with Extended Kalman Filter - FIXED")
    print("=" * 70)
    print()
    
    n_frames = 100
    dt = 0.033
    
    # Object position and orientation
    object_pos = np.array([0.0, 0.0, 0.5])
    object_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    # Occluder bounds
    occluder_bounds = {
        'x_range': [-0.5, 0.5],
        'y_range': [0.8, 1.5],
        'z_range': [0.0, 1.5]
    }
    
    print("1. Loading/Creating object mesh...")
    # Try to load PLY file
    ply_path = '/mnt/user-data/uploads/object.ply'
    mesh_vertices, mesh_faces = load_ply_mesh(ply_path)
    
    if mesh_vertices is None:
        print(f"   PLY file not found at {ply_path}")
        print("   Using default 'bunny' mesh shape")
        mesh_vertices, mesh_faces = create_default_mesh('bunny')
    else:
        print(f"   Loaded PLY mesh: {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")
    
    print("2. Generating ground truth trajectory...")
    camera_positions, camera_orientations = generate_ground_truth_trajectory(n_frames)
    
    print("3. Generating observations with occlusion handling...")
    failure_mode = 'none'  # Change to 'null', 'random', 'previous' to test
    
    observations, occlusion_factors = generate_observations(
        camera_positions, camera_orientations, object_pos, 
        occluder_bounds, failure_mode
    )
    
    print(f"   Using failure mode: {failure_mode}")
    print(f"   Detected {sum(1 for f in occlusion_factors if f > 1)} occluded frames")
    
    print("4. Running Extended Kalman Filter...")
    ekf = ExtendedKalmanFilter(dt=dt)
    
    if observations[0] is not None:
        ekf.x[0:3] = observations[0][0:3]
        ekf.x[3:7] = observations[0][3:7]
    
    estimated_poses = []
    estimated_covs = []
    
    for i in range(n_frames):
        drift_scale = 1.0 + 0.5 * (i / n_frames)
        ekf.predict(drift_scale=drift_scale)
        
        if observations[i] is not None:
            ekf.update(observations[i], occlusion_factor=occlusion_factors[i])
        
        pos, quat = ekf.get_pose()
        pos_cov, quat_cov = ekf.get_covariance()
        estimated_poses.append((pos.copy(), quat.copy()))
        estimated_covs.append((pos_cov.copy(), quat_cov.copy()))
    
    print("5. Launching interactive visualization...")
    print()
    print("Instructions:")
    print("  - Blue mesh: Ground truth object")
    print("  - Orange mesh (transparent): Estimated pose")
    print("  - Orange ellipsoid: Uncertainty (3-sigma)")
    print("  - Use slider to navigate frames")
    print()
    
    visualize_results(camera_positions, observations, estimated_poses, 
                     estimated_covs, object_pos, object_quat, occluder_bounds, 
                     occlusion_factors, mesh_vertices, mesh_faces)
    
    print("Done!")

if __name__ == "__main__":
    main()
