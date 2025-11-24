"""
6D Pose Estimation using Willems' Fundamental Lemma
Data-Driven Control with Gaussian Process for Pose Tracking

Willems et al., 2005: "A note on persistency of excitation"
Direct data-driven control using behavioral systems theory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from scipy.linalg import hankel, lstsq
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import matplotlib.patches as mpatches
import os

np.random.seed(42)

class WillemsDDC:
    """
    Data-Driven Control based on Willems' Fundamental Lemma
    
    Key Idea: Any trajectory of a linear system can be represented as a 
    linear combination of past trajectories (if data is persistently exciting)
    
    For pose control:
    - Use historical pose observations as basis
    - GP models uncertainty in behavioral representation
    - Control signal adjusts estimate based on innovations
    """
    
    def __init__(self, lag=10, horizon=5):
        self.lag = lag  # History length (L)
        self.horizon = horizon  # Prediction horizon (T)
        
        # Data matrices for behavioral representation
        self.U_past = None  # Past inputs (observations)
        self.Y_past = None  # Past outputs (poses)
        
        # Historical data storage
        self.observation_history = []
        self.pose_history = []
        
        # Current state
        self.current_pose = np.zeros(7)  # [x, y, z, qw, qx, qy, qz]
        self.current_pose[3] = 1.0  # Identity quaternion
        
        # Gaussian Process for uncertainty
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        # Uncertainty tracking
        self.uncertainty = np.eye(7) * 0.1
        
        # Control gains
        self.K_feedback = 0.7  # Feedback gain
        self.K_innovation = 0.3  # Innovation gain
        
    def add_trajectory_data(self, observation, pose):
        """
        Add observation-pose pair to historical data
        This builds the behavioral dataset for Willems' lemma
        """
        if observation is not None:
            self.observation_history.append(observation.copy())
            self.pose_history.append(pose.copy())
            
            # Keep limited history for computational efficiency
            max_history = 200
            if len(self.observation_history) > max_history:
                self.observation_history.pop(0)
                self.pose_history.pop(0)
    
    def build_hankel_matrices(self):
        """
        Build Hankel matrices U and Y for behavioral representation
        
        Willems' Lemma: If U is persistently exciting, any trajectory
        can be written as [u; y] = [U; Y] * g for some weights g
        """
        if len(self.observation_history) < self.lag + self.horizon:
            return False
        
        n_data = len(self.observation_history)
        n_cols = n_data - self.lag - self.horizon + 1
        
        if n_cols < 1:
            return False
        
        # Build Hankel matrices
        U_past_list = []
        Y_past_list = []
        
        for i in range(n_cols):
            # Past inputs (observations)
            u_past = np.array(self.observation_history[i:i+self.lag]).flatten()
            U_past_list.append(u_past)
            
            # Past outputs (poses)
            y_past = np.array(self.pose_history[i:i+self.lag]).flatten()
            Y_past_list.append(y_past)
        
        self.U_past = np.array(U_past_list).T
        self.Y_past = np.array(Y_past_list).T
        
        return True
    
    def predict_with_willems(self, recent_observations, occlusion_factor=1.0):
        """
        Predict next pose using Willems' Fundamental Lemma
        
        Find weights g such that recent_observations ≈ U_past @ g
        Then predicted pose = Y_past @ g
        
        Handles occlusion by adjusting innovation gain
        """
        if self.U_past is None or len(recent_observations) < self.lag:
            return self.current_pose, self.uncertainty
        
        # Form recent observation vector
        u_recent = np.array(recent_observations[-self.lag:]).flatten()
        
        # Solve for behavioral weights: U_past @ g ≈ u_recent
        # This finds how to represent current trajectory from past data
        try:
            # Regularized least squares (handles noise)
            lambda_reg = 0.01 * occlusion_factor  # More regularization during occlusion
            A = self.U_past.T @ self.U_past + lambda_reg * np.eye(self.U_past.shape[1])
            b = self.U_past.T @ u_recent
            g = np.linalg.solve(A, b)
            
            # Predict pose using same weights
            y_predicted = self.Y_past @ g
            
            # Reshape to pose format
            n_states = 7
            y_predicted_poses = y_predicted.reshape(self.lag, n_states)
            predicted_pose = y_predicted_poses[-1]  # Take most recent
            
            # Normalize quaternion
            predicted_pose[3:7] /= (np.linalg.norm(predicted_pose[3:7]) + 1e-8)
            
            # Compute residual for uncertainty estimate
            residual = np.linalg.norm(u_recent - self.U_past @ g)
            
            # Update uncertainty based on residual and occlusion
            base_uncertainty = 0.01
            uncertainty_scale = base_uncertainty * (1 + residual) * occlusion_factor
            self.uncertainty = np.eye(7) * uncertainty_scale
            self.uncertainty[3:7, 3:7] *= 0.5  # Lower uncertainty for orientation
            
            return predicted_pose, self.uncertainty
            
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            return self.current_pose, self.uncertainty * 2.0
    
    def control_update(self, measurement, occlusion_factor=1.0):
        """
        Data-driven control update
        
        1. Predict using Willems' lemma (data-driven model)
        2. Compute innovation (measurement - prediction)
        3. Apply feedback control to adjust estimate
        """
        if measurement is None:
            # No measurement - pure prediction (open-loop)
            if len(self.observation_history) >= self.lag:
                predicted, unc = self.predict_with_willems(
                    self.observation_history, 
                    occlusion_factor
                )
                self.current_pose = predicted
                self.uncertainty = unc * occlusion_factor
            else:
                # Not enough data - maintain current pose
                self.uncertainty *= 1.1  # Increase uncertainty
            return
        
        # Build/update behavioral matrices
        self.build_hankel_matrices()
        
        # Predict using Willems' lemma
        if len(self.observation_history) >= self.lag:
            predicted, unc = self.predict_with_willems(
                self.observation_history,
                occlusion_factor
            )
        else:
            predicted = self.current_pose
            unc = self.uncertainty
        
        # Innovation (measurement residual)
        innovation = measurement - predicted
        
        # Handle quaternion wrap-around
        if np.dot(measurement[3:7], predicted[3:7]) < 0:
            innovation[3:7] = -innovation[3:7]
        
        # Control law: feedback + innovation
        # During good observations: high innovation gain
        # During occlusion: rely more on prediction
        innovation_gain = self.K_innovation / occlusion_factor
        feedback_gain = self.K_feedback
        
        # Update pose
        self.current_pose = (
            feedback_gain * predicted + 
            innovation_gain * measurement + 
            (1 - feedback_gain - innovation_gain) * self.current_pose
        )
        
        # Normalize quaternion
        self.current_pose[3:7] /= (np.linalg.norm(self.current_pose[3:7]) + 1e-8)
        
        # Update uncertainty (lower with good measurements)
        self.uncertainty = unc / (1.0 + innovation_gain)
        
        # Add to trajectory database
        self.add_trajectory_data(measurement, self.current_pose)
    
    def fit_gp_uncertainty(self):
        """
        Fit Gaussian Process to model uncertainty over time
        This captures non-Gaussian, time-varying uncertainty
        """
        if len(self.pose_history) < 10:
            return
        
        # Use recent history
        recent_poses = np.array(self.pose_history[-50:])
        time_steps = np.arange(len(recent_poses)).reshape(-1, 1)
        
        try:
            # Fit GP to position trajectory
            self.gp.fit(time_steps, recent_poses[:, :3])
        except:
            pass  # Fail silently if GP fitting fails
    
    def get_gp_prediction(self, steps_ahead=1):
        """
        Use GP to predict future poses and uncertainty
        """
        if len(self.pose_history) < 10:
            return None, None
        
        try:
            current_time = len(self.pose_history)
            future_time = np.array([[current_time + steps_ahead]])
            
            # Predict with uncertainty
            mean, std = self.gp.predict(future_time, return_std=True)
            
            return mean[0], std[0]
        except:
            return None, None
    
    def get_pose(self):
        """Get current pose estimate"""
        return self.current_pose[:3], self.current_pose[3:7]
    
    def get_covariance(self):
        """Get uncertainty covariance"""
        pos_cov = self.uncertainty[:3, :3]
        quat_cov = self.uncertainty[3:7, 3:7]
        return pos_cov, quat_cov

# Helper functions (reuse from previous implementations)
def load_ply_mesh(filepath):
    """Load PLY mesh file"""
    if not os.path.exists(filepath):
        return None, None
    
    try:
        vertices = []
        faces = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
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
        
        for i in range(header_end, header_end + n_vertices):
            parts = lines[i].split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        for i in range(header_end + n_vertices, header_end + n_vertices + n_faces):
            parts = lines[i].split()
            n_verts = int(parts[0])
            face = [int(parts[j+1]) for j in range(n_verts)]
            faces.append(face)
        
        return np.array(vertices), faces
    except Exception as e:
        print(f"Error loading PLY: {e}")
        return None, None

def create_default_mesh(shape='bunny'):
    """Create default mesh"""
    theta = np.linspace(0, 2*np.pi, 20)
    phi = np.linspace(0, np.pi, 15)
    
    vertices = []
    for p in phi:
        for t in theta:
            x = 0.15 * np.sin(p) * np.cos(t)
            y = 0.12 * np.sin(p) * np.sin(t)
            z = 0.18 * np.cos(p)
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
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

def transform_mesh(vertices, position, quaternion):
    """Transform mesh vertices"""
    rot = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rot_matrix = rot.as_matrix()
    transformed = (rot_matrix @ vertices.T).T + position
    return transformed

def generate_ground_truth_trajectory(n_frames=100):
    """Generate circular trajectory"""
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
    """Check if object is occluded"""
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
    """Generate observations with occlusion"""
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

def visualize_results(camera_positions, observations, estimated_poses, 
                     estimated_covs, object_pos, object_quat, occluder_bounds, 
                     occlusion_factors, mesh_vertices, mesh_faces):
    """Interactive visualization"""
    n_frames = len(camera_positions)
    
    fig = plt.figure(figsize=(18, 10))
    
    ax_3d = fig.add_subplot(131, projection='3d')
    ax_behavioral = fig.add_subplot(132)
    ax_2d = fig.add_subplot(233)
    ax_info = fig.add_subplot(236)
    
    plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.3)
    
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, n_frames-1, valinit=0, valstep=1)
    
    def update(frame):
        frame = int(frame)
        
        ax_3d.clear()
        ax_behavioral.clear()
        ax_2d.clear()
        ax_info.clear()
        
        # --- 3D Scene ---
        if mesh_vertices is not None and mesh_faces is not None:
            # True mesh
            true_transformed = transform_mesh(mesh_vertices, object_pos, object_quat)
            poly_collection = []
            for face in mesh_faces:
                face_verts = [true_transformed[i] for i in face]
                poly_collection.append(face_verts)
            poly3d = Poly3DCollection(poly_collection, alpha=0.7, 
                                     facecolor='blue', edgecolor='darkblue', linewidths=0.5)
            ax_3d.add_collection3d(poly3d)
            
            # Estimated mesh
            est_pos, est_quat = estimated_poses[frame]
            est_transformed = transform_mesh(mesh_vertices, est_pos, est_quat)
            poly_collection_est = []
            for face in mesh_faces:
                face_verts = [est_transformed[i] for i in face]
                poly_collection_est.append(face_verts)
            poly3d_est = Poly3DCollection(poly_collection_est, alpha=0.3, 
                                         facecolor='orange', edgecolor='darkorange', linewidths=0.5)
            ax_3d.add_collection3d(poly3d_est)
        
        # Occluder
        x_range = occluder_bounds['x_range']
        y_range = occluder_bounds['y_range']
        z_range = occluder_bounds['z_range']
        
        vertices = [
            [x_range[0], y_range[0], z_range[0]], [x_range[1], y_range[0], z_range[0]],
            [x_range[1], y_range[1], z_range[0]], [x_range[0], y_range[1], z_range[0]],
            [x_range[0], y_range[0], z_range[1]], [x_range[1], y_range[0], z_range[1]],
            [x_range[1], y_range[1], z_range[1]], [x_range[0], y_range[1], z_range[1]]
        ]
        faces_occ = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[7], vertices[6], vertices[2], vertices[3]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]
        ax_3d.add_collection3d(Poly3DCollection(faces_occ, alpha=0.3, facecolor='red', 
                                                edgecolor='darkred'))
        
        cam_pos = camera_positions[frame]
        ax_3d.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]], 
                     c='green', s=100, marker='^', label='Camera')
        
        est_pos = estimated_poses[frame][0]
        ax_3d.scatter([est_pos[0]], [est_pos[1]], [est_pos[2]], 
                     c='orange', s=150, marker='*', label='Estimate', zorder=10)
        
        # Uncertainty ellipsoid
        pos_cov = estimated_covs[frame][0]
        eigenvalues, eigenvectors = np.linalg.eig(pos_cov)
        radii = np.sqrt(np.abs(eigenvalues)) * 3
        
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
        ax_3d.set_title('3D Scene - Willems DDC')
        ax_3d.legend()
        ax_3d.set_xlim([-4, 4])
        ax_3d.set_ylim([-4, 4])
        ax_3d.set_zlim([-1, 3])
        
        # --- Behavioral Representation ---
        ax_behavioral.clear()
        
        # Plot trajectory history
        if frame > 0:
            traj_pos = np.array([estimated_poses[i][0] for i in range(frame+1)])
            ax_behavioral.plot(traj_pos[:, 0], traj_pos[:, 1], 'b-', alpha=0.5, label='Trajectory')
            ax_behavioral.scatter(traj_pos[-1, 0], traj_pos[-1, 1], c='orange', s=100, 
                                marker='*', label='Current', zorder=10)
            
        # Show recent observations used for Willems' lemma
        if frame >= 10:
            recent_frames = max(0, frame-10)
            recent_pos = np.array([observations[i][:3] if observations[i] is not None 
                                  else estimated_poses[i][0] 
                                  for i in range(recent_frames, frame+1)])
            ax_behavioral.scatter(recent_pos[:, 0], recent_pos[:, 1], 
                                c='cyan', s=20, alpha=0.5, label='Behavioral Data')
        
        ax_behavioral.scatter(object_pos[0], object_pos[1], c='blue', s=200, 
                            marker='o', label='True', zorder=5)
        
        ax_behavioral.set_xlabel('X')
        ax_behavioral.set_ylabel('Y')
        ax_behavioral.set_title('Behavioral Trajectory Space')
        ax_behavioral.legend(fontsize=8)
        ax_behavioral.set_xlim([-1, 1])
        ax_behavioral.set_ylim([-1, 1])
        ax_behavioral.set_aspect('equal')
        ax_behavioral.grid(True, alpha=0.3)
        
        # --- 2D Top-Down ---
        if mesh_vertices is not None:
            true_transformed = transform_mesh(mesh_vertices, object_pos, object_quat)
            ax_2d.scatter(true_transformed[:, 0], true_transformed[:, 1], 
                         c='blue', alpha=0.3, s=5, label='True')
            
            est_pos, est_quat = estimated_poses[frame]
            est_transformed = transform_mesh(mesh_vertices, est_pos, est_quat)
            ax_2d.scatter(est_transformed[:, 0], est_transformed[:, 1], 
                         c='orange', alpha=0.2, s=5, label='Est.')
        
        rect = plt.Rectangle((x_range[0], y_range[0]), 
                            x_range[1] - x_range[0], y_range[1] - y_range[0],
                            color='red', alpha=0.3)
        ax_2d.add_patch(rect)
        
        ax_2d.scatter(cam_pos[0], cam_pos[1], c='green', s=100, marker='^', zorder=5)
        
        pos_cov_2d = pos_cov[:2, :2]
        eigenvalues_2d, eigenvectors_2d = np.linalg.eig(pos_cov_2d)
        angle = np.degrees(np.arctan2(eigenvectors_2d[1, 0], eigenvectors_2d[0, 0]))
        width, height = 2 * 3 * np.sqrt(np.abs(eigenvalues_2d))
        
        ellipse = mpatches.Ellipse(est_pos[:2], width, height, angle=angle,
                                   alpha=0.3, color='orange')
        ax_2d.add_patch(ellipse)
        
        ax_2d.set_xlabel('X')
        ax_2d.set_ylabel('Y')
        ax_2d.set_title('Top-Down View')
        ax_2d.legend(loc='upper right', fontsize=7)
        ax_2d.set_xlim([-4, 4])
        ax_2d.set_ylim([-4, 4])
        ax_2d.set_aspect('equal')
        ax_2d.grid(True, alpha=0.3)
        
        # --- Info Panel ---
        ax_info.axis('off')
        
        info_text = f"Frame: {frame}/{n_frames-1}\n\n"
        info_text += f"Method: Willems' DDC + GP\n\n"
        info_text += f"Occlusion: {'YES' if occlusion_factors[frame] > 1 else 'NO'}\n\n"
        
        info_text += f"True Position:\n"
        info_text += f"  [{object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}]\n\n"
        
        info_text += f"Estimated Position:\n"
        info_text += f"  [{est_pos[0]:.3f}, {est_pos[1]:.3f}, {est_pos[2]:.3f}]\n\n"
        
        error = np.linalg.norm(est_pos - object_pos)
        info_text += f"Position Error: {error:.4f} m\n\n"
        
        pos_std = np.sqrt(np.diag(pos_cov))
        info_text += f"Uncertainty (σ):\n"
        info_text += f"  [{pos_std[0]:.4f}, {pos_std[1]:.4f}, {pos_std[2]:.4f}]\n\n"
        
        info_text += f"Behavioral data:\n"
        info_text += f"  History: {min(frame, 200)} samples\n"
        info_text += f"  Lag: 10 steps\n"
        
        ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    update(0)
    plt.show()

def main():
    print("=" * 70)
    print("6D Pose Estimation using Willems' Fundamental Lemma")
    print("Data-Driven Control with Gaussian Process")
    print("=" * 70)
    print()
    
    n_frames = 100
    
    object_pos = np.array([0.0, 0.0, 0.5])
    object_quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    occluder_bounds = {
        'x_range': [-0.5, 0.5],
        'y_range': [0.8, 1.5],
        'z_range': [0.0, 1.5]
    }
    
    print("1. Loading/Creating mesh...")
    ply_path = '/mnt/user-data/uploads/object.ply'
    mesh_vertices, mesh_faces = load_ply_mesh(ply_path)
    
    if mesh_vertices is None:
        print("   Using default bunny mesh")
        mesh_vertices, mesh_faces = create_default_mesh('bunny')
    else:
        print(f"   Loaded: {len(mesh_vertices)} verts, {len(mesh_faces)} faces")
    
    print("2. Generating trajectory...")
    camera_positions, camera_orientations = generate_ground_truth_trajectory(n_frames)
    
    print("3. Generating observations...")
    failure_mode = 'none'
    observations, occlusion_factors = generate_observations(
        camera_positions, camera_orientations, object_pos, 
        occluder_bounds, failure_mode
    )
    
    print(f"   Occluded: {sum(1 for f in occlusion_factors if f > 1)} frames")
    
    print("4. Running Willems' DDC controller...")
    ddc = WillemsDDC(lag=10, horizon=5)
    
    # Initialize with first observation
    if observations[0] is not None:
        ddc.current_pose = observations[0].copy()
    
    estimated_poses = []
    estimated_covs = []
    
    for i in range(n_frames):
        # Control update (data-driven)
        ddc.control_update(observations[i], occlusion_factor=occlusion_factors[i])
        
        # Fit GP periodically for uncertainty modeling
        if i % 10 == 0 and i > 0:
            ddc.fit_gp_uncertainty()
        
        # Store results
        pos, quat = ddc.get_pose()
        pos_cov, quat_cov = ddc.get_covariance()
        estimated_poses.append((pos.copy(), quat.copy()))
        estimated_covs.append((pos_cov.copy(), quat_cov.copy()))
    
    print("5. Launching visualization...")
    print()
    print("Willems' Fundamental Lemma:")
    print("  - Uses behavioral data directly (no explicit model)")
    print("  - Any trajectory = linear combo of past trajectories")
    print("  - GP captures non-Gaussian uncertainty")
    print()
    
    visualize_results(camera_positions, observations, estimated_poses, 
                     estimated_covs, object_pos, object_quat, occluder_bounds, 
                     occlusion_factors, mesh_vertices, mesh_faces)
    
    print("Done!")

if __name__ == "__main__":
    main()
