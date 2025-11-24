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

    Key idea:
      Any trajectory of a linear system can be represented as a
      linear combination of past trajectories (if data is persistently exciting).

    Here:
      - observation_history holds extended behavioral inputs:
          [object_pose_measurement, camera_pose_est, intrinsics, intrinsics_std, ...]
      - pose_history holds estimated object poses (7D: [x, y, z, qw, qx, qy, qz])
      - Willems' lemma provides a data-driven one-step predictor
      - GP models non-Gaussian, time-varying uncertainty on the trajectory
      - "occlusion_factor" is generalized to a measurement-trust factor:
        higher value => less trust in the current measurement
    """

    def __init__(self, lag=10, horizon=5):
        self.lag = lag            # History length (L)
        self.horizon = horizon    # Prediction horizon (T)

        # Data matrices for behavioral representation
        self.U_past = None  # Past inputs (behavioral observations)
        self.Y_past = None  # Past outputs (poses)

        # Historical data storage
        self.observation_history = []  # generic behavioral input
        self.pose_history = []         # 7D object pose estimates

        # Current state
        self.current_pose = np.zeros(7)  # [x, y, z, qw, qx, qy, qz]
        self.current_pose[3] = 1.0       # Identity quaternion

        # Gaussian Process for uncertainty over time
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        # Uncertainty tracking
        self.uncertainty = np.eye(7) * 0.1

        # Control gains
        self.K_feedback = 0.7     # Feedback gain on Willems predictor
        self.K_innovation = 0.3   # Innovation gain on measurement

    def add_trajectory_data(self, observation, pose):
        """
        Add behavioral observation-pose pair to historical data.

        observation: extended input vector (can include camera pose, intrinsics, etc.)
        pose:        7D object pose estimate
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
        Build Hankel matrices U and Y for behavioral representation.

        Willems' Lemma:
          If U is persistently exciting, any trajectory can be written as
          [u; y] = [U; Y] * g for some weights g.
        """
        if len(self.observation_history) < self.lag + self.horizon:
            return False

        n_data = len(self.observation_history)
        n_cols = n_data - self.lag - self.horizon + 1

        if n_cols < 1:
            return False

        U_past_list = []
        Y_past_list = []

        for i in range(n_cols):
            # Past behavioral inputs
            u_past = np.array(self.observation_history[i:i + self.lag]).flatten()
            U_past_list.append(u_past)

            # Past object pose outputs
            y_past = np.array(self.pose_history[i:i + self.lag]).flatten()
            Y_past_list.append(y_past)

        self.U_past = np.array(U_past_list).T
        self.Y_past = np.array(Y_past_list).T
        return True

    def predict_with_willems(self, recent_observations, occlusion_factor=1.0):
        """
        Predict next pose using Willems' Fundamental Lemma.

        Find g such that recent_observations ≈ U_past @ g
        Then predicted pose = Y_past @ g

        occlusion_factor is actually a general "measurement trust" factor:
        - larger => less trust in current measurement, more regularization.
        """
        if self.U_past is None or len(recent_observations) < self.lag:
            return self.current_pose, self.uncertainty

        # Form recent observation vector
        u_recent = np.array(recent_observations[-self.lag:]).flatten()

        try:
            # Regularized least squares (handles noise, uncertainty)
            lambda_reg = 0.01 * occlusion_factor
            A = self.U_past.T @ self.U_past + lambda_reg * np.eye(self.U_past.shape[1])
            b = self.U_past.T @ u_recent
            g = np.linalg.solve(A, b)

            # Predict pose using same weights
            y_predicted = self.Y_past @ g

            # Reshape and take most recent pose in the window
            n_states = 7
            y_predicted_poses = y_predicted.reshape(self.lag, n_states)
            predicted_pose = y_predicted_poses[-1]

            # Normalize quaternion
            predicted_pose[3:7] /= (np.linalg.norm(predicted_pose[3:7]) + 1e-8)

            # Residual for uncertainty estimate
            residual = np.linalg.norm(u_recent - self.U_past @ g)

            # Update uncertainty based on residual and occlusion_factor
            base_uncertainty = 0.01
            uncertainty_scale = base_uncertainty * (1 + residual) * occlusion_factor
            self.uncertainty = np.eye(7) * uncertainty_scale
            self.uncertainty[3:7, 3:7] *= 0.5  # somewhat lower for orientation

            return predicted_pose, self.uncertainty

        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            return self.current_pose, self.uncertainty * 2.0

    def control_update(self, measurement, observation, occlusion_factor=1.0):
        """
        Data-driven control update.

        measurement: 7D object pose measurement [x,y,z,qw,qx,qy,qz] or None
        observation: extended behavioral vector (measurement + camera pose est + intrinsics + ...)
                     used for Willems' Hankel representation
        occlusion_factor: general measurement-trust factor (>=1, higher = less trust)

        Steps:
          1. Update behavior dataset with observation + current pose.
          2. Predict using Willems' lemma.
          3. Compute innovation (measurement - prediction), if measurement exists.
          4. Blend prediction and measurement using feedback and innovation gains.
        """
        # Store behavioral data + pose for Willems
        if observation is not None:
            self.add_trajectory_data(observation, self.current_pose)

        # No measurement -> pure prediction / open-loop
        if measurement is None:
            if len(self.observation_history) >= self.lag:
                predicted, unc = self.predict_with_willems(
                    self.observation_history,
                    occlusion_factor
                )
                self.current_pose = predicted
                self.uncertainty = unc * occlusion_factor
            else:
                # Not enough data - maintain current pose and inflate uncertainty
                self.uncertainty *= 1.1
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

        # Handle quaternion sign ambiguity
        if np.dot(measurement[3:7], predicted[3:7]) < 0:
            innovation[3:7] = -innovation[3:7]

        # Control law: feedback + innovation (innovation gain lowered when occlusion_factor is high)
        innovation_gain = self.K_innovation / occlusion_factor
        feedback_gain = self.K_feedback

        self.current_pose = (
            feedback_gain * predicted +
            innovation_gain * measurement +
            (1 - feedback_gain - innovation_gain) * self.current_pose
        )

        # Normalize quaternion
        self.current_pose[3:7] /= (np.linalg.norm(self.current_pose[3:7]) + 1e-8)

        # Update uncertainty (lower with stronger innovation)
        self.uncertainty = unc / (1.0 + innovation_gain)

    def fit_gp_uncertainty(self):
        """Fit Gaussian Process to model uncertainty over time on the trajectory."""
        if len(self.pose_history) < 10:
            return

        recent_poses = np.array(self.pose_history[-50:])
        time_steps = np.arange(len(recent_poses)).reshape(-1, 1)

        try:
            # Fit GP to position trajectory
            self.gp.fit(time_steps, recent_poses[:, :3])
        except Exception:
            pass

    def get_gp_prediction(self, steps_ahead=1):
        """Use GP to predict future position (toy use)."""
        if len(self.pose_history) < 10:
            return None, None

        try:
            current_time = len(self.pose_history)
            future_time = np.array([[current_time + steps_ahead]])
            mean, std = self.gp.predict(future_time, return_std=True)
            return mean[0], std[0]
        except Exception:
            return None, None

    def get_pose(self):
        """Get current pose estimate."""
        return self.current_pose[:3], self.current_pose[3:7]

    def get_covariance(self):
        """Get uncertainty covariance."""
        pos_cov = self.uncertainty[:3, :3]
        quat_cov = self.uncertainty[3:7, 3:7]
        return pos_cov, quat_cov


# ----------------- Helper functions -----------------

def load_ply_mesh(filepath):
    """Load PLY mesh file (very simple ASCII PLY parser)."""
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

        for i in range(header_end + n_vertices,
                       header_end + n_vertices + n_faces):
            parts = lines[i].split()
            n_verts = int(parts[0])
            face = [int(parts[j + 1]) for j in range(n_verts)]
            faces.append(face)

        return np.array(vertices), faces
    except Exception as e:
        print(f"Error loading PLY: {e}")
        return None, None


def create_default_mesh(shape='bunny'):
    """Create a default ellipsoid-like mesh (placeholder for an object)."""
    theta = np.linspace(0, 2 * np.pi, 20)
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
    """Apply pose (position + quaternion [w,x,y,z]) to mesh vertices."""
    rot = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rot_matrix = rot.as_matrix()
    transformed = (rot_matrix @ vertices.T).T + position
    return transformed


def generate_ground_truth_trajectory(n_frames=100):
    """Generate circular camera trajectory around origin."""
    t = np.linspace(0, 2 * np.pi, n_frames)
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
        quat = R.from_matrix(rot_matrix).as_quat()  # x,y,z,w
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        camera_orientations.append(quat_wxyz)

    return camera_positions, np.array(camera_orientations)


def generate_object_trajectory(n_frames=100):
    """Object moves independently from the camera."""
    t = np.linspace(0, 2 * np.pi, n_frames)

    x = 0.4 * np.cos(2 * t)
    y = 0.3 * np.sin(3 * t)
    z = 0.5 + 0.1 * np.sin(t)

    object_positions = np.column_stack([x, y, z])

    yaw = 0.5 * t  # simple yaw over time
    rot = R.from_euler('z', yaw)
    quats_xyzw = rot.as_quat()
    object_orientations = np.column_stack([
        quats_xyzw[:, 3],
        quats_xyzw[:, 0],
        quats_xyzw[:, 1],
        quats_xyzw[:, 2],
    ])

    return object_positions, object_orientations


def generate_drifty_camera_poses(camera_positions, camera_orientations,
                                 pos_drift_sigma=0.002, rot_drift_deg=0.05):
    """
    Simulate a SLAM-like camera estimator with drift.
    Returns:
      cam_est_positions, cam_est_orientations, cam_covs (3x3 per frame)
    """
    n = len(camera_positions)
    cam_est_positions = np.zeros_like(camera_positions)
    cam_est_orientations = np.zeros_like(camera_orientations)
    cam_covs = np.zeros((n, 3, 3))

    drift_pos = np.zeros(3)
    drift_rot = R.identity()

    for i in range(n):
        # random walk in position
        drift_pos += np.random.randn(3) * pos_drift_sigma

        # small random walk in rotation
        axis = np.random.randn(3)
        axis /= (np.linalg.norm(axis) + 1e-8)
        angle = np.deg2rad(rot_drift_deg) * np.random.randn()
        delta_R = R.from_rotvec(axis * angle)
        drift_rot = delta_R * drift_rot

        true_R = R.from_quat([
            camera_orientations[i, 1],
            camera_orientations[i, 2],
            camera_orientations[i, 3],
            camera_orientations[i, 0]
        ])

        est_R = drift_rot * true_R
        est_quat_xyzw = est_R.as_quat()
        est_quat_wxyz = np.array([
            est_quat_xyzw[3],
            est_quat_xyzw[0],
            est_quat_xyzw[1],
            est_quat_xyzw[2]
        ])

        cam_est_positions[i] = camera_positions[i] + drift_pos
        cam_est_orientations[i] = est_quat_wxyz

        # Simple covariance: proportional to squared position drift
        pos_err = drift_pos
        sigma2 = (np.linalg.norm(pos_err) + 1e-3) ** 2
        cam_covs[i] = np.eye(3) * sigma2

    return cam_est_positions, cam_est_orientations, cam_covs


def generate_intrinsics_and_covariance(n_frames=100,
                                       fx_nominal=800.0,
                                       fy_nominal=800.0,
                                       cx_nominal=320.0,
                                       cy_nominal=240.0):
    """
    Simulate slowly drifting intrinsics with uncertainty.
    Returns:
      intrinsics:      (n,4)  [fx, fy, cx, cy]
      intrinsics_covs: (n,4,4)
    """
    intrinsics = np.zeros((n_frames, 4))
    intrinsics_covs = np.zeros((n_frames, 4, 4))

    # Start with nominal values
    current = np.array([fx_nominal, fy_nominal, cx_nominal, cy_nominal], dtype=float)

    for i in range(n_frames):
        # Slow random walk on intrinsics
        current += np.random.randn(4) * np.array([0.05, 0.05, 0.01, 0.01])

        intrinsics[i] = current

        # Uncertainty grows slowly with time
        base_sigma = np.array([1.0, 1.0, 0.5, 0.5])
        growth = 0.01 * i
        sigma = base_sigma * (1.0 + growth)
        intrinsics_covs[i] = np.diag(sigma ** 2)

    return intrinsics, intrinsics_covs


def check_occlusion(camera_pos, object_pos, occluder_bounds):
    """Check if object is occluded by a box between camera and object."""
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


def compute_trust_factor(is_occluded, Sigma_cam, Sigma_K):
    """
    Turn occlusion + camera and intrinsics covariance into a single factor >= 1.
    Larger => less trust in current measurement.
    """
    base = 1.0

    if is_occluded:
        base += 5.0  # strong hit for occlusion

    # Camera tracking uncertainty
    cam_term = 0.5 * np.trace(Sigma_cam)

    # Intrinsics uncertainty
    intr_term = 0.5 * np.trace(Sigma_K)

    factor = base * (1.0 + cam_term + intr_term)
    return max(factor, 1.0)


def generate_observations(camera_positions, camera_orientations,
                          cam_est_positions, cam_est_orientations, cam_covs,
                          intrinsics, intrinsics_covs,
                          object_positions, object_orientations,
                          occluder_bounds, failure_mode='none'):
    """
    Generate per-frame:
      - meas_poses:   7D object pose measurements (can be None)
      - obs_vectors:  extended behavioral inputs for Willems (may be None)
      - trust_factors: measurement-trust factor (>=1)
      - occlusion_flags: True if object is geometrically occluded

    Extended observation vector (when available):
      [object_pose_meas(7),
       cam_est_pos(3),
       cam_est_quat(4),
       intrinsics(4),
       intrinsics_std(4)]
      => total 22D
    """
    n_frames = len(camera_positions)
    meas_poses = []
    obs_vectors = []
    trust_factors = []
    occlusion_flags = []

    prev_meas = None

    for i in range(n_frames):
        obj_pos_i = object_positions[i]
        obj_quat_i = object_orientations[i]

        is_occluded = check_occlusion(camera_positions[i], obj_pos_i, occluder_bounds)
        occlusion_flags.append(is_occluded)

        # Intrinsic + cam uncertainty => trust factor
        Sigma_cam = cam_covs[i]
        Sigma_K = intrinsics_covs[i]
        trust_factor = compute_trust_factor(is_occluded, Sigma_cam, Sigma_K)
        trust_factors.append(trust_factor)

        # Generate object measurement
        if is_occluded:
            if failure_mode == 'null':
                meas = None
            elif failure_mode == 'previous':
                meas = prev_meas.copy() if prev_meas is not None else None
            elif failure_mode == 'random':
                # Completely spurious measurement
                meas = np.concatenate([
                    np.random.randn(3) * 2.0,
                    R.random().as_quat()[[3, 0, 1, 2]]
                ])
            else:
                # "noisy but biased" occlusion behavior
                meas = np.concatenate([
                    obj_pos_i + np.random.randn(3) * 0.3,
                    obj_quat_i + np.random.randn(4) * 0.2
                ])
                meas[3:7] /= np.linalg.norm(meas[3:7])
        else:
            # Good measurement
            meas = np.concatenate([
                obj_pos_i + np.random.randn(3) * 0.02,
                obj_quat_i + np.random.randn(4) * 0.01
            ])
            meas[3:7] /= np.linalg.norm(meas[3:7])
            prev_meas = meas

        # Build extended behavioral observation vector
        if meas is not None:
            intr = intrinsics[i]
            SigmaK = intrinsics_covs[i]
            intr_std = np.sqrt(np.diag(SigmaK))

            obs = np.concatenate([
                meas,
                cam_est_positions[i],
                cam_est_orientations[i],
                intr,
                intr_std
            ])
        else:
            obs = None

        meas_poses.append(meas)
        obs_vectors.append(obs)

    return meas_poses, obs_vectors, trust_factors, occlusion_flags


def visualize_results(camera_positions, cam_est_positions,
                      observations, estimated_poses, estimated_covs,
                      object_positions, object_orientations,
                      occluder_bounds, trust_factors, occlusion_flags,
                      mesh_vertices, mesh_faces):
    """Interactive visualization with slider."""
    n_frames = len(camera_positions)

    fig = plt.figure(figsize=(18, 10))

    ax_3d = fig.add_subplot(131, projection='3d')
    ax_behavioral = fig.add_subplot(132)
    ax_2d = fig.add_subplot(233)
    ax_info = fig.add_subplot(236)

    plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.3)

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=0, valstep=1)

    def update(frame):
        frame = int(frame)

        ax_3d.clear()
        ax_behavioral.clear()
        ax_2d.clear()
        ax_info.clear()

        obj_pos = object_positions[frame]
        obj_quat = object_orientations[frame]

        # --- 3D Scene ---
        if mesh_vertices is not None and mesh_faces is not None:
            # True mesh
            true_transformed = transform_mesh(mesh_vertices, obj_pos, obj_quat)
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
        ax_3d.add_collection3d(
            Poly3DCollection(faces_occ, alpha=0.3, facecolor='red', edgecolor='darkred')
        )

        cam_pos = camera_positions[frame]
        cam_est_pos = cam_est_positions[frame]
        ax_3d.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
                      c='green', s=80, marker='^', label='Camera (true)')
        ax_3d.scatter([cam_est_pos[0]], [cam_est_pos[1]], [cam_est_pos[2]],
                      c='black', s=40, marker='x', label='Camera (est)')

        est_pos = estimated_poses[frame][0]
        ax_3d.scatter([est_pos[0]], [est_pos[1]], [est_pos[2]],
                      c='orange', s=150, marker='*', label='Estimate', zorder=10)

        # Uncertainty ellipsoid (3D position)
        pos_cov = estimated_covs[frame][0]
        eigenvalues, eigenvectors = np.linalg.eig(pos_cov)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        radii = np.sqrt(eigenvalues) * 3

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x_ell = radii[0] * np.cos(u) * np.sin(v)
        y_ell = radii[1] * np.sin(u) * np.sin(v)
        z_ell = radii[2] * np.cos(v)

        for i in range(x_ell.shape[0]):
            for j in range(x_ell.shape[1]):
                point = np.array([x_ell[i, j], y_ell[i, j], z_ell[i, j]])
                rotated = eigenvectors @ point
                x_ell[i, j] = rotated[0] + est_pos[0]
                y_ell[i, j] = rotated[1] + est_pos[1]
                z_ell[i, j] = rotated[2] + est_pos[2]

        ax_3d.plot_surface(x_ell, y_ell, z_ell, alpha=0.15, color='orange')

        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('3D Scene - Willems DDC')
        ax_3d.legend()
        ax_3d.set_xlim([-4, 4])
        ax_3d.set_ylim([-4, 4])
        ax_3d.set_zlim([-1, 3])

        # --- Behavioral Representation (XY trajectory) ---
        if frame > 0:
            traj_pos = np.array([estimated_poses[i][0] for i in range(frame + 1)])
            ax_behavioral.plot(traj_pos[:, 0], traj_pos[:, 1], 'b-', alpha=0.5,
                               label='Estimated Trajectory')
            ax_behavioral.scatter(traj_pos[-1, 0], traj_pos[-1, 1], c='orange', s=100,
                                  marker='*', label='Current', zorder=10)

        # show recent measurements used for Willems' lemma
        if frame >= 10:
            recent_start = max(0, frame - 10)
            recent_pos = []
            for i in range(recent_start, frame + 1):
                if observations[i] is not None:
                    # the first 3 entries of extended obs are object position
                    recent_pos.append(observations[i][:3])
                else:
                    recent_pos.append(estimated_poses[i][0])
            recent_pos = np.array(recent_pos)
            ax_behavioral.scatter(recent_pos[:, 0], recent_pos[:, 1],
                                  c='cyan', s=20, alpha=0.5, label='Behavioral Data')

        ax_behavioral.scatter(obj_pos[0], obj_pos[1],
                              c='blue', s=200, marker='o', label='True Object', zorder=5)

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
            true_transformed = transform_mesh(mesh_vertices, obj_pos, obj_quat)
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

        ax_2d.scatter(cam_pos[0], cam_pos[1], c='green', s=80, marker='^', zorder=5,
                      label='Camera (true)')
        ax_2d.scatter(cam_est_pos[0], cam_est_pos[1], c='black', s=40, marker='x',
                      zorder=5, label='Camera (est)')

        pos_cov_2d = pos_cov[:2, :2]
        eigenvalues_2d, eigenvectors_2d = np.linalg.eig(pos_cov_2d)
        eigenvalues_2d = np.maximum(eigenvalues_2d, 1e-8)
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

        error = np.linalg.norm(est_pos - obj_pos)
        pos_std = np.sqrt(np.diag(pos_cov))
        trust = trust_factors[frame]
        occ = occlusion_flags[frame]

        info_text = f"Frame: {frame}/{n_frames - 1}\n\n"
        info_text += "Method: Willems' DDC + GP\n\n"
        info_text += f"Occlusion (geom): {'YES' if occ else 'NO'}\n"
        info_text += f"Measurement trust factor: {trust:.3f}\n\n"

        info_text += "True Position:\n"
        info_text += f"  [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]\n\n"

        info_text += "Estimated Position:\n"
        info_text += f"  [{est_pos[0]:.3f}, {est_pos[1]:.3f}, {est_pos[2]:.3f}]\n\n"

        info_text += f"Position Error: {error:.4f} m\n\n"

        info_text += "Uncertainty (sigma):\n"
        info_text += f"  [{pos_std[0]:.4f}, {pos_std[1]:.4f}, {pos_std[2]:.4f}]\n\n"

        info_text += "Behavioral data:\n"
        info_text += f"  History length: {min(frame, 200)} samples\n"
        info_text += f"  Lag: {10} steps\n"

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
    print("Data-Driven Control with Camera Pose + Intrinsics Uncertainty")
    print("=" * 70)
    print()

    n_frames = 100

    # Occluder configuration
    occluder_bounds = {
        'x_range': [-0.5, 0.5],
        'y_range': [0.8, 1.5],
        'z_range': [0.0, 1.5]
    }

    print("1. Loading/Creating mesh...")
    ply_path = '/mnt/user-data/uploads/object.ply'
    mesh_vertices, mesh_faces = load_ply_mesh(ply_path)

    if mesh_vertices is None:
        print("   Using default bunny-like mesh")
        mesh_vertices, mesh_faces = create_default_mesh('bunny')
    else:
        print(f"   Loaded: {len(mesh_vertices)} verts, {len(mesh_faces)} faces")

    print("2. Generating trajectories (camera + object)...")
    camera_positions, camera_orientations = generate_ground_truth_trajectory(n_frames)
    object_positions, object_orientations = generate_object_trajectory(n_frames)

    print("3. Generating drifty camera tracking estimates...")
    cam_est_positions, cam_est_orientations, cam_covs = generate_drifty_camera_poses(
        camera_positions, camera_orientations
    )

    print("4. Generating intrinsics + covariance...")
    intrinsics, intrinsics_covs = generate_intrinsics_and_covariance(n_frames)

    print("5. Generating observations...")
    failure_mode = 'none'  # 'null', 'previous', 'random', etc.
    meas_poses, obs_vectors, trust_factors, occlusion_flags = generate_observations(
        camera_positions, camera_orientations,
        cam_est_positions, cam_est_orientations, cam_covs,
        intrinsics, intrinsics_covs,
        object_positions, object_orientations,
        occluder_bounds, failure_mode
    )

    print(f"   Geometrically occluded frames: {sum(occlusion_flags)}")

    print("6. Running Willems' DDC controller...")
    ddc = WillemsDDC(lag=10, horizon=5)

    # Initialize with first measurement if available
    if meas_poses[0] is not None:
        ddc.current_pose = meas_poses[0].copy()

    estimated_poses = []
    estimated_covs = []

    for i in range(n_frames):
        ddc.control_update(
            measurement=meas_poses[i],
            observation=obs_vectors[i],
            occlusion_factor=trust_factors[i]
        )

        # Fit GP periodically for uncertainty modeling
        if i % 10 == 0 and i > 0:
            ddc.fit_gp_uncertainty()

        pos, quat = ddc.get_pose()
        pos_cov, quat_cov = ddc.get_covariance()
        estimated_poses.append((pos.copy(), quat.copy()))
        estimated_covs.append((pos_cov.copy(), quat_cov.copy()))

    print("7. Launching visualization...")
    print()
    print("Willems' Fundamental Lemma:")
    print("  - Trajectories represented directly from data (no explicit model).")
    print("  - Extended behavioral inputs include camera pose & intrinsics.")
    print("  - Trust factor modulates regularization and innovation.")
    print()

    visualize_results(camera_positions, cam_est_positions,
                      obs_vectors, estimated_poses, estimated_covs,
                      object_positions, object_orientations,
                      occluder_bounds, trust_factors, occlusion_flags,
                      mesh_vertices, mesh_faces)

    print("Done!")


if __name__ == "__main__":
    main()

