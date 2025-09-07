# pip install open3d numpy
import time
import threading
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from math import erf, sqrt

# ================= CONFIG =================
START_DIST_Z = 10.0
MAIN_ARROW_LEN = 1.2

# Uncertainty (t=0)
POS_SIGMA0 = 0.4                 # pos std (units)
ORI_STD0_DEG = 12.0              # ori std (deg, half-angle-like)

# Diffusion (variance/sec) -> uncertainty grows with time if no measurements
POS_Q = 0.10                     # units^2 / s
ORI_Q = np.deg2rad(6.0)**2       # rad^2 / s

# Measurement noise (std); click button to update at SAME pose
MEAS_POS_STD = 0.10              # units
MEAS_ORI_STD_DEG = 4.0           # deg

# Orientation-pos coupling (worse ori far from mean)
ORI_POS_COUPLING = 1.0           # multiply θ by (1 + k * r / r_max)

# “Upright” tolerance in likelihood
UPRIGHT_EPS_DEG = 10.0

# Point cloud
N_POINTS_INIT = 100              # << default requested
N_POINTS_MIN  = 50
N_POINTS_MAX  = 40000
K_NEIGHBORS   = 12
R_MULT        = 3.0              # ball radius = R_MULT * sigma_pos(t)
UPDATE_HZ     = 5
POINT_SIZE    = 3.0

SHOW_WORLD_AXES = True
AXIS_SIZE = 0.6
RNG_SEED = 42
# ==========================================

def colormap_red_yellow_green(s: np.ndarray):
    s = np.clip(s, 0.0, 1.0)
    rgb = np.zeros((s.size, 3), dtype=np.float32)
    mask = s <= 0.5
    rgb[mask, 0] = 1.0
    rgb[mask, 1] = 2.0 * s[mask]
    inv = 1.0 - s[~mask]
    rgb[~mask, 0] = 2.0 * inv
    rgb[~mask, 1] = 1.0
    return rgb

def make_arrow_mesh(total_length, radius_scale=0.06):
    cyl_h = 0.80 * total_length
    cone_h = 0.20 * total_length
    cyl_r = radius_scale * total_length * 0.5
    cone_r = radius_scale * total_length
    m = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cyl_r, cone_radius=cone_r,
        cylinder_height=cyl_h, cone_height=cone_h,
        resolution=20, cylinder_split=1, cone_split=1)
    m.compute_vertex_normals()
    return m

def transform_matrix(scale: float, R: np.ndarray, t: np.ndarray):
    T = np.eye(4); T[:3, :3] = R * float(scale); T[:3, 3] = t; return T

def sample_spherical_weighted(n, sigma, r_max, rng):
    """N points in sphere with importance weights ~ exp(-0.5*(r/sigma)^2)."""
    over = int(max(n * 6, 2000))
    u = rng.normal(size=(over, 3)); u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    radii = r_max * rng.random(over) ** (1.0 / 3.0)
    pts = u * radii[:, None]
    w = np.exp(-0.5 * (radii / max(1e-9, sigma)) ** 2)
    p = w / (np.sum(w) or 1.0)
    idx = rng.choice(over, size=n, replace=False, p=p)
    return pts[idx], radii[idx]

class PoseBallApp:
    def __init__(self):
        self.rng = np.random.default_rng(RNG_SEED)
        self.t_mean = np.array([0.0, 0.0, START_DIST_Z], dtype=float)
        self.R_mean = np.eye(3)

        # dynamic uncertainty state (updated by diffusion and measurements)
        self.pos_sigma0 = POS_SIGMA0
        self.ori_std0_rad = np.deg2rad(ORI_STD0_DEG)
        self.t0 = time.perf_counter()

        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window("Pose Uncertainty Ball", 1280, 800)

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.05, 0.05, 0.07, 1.0])
        self.scene.scene.scene.set_sun_light([1,1,1], [1,1,1], 50000)
        self.scene.scene.scene.enable_sun_light(True)
        self.window.add_child(self.scene)

        # Right panel
        em = self.window.theme.font_size
        panel = gui.Vert(0.5 * em, gui.Margins(1*em, 1*em, 1*em, 1*em))
        panel.add_child(gui.Label("Point cloud size"))
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(N_POINTS_MIN, N_POINTS_MAX)
        self.slider.int_value = N_POINTS_INIT
        self.slider.set_on_value_changed(self._on_slider)
        panel.add_child(self.slider)

        btn = gui.Button("Take measurement (same pose)")
        btn.set_on_clicked(self._on_measurement)
        panel.add_child(btn)

        self.lbl_sigma = gui.Label("")
        self.lbl_theta = gui.Label("")
        panel.add_child(self.lbl_sigma)
        panel.add_child(self.lbl_theta)

        self.window.set_on_layout(lambda ctx: self._on_layout(ctx, panel))
        self.window.add_child(panel)

        # Materials
        self.mat_unlit = rendering.MaterialRecord(); self.mat_unlit.shader = "defaultUnlit"
        self.mat_points = rendering.MaterialRecord(); self.mat_points.shader = "defaultUnlit"; self.mat_points.point_size = POINT_SIZE

        # Static geoms
        if SHOW_WORLD_AXES:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE)
            self.scene.scene.add_geometry("world_axes", axes, self.mat_unlit)
        main_arrow = make_arrow_mesh(MAIN_ARROW_LEN); main_arrow.paint_uniform_color([0.1, 0.5, 1.0])  # BLUE
        self.scene.scene.add_geometry("main_arrow", main_arrow, self.mat_unlit)
        self.scene.scene.set_geometry_transform("main_arrow", transform_matrix(1.0, self.R_mean, self.t_mean))

        # Dynamic: point cloud
        self.n_points = N_POINTS_INIT
        self._rebuild_point_cloud()

        # Camera
        bbox = self.scene.scene.bounding_box
        self.scene.setup_camera(60.0, bbox, bbox.get_center())
        self.scene.look_at(self.t_mean, [0,0,-1], [0,-1,0])

        # Update loop (diffusion)
        self.running = True
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        gui.Application.instance.run()
        self.running = False

    # ---------- layout ----------
    def _on_layout(self, ctx, panel):
        r = self.window.content_rect
        panel_w = int(min(340, 0.28 * r.width))
        panel.frame = gui.Rect(r.get_right() - panel_w, r.y, panel_w, r.height)
        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)

    def _on_slider(self, v):
        self.n_points = int(v)
        gui.Application.instance.post_to_main_thread(self.window, self._rebuild_point_cloud)

    # ---------- uncertainty ----------
    def _current_sigmas(self):
        t = max(0.0, time.perf_counter() - self.t0)
        pos_sigma = sqrt(self.pos_sigma0**2 + POS_Q * t)
        ori_std = sqrt(self.ori_std0_rad**2 + ORI_Q * t)  # radians
        return pos_sigma, ori_std, t

    def _kalman_posterior_std(self, prior_std, meas_std):
        P, R = prior_std**2, meas_std**2
        if P <= 0 or R <= 0:
            return max(1e-9, min(prior_std, meas_std))
        return sqrt((P * R) / (P + R))

    def _on_measurement(self):
        # “Same pose”: means unchanged; shrink uncertainties via posterior and reset diffusion clock.
        pos_sigma_pred, ori_std_pred, _ = self._current_sigmas()
        pos_sigma_post = self._kalman_posterior_std(pos_sigma_pred, MEAS_POS_STD)
        ori_std_post = self._kalman_posterior_std(ori_std_pred, np.deg2rad(MEAS_ORI_STD_DEG))
        self.pos_sigma0 = pos_sigma_post
        self.ori_std0_rad = ori_std_post
        self.t0 = time.perf_counter()
        gui.Application.instance.post_to_main_thread(self.window, self._rebuild_point_cloud)

    # ---------- rebuild ----------
    def _rebuild_point_cloud(self):
        if self.scene.scene.has_geometry("ball"):
            self.scene.scene.remove_geometry("ball")

        sigma_pos, theta_rad, _ = self._current_sigmas()
        r_max = R_MULT * sigma_pos

        # Importance sample positions
        pts_local, radii = sample_spherical_weighted(self.n_points, sigma_pos, r_max, self.rng)
        pts_world = pts_local + self.t_mean[None, :]

        # Orientation likelihood per point (upright within epsilon), with coupling
        eps_rad = np.deg2rad(UPRIGHT_EPS_DEG)
        theta_eff = theta_rad * (1.0 + ORI_POS_COUPLING * (radii / max(1e-9, r_max)))
        p_upright = np.array([erf(eps_rad / (np.sqrt(2.0) * max(1e-9, th))) for th in theta_eff],
                             dtype=np.float32)

        # Neighbor-smoothed scores (average over K nearest, incl. self)
        pcd_tmp = o3d.geometry.PointCloud(); pcd_tmp.points = o3d.utility.Vector3dVector(pts_world)
        kdt = o3d.geometry.KDTreeFlann(pcd_tmp)
        scores = np.empty(self.n_points, dtype=np.float32)
        for i in range(self.n_points):
            _, idxs, _ = kdt.search_knn_vector_3d(pts_world[i], max(1, K_NEIGHBORS))
            scores[i] = float(np.mean(p_upright[idxs]))

        colors = colormap_red_yellow_green(scores)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        self.scene.scene.add_geometry("ball", pcd, self.mat_points)

        self.lbl_sigma.text = f"pos σ: {sigma_pos:.3f}  | r_max: {r_max:.3f} | N: {self.n_points}"
        self.lbl_theta.text = f"ori σ: {np.rad2deg(theta_rad):.2f}°   (ε={UPRIGHT_EPS_DEG:.1f}°, k={K_NEIGHBORS})"

    # ---------- animation loop ----------
    def _loop(self):
        period = 1.0 / UPDATE_HZ
        while self.running:
            time.sleep(period)
            gui.Application.instance.post_to_main_thread(self.window, self._rebuild_point_cloud)

if __name__ == "__main__":
    PoseBallApp()
