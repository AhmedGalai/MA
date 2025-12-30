

# pip install open3d numpy
import time
import numpy as np
import open3d as o3d
from pathlib import Path

# ----------------- hardcoded inputs -----------------
PLY_PATH = Path(r"C:\Users\Lenovo\Desktop\Ahmed\master\MA\sandbox\full_project_python\models\cube.ply")  # <- change this
START_DIST_Z = 10.0    # "far from the camera" along +Z
AXIS_SIZE = 0.5
YAW_SPEED_DPS = 25.0    # deg/sec around Z
PITCH_SPEED_DPS = 60.0  # deg/sec around Y
# -----------------------------------------

def Rz(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[ c,-s, 0.0],
                     [ s, c, 0.0],
                     [0.0,0.0, 1.0]])

def Ry(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[ c, 0.0, s],
                     [0.0, 1.0,0.0],
                     [-s, 0.0, c]])

# Load mesh
mesh = o3d.io.read_triangle_mesh(str(PLY_PATH))
if mesh.is_empty():
    raise SystemExit(f"Failed to load mesh: {PLY_PATH}")
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# Precompute for fast pose updates
V0 = np.asarray(mesh.vertices).astype(np.float64)
C  = mesh.get_center()
V0c = V0 - C
N0 = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None

# Scale arrow relative to mesh size
extent = np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound())
arrow_len = max(1e-3, 0.6 * extent)
cyl_h = 0.8 * arrow_len
cone_h = 0.2 * arrow_len
cyl_r = 0.03 * extent
cone_r = 0.06 * extent

# Create a pose arrow that points along +Z in local frame (base at origin)
arrow = o3d.geometry.TriangleMesh.create_arrow(
    cylinder_radius=cyl_r,
    cone_radius=cone_r,
    cylinder_height=cyl_h,
    cone_height=cone_h,
    resolution=20, cylinder_split=1, cone_split=1
)
arrow.compute_vertex_normals()
A0 = np.asarray(arrow.vertices).astype(np.float64)
NA0 = np.asarray(arrow.vertex_normals)

# Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="6D Pose + Arrow (two-axis rates)")
mesh_draw = o3d.geometry.TriangleMesh(mesh)  # copy
arrow_draw = o3d.geometry.TriangleMesh(arrow)
world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE)

vis.add_geometry(mesh_draw)
vis.add_geometry(arrow_draw)
vis.add_geometry(world_axes)

# Initial translation (pose position)
t = np.array([0.0, 0.0, START_DIST_Z])
yaw = 0.0
pitch = 0.0

# Camera
ctr = vis.get_view_control()
ctr.set_lookat(C + t)
ctr.set_front([0.0, 0.0, -1.0])
ctr.set_up([0.0, -1.0, 0.0])
ctr.set_zoom(0.6)

last = time.perf_counter()
try:
    while True:
        now = time.perf_counter()
        dt = now - last
        last = now

        # Pose rotation: pitch (Y) and yaw (Z), different speeds
        yaw   += np.deg2rad(YAW_SPEED_DPS)   * dt
        pitch += np.deg2rad(PITCH_SPEED_DPS) * dt
        R = Rz(yaw) @ Ry(pitch)

        # Update mesh vertices/normals (pose origin at C + t)
        V = (V0c @ R.T) + C + t
        mesh_draw.vertices = o3d.utility.Vector3dVector(V)
        if N0 is not None:
            mesh_draw.vertex_normals = o3d.utility.Vector3dVector(N0 @ R.T)

        # Update arrow: base at pose origin, pointing along pose +Z
        A = (A0 @ R.T) + (C + t)
        NA = NA0 @ R.T
        arrow_draw.vertices = o3d.utility.Vector3dVector(A)
        arrow_draw.vertex_normals = o3d.utility.Vector3dVector(NA)

        vis.update_geometry(mesh_draw)
        vis.update_geometry(arrow_draw)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1/60)  # ~60 FPS cap
except KeyboardInterrupt:
    pass
finally:
    vis.destroy_window()