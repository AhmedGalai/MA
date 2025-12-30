# pip install open3d numpy
import numpy as np
import open3d as o3d
from pathlib import Path

# ---------- HARD-CODE THIS ----------
PLY_PATH = Path(r"C:\Users\Lenovo\Desktop\Ahmed\master\MA\sandbox\full_project_python\models\cube.ply")
TX, TY, TZ = 0.0, 0.0, 0.5              # translation
ROLL, PITCH, YAW = 0.0, 0.0, 45.0       # degrees (intrinsic X,Y,Z)
AXIS_SIZE = 0.25                        # COM axes length
WIRE_COLOR = (0.7, 0.7, 0.7)            # wireframe color
# ------------------------------------

def rotation_from_euler_xyz_deg(roll, pitch, yaw):
    rx, ry, rz = np.deg2rad([roll, pitch, yaw])
    return o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))

def center_of_mass_uniform(mesh: o3d.geometry.TriangleMesh):
    """Solid centroid (uniform density). Falls back to surface centroid if volume≈0."""
    if len(mesh.triangles) == 0:
        raise SystemExit("Mesh has no faces; cannot compute solid centroid.")
    m = o3d.geometry.TriangleMesh(mesh)
    m.orient_triangles()
    V = np.asarray(m.vertices, float)
    F = np.asarray(m.triangles, int)
    p0, p1, p2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    Vi = np.einsum("ij,ij->i", p0, np.cross(p1, p2)) / 6.0
    Vtot = Vi.sum()
    if abs(Vtot) < 1e-12:
        # surface (area) centroid
        Ai = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        Atot = Ai.sum()
        if Atot < 1e-12:
            raise SystemExit("Degenerate mesh (zero area/volume).")
        Ci = (p0 + p1 + p2) / 3.0
        return (Ai[:, None] * Ci).sum(axis=0) / Atot
    Ci_tet = (p0 + p1 + p2) / 4.0
    return (Vi[:, None] * Ci_tet).sum(axis=0) / Vtot

def main():
    mesh = o3d.io.read_triangle_mesh(str(PLY_PATH))
    if mesh.is_empty():
        raise SystemExit(f"Failed to load mesh: {PLY_PATH}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # COM in object frame (before pose)
    com_obj = center_of_mass_uniform(mesh)

    # Pose
    R = rotation_from_euler_xyz_deg(ROLL, PITCH, YAW)
    t = np.array([TX, TY, TZ], float)

    # Transform mesh to world
    mesh_world = o3d.geometry.TriangleMesh(mesh)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    mesh_world.transform(T)

    # COM in world
    com_world = R @ com_obj + t

    # ---- Wireframe (edges only), no faces ----
    if len(mesh_world.triangles) > 0:
        wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_world)
        wire.paint_uniform_color(WIRE_COLOR)
        to_draw = [wire]
    else:
        # Fallback: vertex-only display if model has no faces
        pcd = o3d.geometry.PointCloud(points=mesh_world.vertices)
        pcd.paint_uniform_color(WIRE_COLOR)
        to_draw = [pcd]

    # COM coordinate frame aligned with pose
    com_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE)
    T_axes = np.eye(4); T_axes[:3,:3] = R; T_axes[:3,3] = com_world
    com_axes.transform(T_axes)

    # Optional world axes
    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE*1.2)

    to_draw += [com_axes, world_axes]

    print(f"COM (object): {com_obj}")
    print(f"COM (world):  {com_world}")
    o3d.visualization.draw_geometries(to_draw,
        window_name="Wireframe + COM frame", width=1280, height=800)

if __name__ == "__main__":
    main()
