import open3d as o3d
m = o3d.io.read_triangle_mesh("Banana.stl")
o3d.io.write_triangle_mesh("Banana.ply", m)
m = o3d.io.read_triangle_mesh("Football.stl")
o3d.io.write_triangle_mesh("Football.ply", m)
