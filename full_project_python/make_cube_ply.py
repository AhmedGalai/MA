# Writes an ASCII PLY for a unit cube centered at the origin.
# 8 vertices, 12 triangle faces (2 per cube face).
# Usage: python make_cube_ply.py cube.ply

import sys

def cube_ply_text(size=1.0):
    s = size / 2.0
    verts = [
        (-s,-s,-s), ( s,-s,-s), ( s, s,-s), (-s, s,-s),  # z = -s
        (-s,-s, s), ( s,-s, s), ( s, s, s), (-s, s, s),  # z = +s
    ]
    # Triangles (counter-clockwise)
    faces = [
        # bottom (z=-s)
        (0, 1, 2), (0, 2, 3),
        # top (z=+s)
        (4, 6, 5), (4, 7, 6),
        # front (y=-s)
        (0, 5, 1), (0, 4, 5),
        # back (y=+s)
        (3, 2, 6), (3, 6, 7),
        # left (x=-s)
        (0, 3, 7), (0, 7, 4),
        # right (x=+s)
        (1, 5, 6), (1, 6, 2),
    ]

    lines = []
    lines.append("ply")
    lines.append("format ascii 1.0")
    lines.append(f"element vertex {len(verts)}")
    lines.append("property float x")
    lines.append("property float y")
    lines.append("property float z")
    lines.append(f"element face {len(faces)}")
    lines.append("property list uchar int vertex_indices")
    lines.append("end_header")
    for v in verts:
        lines.append(f"{v[0]} {v[1]} {v[2]}")
    for f in faces:
        lines.append(f"3 {f[0]} {f[1]} {f[2]}")
    return "\n".join(lines) + "\n"

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "cube.ply"
    with open(out, "w", encoding="utf-8") as f:
        f.write(cube_ply_text(size=1.0))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()

