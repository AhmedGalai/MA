#!/usr/bin/env python3
"""
Write ASCII PLY meshes.

Examples:
  # rectangle (box) of size 2 x 1 x 0.5
  python make_shape_ply.py rectangle box.ply --sx 2 --sy 1 --sz 0.5

  # cylinder of radius 1, height 2, with 48 segments
  python make_shape_ply.py cylinder cyl.ply --radius 1 --height 2 --segments 48

  # sphere (ball) of radius 1 with 32 rings x 64 sectors
  python make_shape_ply.py ball ball.ply --radius 1 --rings 32 --sectors 64
"""

import sys
import math
import argparse

# ----------------- Shared -----------------
def ply_text_from_verts_faces(verts, faces):
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
    for x, y, z in verts:
        lines.append(f"{x} {y} {z}")
    for a, b, c in faces:
        lines.append(f"3 {a} {b} {c}")
    return "\n".join(lines) + "\n"

# ----------------- Rectangle (box) -----------------
def rectangle_ply_text(sx=1.0, sy=1.0, sz=1.0):
    hx, hy, hz = sx/2.0, sy/2.0, sz/2.0
    v = [
        (-hx, -hy, -hz), ( hx, -hy, -hz), ( hx,  hy, -hz), (-hx,  hy, -hz),  # z = -hz
        (-hx, -hy,  hz), ( hx, -hy,  hz), ( hx,  hy,  hz), (-hx,  hy,  hz),  # z = +hz
    ]
    f = [
        # bottom (z=-hz)
        (0, 1, 2), (0, 2, 3),
        # top (z=+hz)
        (4, 6, 5), (4, 7, 6),
        # front (y=-hy)
        (0, 5, 1), (0, 4, 5),
        # back (y=+hy)
        (3, 2, 6), (3, 6, 7),
        # left (x=-hx)
        (0, 3, 7), (0, 7, 4),
        # right (x=+hx)
        (1, 5, 6), (1, 6, 2),
    ]
    return ply_text_from_verts_faces(v, f)

# ----------------- Cylinder -----------------
def cylinder_ply_text(radius=1.0, height=2.0, segments=48):
    n = max(3, int(segments))
    r = float(radius)
    h = float(height)
    hz = h / 2.0

    verts = []
    # Rings: bottom (z=-hz), top (z=+hz)
    for z in (-hz, +hz):
        for i in range(n):
            ang = 2.0 * math.pi * i / n
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            verts.append((x, y, z))
    # centers
    idx_c_bot = len(verts); verts.append((0.0, 0.0, -hz))
    idx_c_top = len(verts); verts.append((0.0, 0.0, +hz))

    faces = []
    # Sides (two tris per segment)
    for i in range(n):
        i2 = (i + 1) % n
        b0 = i
        b1 = i2
        t0 = n + i
        t1 = n + i2
        # outward CCW
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))

    # Bottom cap (outward normal -Z): use reversed order around ring
    for i in range(n):
        i2 = (i + 1) % n
        b0 = i
        b1 = i2
        faces.append((idx_c_bot, b1, b0))

    # Top cap (outward normal +Z)
    for i in range(n):
        i2 = (i + 1) % n
        t0 = n + i
        t1 = n + i2
        faces.append((idx_c_top, t0, t1))

    return ply_text_from_verts_faces(verts, faces)

# ----------------- Sphere (UV) -----------------
def sphere_ply_text(radius=1.0, rings=32, sectors=64):
    R = float(radius)
    stacks = max(2, int(rings))      # latitudes (including poles)
    slices = max(3, int(sectors))    # longitudes

    verts = []
    # Latitude from -pi/2 to +pi/2
    for i in range(stacks + 1):
        phi = -0.5 * math.pi + (i / stacks) * math.pi
        z = R * math.sin(phi)
        r_xy = R * math.cos(phi)
        for j in range(slices):
            theta = 2.0 * math.pi * (j / slices)
            x = r_xy * math.cos(theta)
            y = r_xy * math.sin(theta)
            verts.append((x, y, z))

    def vid(i, j):
        return i * slices + (j % slices)

    faces = []
    for i in range(stacks):
        for j in range(slices):
            v00 = vid(i, j)
            v01 = vid(i, j + 1)
            v10 = vid(i + 1, j)
            v11 = vid(i + 1, j + 1)
            # two triangles for each quad strip; outward CCW
            # skip degenerate at poles is not necessary with this grid
            faces.append((v00, v01, v11))
            faces.append((v00, v11, v10))

    return ply_text_from_verts_faces(verts, faces)

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="Write ASCII PLY for rectangle/cylinder/ball.")
    sub = ap.add_subparsers(dest="shape", required=True)

    ap_rect = sub.add_parser("rectangle", help="axis-aligned box centered at origin")
    ap_rect.add_argument("out", help="output .ply")
    ap_rect.add_argument("--sx", type=float, default=1.0, help="size along X (default 1.0)")
    ap_rect.add_argument("--sy", type=float, default=1.0, help="size along Y (default 1.0)")
    ap_rect.add_argument("--sz", type=float, default=1.0, help="size along Z (default 1.0)")

    ap_cyl = sub.add_parser("cylinder", help="capped cylinder centered at origin")
    ap_cyl.add_argument("out", help="output .ply")
    ap_cyl.add_argument("--radius", type=float, default=1.0, help="radius (default 1.0)")
    ap_cyl.add_argument("--height", type=float, default=2.0, help="height (default 2.0)")
    ap_cyl.add_argument("--segments", type=int, default=48, help="circular segments (>=3)")

    ap_sph = sub.add_parser("ball", help="UV sphere centered at origin")
    ap_sph.add_argument("out", help="output .ply")
    ap_sph.add_argument("--radius", type=float, default=1.0, help="radius (default 1.0)")
    ap_sph.add_argument("--rings", type=int, default=32, help="latitudinal rings (>=2)")
    ap_sph.add_argument("--sectors", type=int, default=64, help="longitudinal sectors (>=3)")

    args = ap.parse_args()

    if args.shape == "rectangle":
        txt = rectangle_ply_text(args.sx, args.sy, args.sz)
    elif args.shape == "cylinder":
        txt = cylinder_ply_text(args.radius, args.height, args.segments)
    elif args.shape == "ball":
        txt = sphere_ply_text(args.radius, args.rings, args.sectors)
    else:
        print("Unknown shape", file=sys.stderr)
        sys.exit(2)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
