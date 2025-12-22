#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _load_latest_request(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Request dir not found: {base_dir}")
    candidates = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No request folders in {base_dir}")
    return candidates[-1]


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _draw_axes(img: np.ndarray, pose_matrix: np.ndarray, K: np.ndarray, length=0.05) -> np.ndarray:
    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    points = np.float32(
        [
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, length],
        ]
    ).reshape(-1, 3)
    img_points, _ = cv2.projectPoints(points, rvec, tvec, K, np.zeros(5))
    img_points = img_points.reshape(-1, 2).astype(int)
    out = img.copy()
    cv2.line(out, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 2)
    cv2.line(out, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 2)
    cv2.line(out, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 2)
    return out


def _tile(images, labels):
    resized = []
    target_h, target_w = images[0].shape[:2]
    for img, label in zip(images, labels):
        im = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.putText(im, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        resized.append(im)
    top = np.hstack(resized[:2])
    bottom = np.hstack(resized[2:])
    return np.vstack([top, bottom])


def main():
    parser = argparse.ArgumentParser(description="Visualize saved FoundationPose requests")
    parser.add_argument("--requests-dir", default="fp_requests", help="Base directory for requests")
    parser.add_argument("--request-id", default=None, help="Specific request ID folder")
    parser.add_argument("--latest", action="store_true", help="Use latest request (default)")
    parser.add_argument("--save", default=None, help="Save composite image to path")
    parser.add_argument("--no-show", action="store_true", help="Do not open a window")
    args = parser.parse_args()

    base_dir = Path(args.requests_dir)
    if args.request_id:
        req_dir = base_dir / args.request_id
    else:
        req_dir = _load_latest_request(base_dir)

    meta_path = req_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    rgb = cv2.imread(str(req_dir / "rgb.png"), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(req_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(req_dir / "depth.png"), cv2.IMREAD_UNCHANGED)

    if rgb is None or mask is None or depth is None:
        raise FileNotFoundError(f"Missing images in {req_dir}")

    K = np.array(meta["K"], dtype=np.float32)
    pose_matrix = meta.get("pose_matrix")
    roi = meta.get("roi") or {}

    rgb_vis = rgb.copy()
    if roi:
        cx = int(roi.get("x_center", 0))
        cy = int(roi.get("y_center", 0))
        radius = int(roi.get("radius", 0))
        cv2.circle(rgb_vis, (cx, cy), radius, (0, 255, 255), 2)
        cv2.circle(rgb_vis, (cx, cy), 3, (0, 255, 255), -1)

    if pose_matrix:
        pose = np.array(pose_matrix, dtype=np.float32)
        rgb_vis = _draw_axes(rgb_vis, pose, K)

    mask_vis = _ensure_bgr(mask)
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    error_text = meta.get("error")
    if error_text:
        err_img = np.zeros_like(rgb_vis)
        cv2.putText(err_img, f"Error: {error_text}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        err_img = rgb_vis.copy()
        cv2.putText(err_img, "Pose OK", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    composite = _tile([rgb_vis, mask_vis, depth_vis, err_img], ["RGB", "Mask", "Depth", "Result"])

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), composite)

    if not args.no_show:
        cv2.imshow(f"FoundationPose Request: {req_dir.name}", composite)
        cv2.waitKey(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
