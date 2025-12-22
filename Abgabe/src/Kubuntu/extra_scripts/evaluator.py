#!/usr/bin/env python3
"""
visualize_fp_pair.py

Visualize a saved FoundationPose request pair (AVP/UxPlay + RS) and plot pose error.

Assumes you saved paired captures with a shared "capture_id" in meta JSON:
  - AVP saves under:  <frame_dir>/foundationpose/
      fp_*_rgb.jpg, fp_*_mask.png, fp_*_depth.npy, fp_*_depth.png, fp_*_pose.npy (optional), fp_*_meta.json
  - RS saves under:   <frame_dir>/foundationpose_rs/
      fp_rs_*_rgb.jpg, fp_rs_*_mask.png, fp_rs_*_depth.npy, fp_rs_*_depth.png, fp_rs_*_pose.npy (optional), fp_rs_*_meta.json

It will:
  1) Find matching AVP + RS entries for a capture_id (or newest pair if not specified)
  2) Show RGB, mask, depth preview (and pose if available)
  3) Compute pose error:
       T_rs_obj_ref  = pose from RS request  (assumed RS camera frame)
       T_avp_obj     = pose from AVP request (assumed AVP camera frame)
       T_rs_obj_est  = inv(T_avp_rs) @ T_avp_obj
       Error:
         translation: ||t_ref - t_est|| (meters)
         rotation: angle(R_ref * R_est^T) (degrees)
  4) Plot a bar chart for translation and rotation error.

Usage:
  python visualize_fp_pair.py --base-dir /path/to/frames --capture-id <id>
  python visualize_fp_pair.py --base-dir /path/to/frames --latest
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import cv2
import matplotlib.pyplot as plt


@dataclass
class Entry:
    meta_path: str
    prefix: str  # full path without suffix (e.g. ".../fp_20250101_120000")
    meta: Dict[str, Any]
    rgb_path: Optional[str]
    mask_path: Optional[str]
    depth_npy_path: Optional[str]
    depth_png_path: Optional[str]
    pose_npy_path: Optional[str]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _try_find_one(prefix: str, suffix: str) -> Optional[str]:
    p = prefix + suffix
    return p if os.path.exists(p) else None


def _entry_from_meta(meta_path: str) -> Entry:
    meta = _load_json(meta_path)
    prefix = meta_path[:-len("_meta.json")] if meta_path.endswith("_meta.json") else os.path.splitext(meta_path)[0]

    rgb = _try_find_one(prefix, "_rgb.jpg")
    mask = _try_find_one(prefix, "_mask.png")
    depth_npy = _try_find_one(prefix, "_depth.npy")
    depth_png = _try_find_one(prefix, "_depth.png")
    pose_npy = _try_find_one(prefix, "_pose.npy")

    return Entry(
        meta_path=meta_path,
        prefix=prefix,
        meta=meta,
        rgb_path=rgb,
        mask_path=mask,
        depth_npy_path=depth_npy,
        depth_png_path=depth_png,
        pose_npy_path=pose_npy,
    )


def _find_entries(dir_path: str) -> List[Entry]:
    metas = sorted(glob.glob(os.path.join(dir_path, "*_meta.json")))
    return [_entry_from_meta(m) for m in metas]


def _index_by_capture_id(entries: List[Entry]) -> Dict[str, List[Entry]]:
    out: Dict[str, List[Entry]] = {}
    for e in entries:
        cid = str(e.meta.get("capture_id", "")).strip()
        if not cid:
            continue
        out.setdefault(cid, []).append(e)
    return out


def _pick_one(entries: List[Entry]) -> Entry:
    # pick the newest by meta file mtime
    entries = sorted(entries, key=lambda e: os.path.getmtime(e.meta_path), reverse=True)
    return entries[0]


def _load_pose_npy(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    try:
        T = np.load(path)
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            return None
        return T
    except Exception:
        return None


def _invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _rot_angle_deg(R_err: np.ndarray) -> float:
    # angle = acos((trace(R)-1)/2)
    tr = float(np.trace(R_err))
    x = (tr - 1.0) * 0.5
    x = max(-1.0, min(1.0, x))
    ang = np.arccos(x)
    return float(np.degrees(ang))


def _pose_errors(T_ref: np.ndarray, T_est: np.ndarray) -> Tuple[float, float]:
    t_ref = T_ref[:3, 3]
    t_est = T_est[:3, 3]
    trans_err = float(np.linalg.norm(t_ref - t_est))

    R_ref = T_ref[:3, :3]
    R_est = T_est[:3, :3]
    R_err = R_ref @ R_est.T
    rot_err = _rot_angle_deg(R_err)
    return trans_err, rot_err


def _read_bgr(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def _read_gray(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def _depth_preview(depth_m: np.ndarray) -> np.ndarray:
    # depth_m float32 meters -> 8-bit colormap
    d = np.asarray(depth_m, dtype=np.float32)
    valid = d > 0
    if not np.any(valid):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    dv = d.copy()
    dv[~valid] = 0
    d_norm = cv2.normalize(dv, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cm = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
    return cm


def visualize_pair(avp: Entry, rs: Entry, show: bool = True, save_fig: Optional[str] = None) -> None:
    # Load images
    avp_rgb = _read_bgr(avp.rgb_path)
    avp_mask = _read_gray(avp.mask_path)
    avp_depth = None
    if avp.depth_npy_path and os.path.exists(avp.depth_npy_path):
        avp_depth = np.load(avp.depth_npy_path).astype(np.float32)

    rs_rgb = _read_bgr(rs.rgb_path)
    rs_mask = _read_gray(rs.mask_path)
    rs_depth = None
    if rs.depth_npy_path and os.path.exists(rs.depth_npy_path):
        rs_depth = np.load(rs.depth_npy_path).astype(np.float32)

    # Pose + transform
    T_avp_obj = _load_pose_npy(avp.pose_npy_path) if avp.pose_npy_path else None
    T_rs_obj_ref = _load_pose_npy(rs.pose_npy_path) if rs.pose_npy_path else None

    T_avp_rs = None
    # best-effort: meta may store it under "T_avp_rs"
    if isinstance(avp.meta.get("T_avp_rs"), list):
        try:
            T_avp_rs = np.asarray(avp.meta["T_avp_rs"], dtype=np.float64)
            if T_avp_rs.shape != (4, 4):
                T_avp_rs = None
        except Exception:
            T_avp_rs = None

    # If any are missing, we can still visualize I/O; error plot needs all three.
    trans_err = rot_err = None
    T_rs_obj_est = None
    if (T_avp_obj is not None) and (T_rs_obj_ref is not None) and (T_avp_rs is not None):
        T_rs_obj_est = _invert_T(T_avp_rs) @ T_avp_obj
        trans_err, rot_err = _pose_errors(T_rs_obj_ref, T_rs_obj_est)

    cid = avp.meta.get("capture_id", "unknown")
    title = f"FoundationPose Pair | capture_id={cid}"

    # Build figure (I/O)
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(title)

    # Helper to show BGR with matplotlib (expects RGB)
    def show_bgr(ax, img_bgr, label):
        ax.set_title(label)
        ax.axis("off")
        if img_bgr is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

    def show_gray(ax, img_g, label):
        ax.set_title(label)
        ax.axis("off")
        if img_g is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            return
        ax.imshow(img_g, cmap="gray", vmin=0, vmax=255)

    def show_depth(ax, depth_m, label):
        ax.set_title(label)
        ax.axis("off")
        if depth_m is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            return
        cm_bgr = _depth_preview(depth_m)
        cm_rgb = cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(cm_rgb)

    # Layout: 2 rows x 3 cols for AVP + RS
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    show_bgr(ax1, avp_rgb, "AVP RGB")
    show_gray(ax2, avp_mask, "AVP mask")
    show_depth(ax3, avp_depth, "AVP depth (npy)")

    show_bgr(ax4, rs_rgb, "RS RGB")
    show_gray(ax5, rs_mask, "RS mask")
    show_depth(ax6, rs_depth, "RS depth (npy)")

    # Text block with pose info
    txt_lines = []
    txt_lines.append(f"capture_id: {cid}")
    txt_lines.append(f"AVP pose saved: {T_avp_obj is not None}")
    txt_lines.append(f"RS pose saved: {T_rs_obj_ref is not None}")
    txt_lines.append(f"T_avp_rs in meta: {T_avp_rs is not None}")
    if trans_err is not None and rot_err is not None:
        txt_lines.append(f"translation error (m): {trans_err:.6f}")
        txt_lines.append(f"rotation error (deg): {rot_err:.4f}")
    else:
        txt_lines.append("pose error: missing required pose/transform")

    fig.text(0.01, 0.01, "\n".join(txt_lines), fontsize=10, va="bottom")

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    if save_fig:
        fig.savefig(save_fig, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Error bar chart (separate fig)
    if trans_err is not None and rot_err is not None:
        fig2 = plt.figure(figsize=(7, 4))
        ax = fig2.add_subplot(1, 1, 1)
        ax.set_title(f"Pose error (RS vs AVP→RS) | capture_id={cid}")
        labels = ["translation (m)", "rotation (deg)"]
        values = [trans_err, rot_err]
        ax.bar(labels, values)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig2.tight_layout()
        if save_fig:
            root, ext = os.path.splitext(save_fig)
            fig2.savefig(root + "_errorbars.png", dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="Directory containing foundationpose/ and foundationpose_rs/")
    ap.add_argument("--capture-id", default=None, help="capture_id to visualize")
    ap.add_argument("--latest", action="store_true", help="use newest pair found")
    ap.add_argument("--no-show", action="store_true", help="do not open windows; useful with --save-fig")
    ap.add_argument("--save-fig", default=None, help="save I/O figure to this path (png/jpg)")
    args = ap.parse_args()

    avp_dir = os.path.join(args.base_dir, "foundationpose")
    rs_dir = os.path.join(args.base_dir, "foundationpose_rs")
    if not os.path.isdir(avp_dir) or not os.path.isdir(rs_dir):
        raise SystemExit(f"Expected directories:\n  {avp_dir}\n  {rs_dir}")

    avp_entries = _find_entries(avp_dir)
    rs_entries = _find_entries(rs_dir)

    avp_idx = _index_by_capture_id(avp_entries)
    rs_idx = _index_by_capture_id(rs_entries)

    common = sorted(set(avp_idx.keys()) & set(rs_idx.keys()))
    if not common:
        raise SystemExit("No matching capture_id found in both AVP and RS directories.")

    if args.capture_id:
        cid = args.capture_id
        if cid not in avp_idx or cid not in rs_idx:
            raise SystemExit(f"capture_id '{cid}' not found in both dirs.")
    else:
        if not args.latest:
            # default: newest common capture_id by meta mtime max across both
            def cid_mtime(c: str) -> float:
                a = max(os.path.getmtime(e.meta_path) for e in avp_idx[c])
                r = max(os.path.getmtime(e.meta_path) for e in rs_idx[c])
                return max(a, r)
            cid = sorted(common, key=cid_mtime, reverse=True)[0]
        else:
            # explicit latest = same behavior
            def cid_mtime(c: str) -> float:
                a = max(os.path.getmtime(e.meta_path) for e in avp_idx[c])
                r = max(os.path.getmtime(e.meta_path) for e in rs_idx[c])
                return max(a, r)
            cid = sorted(common, key=cid_mtime, reverse=True)[0]

    avp = _pick_one(avp_idx[cid])
    rs = _pick_one(rs_idx[cid])

    visualize_pair(avp, rs, show=(not args.no_show), save_fig=args.save_fig)


if __name__ == "__main__":
    main()
