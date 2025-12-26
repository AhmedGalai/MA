#!/usr/bin/env python3
"""
Interactive FoundationPose Request Evaluator

Loads all saved FoundationPose requests from fp_requests directory,
displays images for user selection, and evaluates:
1. Pose accuracy between RS and AVP side FP requests
2. End-to-end latency measurements

Based on evaluation methodology from evaluation.tex:
- RS-FoundationPose: Reference pipeline (direct RS camera frame)
- RS→UxPlay-FoundationPose: Evaluated pipeline (AVP frame with reprojected depth)

Usage:
    python evaluator.py --fp-dir /path/to/fp_requests
    python evaluator.py --fp-dir /path/to/fp_requests --auto-select-all
    python evaluator.py --fp-dir /path/to/fp_requests --capture-id 20251221_172838_465001
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons


@dataclass
class FPRequest:
    """Represents a single FoundationPose request (AVP or RS)"""
    capture_id: str
    source: str  # "avp" or "rs"
    timestamp: str
    directory: Path
    rgb_path: Path
    depth_path: Optional[Path]
    mask_path: Optional[Path]
    metadata_path: Path
    metadata: Dict[str, Any]
    pose_matrix: Optional[np.ndarray]
    latency_ms: Optional[float]


@dataclass
class FPPair:
    """Paired AVP and RS requests with same capture_id"""
    capture_id: str
    avp_request: FPRequest
    rs_request: FPRequest

    def has_both_poses(self) -> bool:
        return (self.avp_request.pose_matrix is not None and
                self.rs_request.pose_matrix is not None)

    def has_latency_data(self) -> bool:
        return (self.avp_request.latency_ms is not None or
                self.rs_request.latency_ms is not None)


def load_pose_matrix(metadata: Dict[str, Any], pose_path: Optional[Path]) -> Optional[np.ndarray]:
    """Load pose matrix from metadata or a .npy file."""
    pose_matrix = None
    if "pose_matrix" in metadata and metadata["pose_matrix"] is not None:
        try:
            pose_matrix = np.array(metadata["pose_matrix"], dtype=np.float64)
            if pose_matrix.shape != (4, 4):
                pose_matrix = None
        except Exception:
            pose_matrix = None

    if pose_matrix is None and pose_path is not None and pose_path.exists():
        try:
            pose_matrix = np.load(pose_path)
            if pose_matrix.shape != (4, 4):
                pose_matrix = None
        except Exception:
            pose_matrix = None

    return pose_matrix


def load_request(directory: Path) -> Optional[FPRequest]:
    """Load a single FP request from a directory"""
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading {metadata_path}: {e}")
        return None

    # Extract fields
    capture_id = metadata.get("capture_id") or directory.name
    source = metadata.get("source", "unknown")
    timestamp = metadata.get("timestamp", directory.name)

    pose_path = directory / "pose.npy"
    pose_matrix = load_pose_matrix(metadata, pose_path)

    # Extract latency
    latency_ms = None
    if "latency" in metadata and isinstance(metadata["latency"], dict):
        latency_ms = metadata["latency"].get("foundationpose_api_ms")

    # Find image files
    rgb_path = directory / "rgb.png"
    if not rgb_path.exists():
        rgb_path = directory / "rgb.jpg"

    depth_path = directory / "depth.png"
    if not depth_path.exists():
        depth_path = None

    mask_path = directory / "mask.png"
    if not mask_path.exists():
        mask_path = None

    return FPRequest(
        capture_id=capture_id,
        source=source,
        timestamp=timestamp,
        directory=directory,
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_path=mask_path,
        metadata_path=metadata_path,
        metadata=metadata,
        pose_matrix=pose_matrix,
        latency_ms=latency_ms
    )


def load_flat_request(metadata_path: Path) -> Optional[FPRequest]:
    """Load a single FP request from flat files in a directory."""
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading {metadata_path}: {e}")
        return None

    base_name = metadata_path.name
    if not base_name.endswith("_meta.json"):
        return None
    prefix = base_name[:-len("_meta.json")]

    capture_id = metadata.get("capture_id")
    if not capture_id:
        capture_id = prefix
        if capture_id.startswith("fp_"):
            capture_id = capture_id[len("fp_"):]

    source = metadata.get("source", "unknown")
    timestamp = metadata.get("timestamp", capture_id)

    rgb_path = metadata_path.with_name(f"{prefix}_rgb.png")
    if not rgb_path.exists():
        rgb_path = metadata_path.with_name(f"{prefix}_rgb.jpg")

    depth_path = metadata_path.with_name(f"{prefix}_depth.png")
    if not depth_path.exists():
        depth_path = None

    mask_path = metadata_path.with_name(f"{prefix}_mask.png")
    if not mask_path.exists():
        mask_path = None

    pose_path = metadata_path.with_name(f"{prefix}_pose.npy")
    pose_matrix = load_pose_matrix(metadata, pose_path)

    latency_ms = None
    if "latency" in metadata and isinstance(metadata["latency"], dict):
        latency_ms = metadata["latency"].get("foundationpose_api_ms")

    return FPRequest(
        capture_id=capture_id,
        source=source,
        timestamp=timestamp,
        directory=metadata_path.parent,
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_path=mask_path,
        metadata_path=metadata_path,
        metadata=metadata,
        pose_matrix=pose_matrix,
        latency_ms=latency_ms
    )


def load_all_requests(fp_dir: Path) -> List[FPRequest]:
    """Load all FP requests from directory"""
    requests = []

    for subdir in sorted(fp_dir.iterdir()):
        if not subdir.is_dir():
            continue

        request = load_request(subdir)
        if request:
            requests.append(request)

    if not requests:
        for metadata_path in sorted(fp_dir.glob("*_meta.json")):
            request = load_flat_request(metadata_path)
            if request:
                requests.append(request)

    print(f"Loaded {len(requests)} requests from {fp_dir}")
    return requests


def pair_requests(requests: List[FPRequest]) -> List[FPPair]:
    """Match AVP and RS requests by capture_id"""
    # Index by capture_id and source
    by_id: Dict[str, Dict[str, FPRequest]] = {}

    for req in requests:
        if req.capture_id not in by_id:
            by_id[req.capture_id] = {}
        by_id[req.capture_id][req.source] = req

    # Create pairs
    pairs = []
    for capture_id, sources in by_id.items():
        if "avp" in sources and "rs" in sources:
            pairs.append(FPPair(
                capture_id=capture_id,
                avp_request=sources["avp"],
                rs_request=sources["rs"]
            ))

    print(f"Found {len(pairs)} matched pairs (AVP + RS)")
    return pairs


def compute_pose_errors(T_ref: np.ndarray, T_est: np.ndarray) -> Tuple[float, float]:
    """
    Compute translation and rotation errors between two poses.

    From evaluation.tex section 4.2:
    - Translation error: ||t_ref - t_est||_2
    - Rotation error: arccos((trace(R_ref @ R_est^T) - 1) / 2)

    Returns:
        (translation_error_m, rotation_error_deg)
    """
    # Translation error
    t_ref = T_ref[:3, 3]
    t_est = T_est[:3, 3]
    trans_err = float(np.linalg.norm(t_ref - t_est))

    # Rotation error
    R_ref = T_ref[:3, :3]
    R_est = T_est[:3, :3]
    R_err = R_ref @ R_est.T

    trace_val = float(np.trace(R_err))
    # Clamp to avoid numerical issues with arccos
    cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    rot_err_rad = np.arccos(cos_angle)
    rot_err_deg = float(np.degrees(rot_err_rad))

    return trans_err, rot_err_deg


def transform_avp_to_rs(T_avp_obj: np.ndarray, T_avp_rs: np.ndarray) -> np.ndarray:
    """
    Transform AVP-frame pose to RS frame.

    T_rs_obj = inv(T_avp_rs) @ T_avp_obj
    """
    R = T_avp_rs[:3, :3]
    t = T_avp_rs[:3, 3]

    T_rs_avp = np.eye(4)
    T_rs_avp[:3, :3] = R.T
    T_rs_avp[:3, 3] = -R.T @ t

    return T_rs_avp @ T_avp_obj


def evaluate_pair(pair: FPPair) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single pair of AVP/RS requests.

    Returns dictionary with:
    - translation_error_m
    - rotation_error_deg
    - avp_latency_ms
    - rs_latency_ms
    - has_pose_error: bool
    - has_latency: bool
    """
    result = {
        "capture_id": pair.capture_id,
        "translation_error_m": None,
        "rotation_error_deg": None,
        "avp_latency_ms": pair.avp_request.latency_ms,
        "rs_latency_ms": pair.rs_request.latency_ms,
        "has_pose_error": False,
        "has_latency": False,
    }

    # Compute pose error if both poses available
    if pair.has_both_poses():
        # Get T_avp_rs from AVP metadata
        T_avp_rs = None
        if "T_avp_rs" in pair.avp_request.metadata:
            try:
                T_avp_rs = np.array(pair.avp_request.metadata["T_avp_rs"], dtype=np.float64)
                if T_avp_rs.shape != (4, 4):
                    T_avp_rs = None
            except:
                T_avp_rs = None

        if T_avp_rs is not None:
            # Transform AVP pose to RS frame
            T_avp_obj = pair.avp_request.pose_matrix
            T_rs_obj_from_avp = transform_avp_to_rs(T_avp_obj, T_avp_rs)

            # Compare with RS reference pose
            T_rs_obj_ref = pair.rs_request.pose_matrix

            trans_err, rot_err = compute_pose_errors(T_rs_obj_ref, T_rs_obj_from_avp)

            result["translation_error_m"] = trans_err
            result["rotation_error_deg"] = rot_err
            result["has_pose_error"] = True
        else:
            print(f"  Warning: {pair.capture_id} - T_avp_rs not available in metadata")

    # Check latency
    if pair.has_latency_data():
        result["has_latency"] = True

    return result


class InteractiveSelectorGUI:
    """Interactive GUI for selecting pairs to evaluate"""

    def __init__(self, pairs: List[FPPair]):
        self.pairs = pairs
        self.selected = [False] * len(pairs)
        self.current_idx = 0

        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("FoundationPose Request Selector", fontsize=14, fontweight='bold')

        # Create axes for images
        self.ax_avp_rgb = self.fig.add_subplot(2, 4, 1)
        self.ax_avp_mask = self.fig.add_subplot(2, 4, 2)
        self.ax_avp_depth = self.fig.add_subplot(2, 4, 3)
        self.ax_rs_rgb = self.fig.add_subplot(2, 4, 5)
        self.ax_rs_mask = self.fig.add_subplot(2, 4, 6)
        self.ax_rs_depth = self.fig.add_subplot(2, 4, 7)

        # Info panel
        self.ax_info = self.fig.add_subplot(1, 4, 4)
        self.ax_info.axis('off')

        # Navigation panel
        self.ax_nav = self.fig.add_subplot(2, 4, 8)
        self.ax_nav.axis('off')

        # Buttons
        ax_prev = plt.axes([0.15, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.26, 0.02, 0.1, 0.04])
        ax_toggle = plt.axes([0.37, 0.02, 0.15, 0.04])
        ax_all = plt.axes([0.53, 0.02, 0.12, 0.04])
        ax_done = plt.axes([0.66, 0.02, 0.12, 0.04])

        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_toggle = Button(ax_toggle, 'Toggle Selected')
        self.btn_all = Button(ax_all, 'Select All')
        self.btn_done = Button(ax_done, 'Done')

        self.btn_prev.on_clicked(lambda e: self.prev_pair())
        self.btn_next.on_clicked(lambda e: self.next_pair())
        self.btn_toggle.on_clicked(lambda e: self.toggle_current())
        self.btn_all.on_clicked(lambda e: self.select_all())
        self.btn_done.on_clicked(lambda e: plt.close(self.fig))

        self.display_current()

    def load_and_show_image(self, ax, path: Optional[Path], title: str, is_depth=False):
        """Load and display an image in an axes"""
        ax.clear()
        ax.set_title(title)
        ax.axis('off')

        if path is None or not path.exists():
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            return

        try:
            if is_depth:
                # Load and colormap depth
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(str(path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'Load Error', ha='center', va='center', fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}', ha='center', va='center', fontsize=8)

    def display_current(self):
        """Display current pair"""
        if not self.pairs:
            return

        pair = self.pairs[self.current_idx]

        # Load AVP images
        self.load_and_show_image(self.ax_avp_rgb, pair.avp_request.rgb_path,
                                 "AVP RGB")
        self.load_and_show_image(self.ax_avp_mask, pair.avp_request.mask_path,
                                 "AVP Mask")
        self.load_and_show_image(self.ax_avp_depth, pair.avp_request.depth_path,
                                 "AVP Depth", is_depth=True)

        # Load RS images
        self.load_and_show_image(self.ax_rs_rgb, pair.rs_request.rgb_path,
                                 "RS RGB")
        self.load_and_show_image(self.ax_rs_mask, pair.rs_request.mask_path,
                                 "RS Mask")
        self.load_and_show_image(self.ax_rs_depth, pair.rs_request.depth_path,
                                 "RS Depth", is_depth=True)

        # Update info panel
        self.ax_info.clear()
        self.ax_info.axis('off')

        info_text = [
            f"Pair {self.current_idx + 1} / {len(self.pairs)}",
            "",
            f"Capture ID:",
            f"  {pair.capture_id}",
            "",
            f"Selected: {'✓ YES' if self.selected[self.current_idx] else '✗ NO'}",
            "",
            f"AVP:",
            f"  Timestamp: {pair.avp_request.timestamp}",
            f"  Has pose: {pair.avp_request.pose_matrix is not None}",
            f"  Latency: {pair.avp_request.latency_ms:.1f}ms" if pair.avp_request.latency_ms else "  Latency: N/A",
            "",
            f"RS:",
            f"  Timestamp: {pair.rs_request.timestamp}",
            f"  Has pose: {pair.rs_request.pose_matrix is not None}",
            f"  Latency: {pair.rs_request.latency_ms:.1f}ms" if pair.rs_request.latency_ms else "  Latency: N/A",
            "",
            f"Evaluable:",
            f"  Pose error: {pair.has_both_poses()}",
            f"  Latency: {pair.has_latency_data()}",
        ]

        self.ax_info.text(0.05, 0.95, '\n'.join(info_text),
                         fontsize=9, va='top', family='monospace')

        # Update nav panel
        self.ax_nav.clear()
        self.ax_nav.axis('off')
        selected_count = sum(self.selected)
        nav_text = f"Selected: {selected_count} / {len(self.pairs)}"
        self.ax_nav.text(0.5, 0.5, nav_text, ha='center', va='center', fontsize=12)

        plt.draw()

    def prev_pair(self):
        """Go to previous pair"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.display_current()

    def next_pair(self):
        """Go to next pair"""
        if self.current_idx < len(self.pairs) - 1:
            self.current_idx += 1
            self.display_current()

    def toggle_current(self):
        """Toggle selection of current pair"""
        self.selected[self.current_idx] = not self.selected[self.current_idx]
        self.display_current()

    def select_all(self):
        """Select all pairs"""
        self.selected = [True] * len(self.pairs)
        self.display_current()

    def get_selected_pairs(self) -> List[FPPair]:
        """Get list of selected pairs"""
        return [pair for pair, sel in zip(self.pairs, self.selected) if sel]


def generate_report(results: List[Dict[str, Any]], output_dir: Path):
    """Generate evaluation report with plots and statistics"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter valid results
    pose_results = [r for r in results if r["has_pose_error"]]
    latency_results = [r for r in results if r["has_latency"]]

    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Total pairs evaluated: {len(results)}")
    print(f"  With pose error: {len(pose_results)}")
    print(f"  With latency data: {len(latency_results)}")

    # Pose accuracy statistics
    if pose_results:
        trans_errors = [r["translation_error_m"] for r in pose_results]
        rot_errors = [r["rotation_error_deg"] for r in pose_results]

        print("\n" + "-"*70)
        print("POSE ACCURACY (RS Reference vs AVP→RS Evaluated)")
        print("-"*70)
        print(f"Translation Error (m):")
        print(f"  Mean:   {np.mean(trans_errors):.6f}")
        print(f"  Median: {np.median(trans_errors):.6f}")
        print(f"  Std:    {np.std(trans_errors):.6f}")
        print(f"  Min:    {np.min(trans_errors):.6f}")
        print(f"  Max:    {np.max(trans_errors):.6f}")
        print()
        print(f"Rotation Error (deg):")
        print(f"  Mean:   {np.mean(rot_errors):.4f}")
        print(f"  Median: {np.median(rot_errors):.4f}")
        print(f"  Std:    {np.std(rot_errors):.4f}")
        print(f"  Min:    {np.min(rot_errors):.4f}")
        print(f"  Max:    {np.max(rot_errors):.4f}")

        # Plot pose errors
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Pose Error Distribution", fontsize=14, fontweight='bold')

        ax1.hist(trans_errors, bins=20, edgecolor='black')
        ax1.set_xlabel("Translation Error (m)")
        ax1.set_ylabel("Count")
        ax1.set_title("Translation Error Distribution")
        ax1.grid(True, alpha=0.3)

        ax2.hist(rot_errors, bins=20, edgecolor='black', color='orange')
        ax2.set_xlabel("Rotation Error (deg)")
        ax2.set_ylabel("Count")
        ax2.set_title("Rotation Error Distribution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        pose_plot_path = output_dir / "pose_errors.png"
        plt.savefig(pose_plot_path, dpi=150)
        print(f"\nPose error plot saved to: {pose_plot_path}")
        plt.close()

    # Latency statistics
    if latency_results:
        avp_latencies = [r["avp_latency_ms"] for r in latency_results if r["avp_latency_ms"]]
        rs_latencies = [r["rs_latency_ms"] for r in latency_results if r["rs_latency_ms"]]

        print("\n" + "-"*70)
        print("END-TO-END LATENCY")
        print("-"*70)

        if avp_latencies:
            print(f"AVP Pipeline (ms):")
            print(f"  Mean:   {np.mean(avp_latencies):.2f}")
            print(f"  Median: {np.median(avp_latencies):.2f}")
            print(f"  Std:    {np.std(avp_latencies):.2f}")
            print(f"  Min:    {np.min(avp_latencies):.2f}")
            print(f"  Max:    {np.max(avp_latencies):.2f}")

        if rs_latencies:
            print()
            print(f"RS Pipeline (ms):")
            print(f"  Mean:   {np.mean(rs_latencies):.2f}")
            print(f"  Median: {np.median(rs_latencies):.2f}")
            print(f"  Std:    {np.std(rs_latencies):.2f}")
            print(f"  Min:    {np.min(rs_latencies):.2f}")
            print(f"  Max:    {np.max(rs_latencies):.2f}")

        # Plot latency comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        if avp_latencies and rs_latencies:
            ax.boxplot([avp_latencies, rs_latencies], labels=['AVP Pipeline', 'RS Pipeline'])
            ax.set_ylabel("Latency (ms)")
            ax.set_title("FoundationPose API Latency Comparison")
            ax.grid(True, alpha=0.3, axis='y')
        elif avp_latencies:
            ax.hist(avp_latencies, bins=20, edgecolor='black', label='AVP')
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("Count")
            ax.set_title("AVP Pipeline Latency Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)
        elif rs_latencies:
            ax.hist(rs_latencies, bins=20, edgecolor='black', label='RS', color='orange')
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("Count")
            ax.set_title("RS Pipeline Latency Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        latency_plot_path = output_dir / "latency_comparison.png"
        plt.savefig(latency_plot_path, dpi=150)
        print(f"\nLatency plot saved to: {latency_plot_path}")
        plt.close()

    # Save detailed results to JSON
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive FoundationPose Request Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive selection
  python evaluator.py --fp-dir /path/to/fp_requests

  # Auto-select all pairs
  python evaluator.py --fp-dir /path/to/fp_requests --auto-select-all

  # Evaluate specific capture ID
  python evaluator.py --fp-dir /path/to/fp_requests --capture-id 20251221_172838_465001
        """
    )

    parser.add_argument("--fp-dir", required=True,
                       help="Directory containing FP request subdirectories")
    parser.add_argument("--capture-id", default=None,
                       help="Evaluate specific capture_id only")
    parser.add_argument("--auto-select-all", action="store_true",
                       help="Automatically select all pairs (no GUI)")
    parser.add_argument("--output-dir", default="./evaluation_output",
                       help="Output directory for results (default: ./evaluation_output)")

    args = parser.parse_args()

    fp_dir = Path(args.fp_dir)
    if not fp_dir.exists():
        print(f"Error: {fp_dir} does not exist")
        return 1

    output_dir = Path(args.output_dir)

    # Load all requests
    print("Loading requests...")
    all_requests = load_all_requests(fp_dir)

    if not all_requests:
        print("No requests found!")
        return 1

    # Pair requests
    pairs = pair_requests(all_requests)

    if not pairs:
        print("No matching pairs found!")
        return 1

    # Filter by capture_id if specified
    if args.capture_id:
        pairs = [p for p in pairs if p.capture_id == args.capture_id]
        if not pairs:
            print(f"No pair found with capture_id: {args.capture_id}")
            return 1
        print(f"Evaluating single pair: {args.capture_id}")
        selected_pairs = pairs
    elif args.auto_select_all:
        print("Auto-selecting all pairs...")
        selected_pairs = pairs
    else:
        # Interactive selection
        print("\n" + "="*70)
        print("INTERACTIVE PAIR SELECTION")
        print("="*70)
        print("Use the GUI to select pairs for evaluation:")
        print("  - Click 'Toggle Selected' to select/deselect current pair")
        print("  - Use 'Previous'/'Next' to navigate")
        print("  - Click 'Select All' to select all pairs")
        print("  - Click 'Done' when finished")
        print("="*70 + "\n")

        gui = InteractiveSelectorGUI(pairs)
        plt.show()
        selected_pairs = gui.get_selected_pairs()

    if not selected_pairs:
        print("\nNo pairs selected for evaluation.")
        return 0

    print(f"\nEvaluating {len(selected_pairs)} pairs...")

    # Evaluate selected pairs
    results = []
    for i, pair in enumerate(selected_pairs, 1):
        print(f"  [{i}/{len(selected_pairs)}] {pair.capture_id}")
        result = evaluate_pair(pair)
        if result:
            results.append(result)

    # Generate report
    generate_report(results, output_dir)

    print("\nEvaluation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
