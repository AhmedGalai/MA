#!/usr/bin/env python3
"""
Compare RealSense depth/disparity with HuggingFace monocular depth estimation.
Displays side-by-side comparison in real-time.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from PIL import Image
import time


class DepthComparison:
    def __init__(self, width=640, height=480, fps=30, model_name="Intel/dpt-hybrid-midas"):
        """
        Initialize RealSense camera and HuggingFace depth estimation model.

        Args:
            width: Camera width
            height: Camera height
            fps: Camera FPS
            model_name: HuggingFace model for depth estimation
                Options:
                - "Intel/dpt-hybrid-midas" (default, good balance)
                - "Intel/dpt-large" (more accurate, slower)
                - "facebook/dpt-dinov2-small-kitti" (faster, less accurate)
        """
        self.width = width
        self.height = height
        self.fps = fps

        print("Initializing RealSense camera...")
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Enable streams
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Start streaming
        self.profile = self.pipeline.start(config)

        # Get depth scale (for converting depth values to meters)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {self.depth_scale}")

        # Create align object to align depth to color
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Get camera intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.baseline = 0.055  # Approximate baseline for D435 in meters

        print(f"Camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}")

        # Initialize HuggingFace depth estimation model
        print(f"Loading HuggingFace model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("Initialization complete!")

        # Statistics
        self.rs_frame_count = 0
        self.hf_frame_count = 0
        self.hf_inference_times = []

    def depth_to_disparity(self, depth, method="realsense"):
        """
        Convert depth map to disparity.

        Args:
            depth: Depth map (in meters for RS, arbitrary units for HF)
            method: "realsense" or "huggingface"
        """
        if method == "realsense":
            # For RealSense: disparity = baseline * fx / depth
            # Handle division by zero
            valid_mask = depth > 0
            disparity = np.zeros_like(depth, dtype=np.float32)
            disparity[valid_mask] = (self.baseline * self.fx) / depth[valid_mask]
            return disparity
        else:
            # For HuggingFace: inverse depth (already relative disparity)
            # Normalize and invert
            valid_mask = depth > 0
            disparity = np.zeros_like(depth, dtype=np.float32)
            if np.any(valid_mask):
                # Inverse for disparity-like representation
                disparity[valid_mask] = 1.0 / (depth[valid_mask] + 1e-6)
            return disparity

    def normalize_for_display(self, disparity, colormap=cv2.COLORMAP_JET, reverse=False):
        """Normalize disparity/depth for visualization.

        Args:
            disparity: Disparity/depth map
            colormap: OpenCV colormap to use
            reverse: If True, reverse the colormap (far=red, near=blue instead of far=blue, near=red)
        """
        if disparity.max() > disparity.min():
            normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            normalized = np.zeros_like(disparity, dtype=np.uint8)

        # Reverse colormap if requested (inverts the color mapping)
        if reverse:
            normalized = 255 - normalized

        # Apply colormap
        colored = cv2.applyColorMap(normalized, colormap)
        return colored

    def estimate_depth_huggingface(self, rgb_image):
        """
        Estimate depth using HuggingFace model.

        Args:
            rgb_image: RGB image (numpy array, BGR format from cv2)

        Returns:
            Depth map (numpy array)
        """
        start_time = time.perf_counter()

        # Convert BGR to RGB
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Prepare image for model
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=rgb_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )

        # Convert to numpy
        depth = prediction.squeeze().cpu().numpy()

        inference_time = (time.perf_counter() - start_time) * 1000
        self.hf_inference_times.append(inference_time)
        if len(self.hf_inference_times) > 100:
            self.hf_inference_times.pop(0)

        return depth, inference_time

    def get_frames(self):
        """Get aligned frames from RealSense."""
        frames = self.pipeline.wait_for_frames()

        # Align depth to color
        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            return None, None, None

        # Convert to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert depth to meters
        depth_meters = depth_image * self.depth_scale

        return color_image, depth_meters, depth_image

    def create_comparison_view(self, color_image, rs_disparity, hf_disparity,
                               rs_inference_time=0, hf_inference_time=0):
        """Create side-by-side comparison visualization."""
        # Normalize both for display
        # RealSense: normal colormap (near=red, far=blue)
        rs_colored = self.normalize_for_display(rs_disparity, cv2.COLORMAP_JET, reverse=False)
        # HuggingFace: reversed colormap (near=blue, far=red) to match depth convention
        hf_colored = self.normalize_for_display(hf_disparity, cv2.COLORMAP_JET, reverse=True)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)

        # RealSense label
        rs_labeled = rs_colored.copy()
        cv2.putText(rs_labeled, "RealSense Disparity", (10, 30),
                   font, font_scale, color, thickness)
        cv2.putText(rs_labeled, f"Aligned depth sensor", (10, 60),
                   font, 0.5, color, 1)

        # HuggingFace label
        hf_labeled = hf_colored.copy()
        cv2.putText(hf_labeled, "HuggingFace Depth Est.", (10, 30),
                   font, font_scale, color, thickness)
        cv2.putText(hf_labeled, f"Inference: {hf_inference_time:.1f}ms", (10, 60),
                   font, 0.5, color, 1)

        # Create top row: original image + stats
        stats_image = color_image.copy()

        # Calculate statistics
        rs_valid = rs_disparity[rs_disparity > 0]
        hf_valid = hf_disparity[hf_disparity > 0]

        stats_text = [
            "Depth Comparison Statistics",
            f"RealSense: min={rs_valid.min():.2f}, max={rs_valid.max():.2f}" if len(rs_valid) > 0 else "RealSense: No data",
            f"HuggingFace: min={hf_valid.min():.4f}, max={hf_valid.max():.4f}" if len(hf_valid) > 0 else "HuggingFace: No data",
            f"Avg HF inference: {np.mean(self.hf_inference_times):.1f}ms" if self.hf_inference_times else "",
            "",
            "Press 'q' to quit, 's' to save snapshot"
        ]

        y_offset = 30
        for text in stats_text:
            cv2.putText(stats_image, text, (10, y_offset),
                       font, 0.5, (0, 255, 0), 1)
            y_offset += 25

        # Stack images: [Original | Stats]
        #                [RS Disp | HF Disp]
        top_row = np.hstack([color_image, stats_image])
        bottom_row = np.hstack([rs_labeled, hf_labeled])

        # Ensure same width
        if top_row.shape[1] != bottom_row.shape[1]:
            # Resize top row to match bottom
            top_row = cv2.resize(top_row, (bottom_row.shape[1], top_row.shape[0]))

        combined = np.vstack([top_row, bottom_row])

        return combined

    def run(self):
        """Main loop: capture, process, and display."""
        print("\nStarting comparison...")
        print("Press 'q' to quit")
        print("Press 's' to save snapshot")
        print("-" * 50)

        try:
            while True:
                # Get RealSense frames
                color_image, depth_meters, depth_raw = self.get_frames()

                if color_image is None:
                    continue

                self.rs_frame_count += 1

                # Convert RealSense depth to disparity
                rs_disparity = self.depth_to_disparity(depth_meters, method="realsense")

                # Estimate depth with HuggingFace (every frame or skip for performance)
                # For real-time, you might want to process every N frames
                if self.hf_frame_count % 1 == 0:  # Process every frame
                    hf_depth, hf_time = self.estimate_depth_huggingface(color_image)
                    hf_disparity = self.depth_to_disparity(hf_depth, method="huggingface")
                else:
                    hf_time = 0

                self.hf_frame_count += 1

                # Create comparison view
                comparison = self.create_comparison_view(
                    color_image, rs_disparity, hf_disparity,
                    0, hf_time
                )

                # Display
                cv2.imshow("Depth Comparison: RealSense vs HuggingFace", comparison)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save snapshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"depth_comparison_{timestamp}.png"
                    cv2.imwrite(filename, comparison)
                    print(f"Saved snapshot: {filename}")

                    # Also save individual arrays
                    np.save(f"rs_disparity_{timestamp}.npy", rs_disparity)
                    np.save(f"hf_disparity_{timestamp}.npy", hf_disparity)
                    print(f"Saved disparity arrays")

        finally:
            self.cleanup()

    def cleanup(self):
        """Stop camera and close windows."""
        print("\nCleaning up...")
        self.pipeline.stop()
        cv2.destroyAllWindows()

        print(f"Total frames: RS={self.rs_frame_count}, HF={self.hf_frame_count}")
        if self.hf_inference_times:
            print(f"HuggingFace inference: avg={np.mean(self.hf_inference_times):.1f}ms, "
                  f"min={np.min(self.hf_inference_times):.1f}ms, "
                  f"max={np.max(self.hf_inference_times):.1f}ms")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare RealSense and HuggingFace depth estimation")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--model", type=str, default="Intel/dpt-hybrid-midas",
                       help="HuggingFace model name")

    args = parser.parse_args()

    print("=" * 60)
    print("Depth Estimation Comparison")
    print("=" * 60)
    print(f"Camera: {args.width}x{args.height} @ {args.fps}fps")
    print(f"Model: {args.model}")
    print("=" * 60)

    comparator = DepthComparison(
        width=args.width,
        height=args.height,
        fps=args.fps,
        model_name=args.model
    )

    comparator.run()


if __name__ == "__main__":
    main()
