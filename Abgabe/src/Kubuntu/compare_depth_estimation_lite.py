#!/usr/bin/env python3
"""
Lightweight depth comparison - processes HuggingFace model every N frames for better performance.
Optimized for CPU systems.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from PIL import Image
import time
import threading


class DepthComparisonLite:
    def __init__(self, width=640, height=480, fps=30,
                 model_name="Intel/dpt-hybrid-midas",
                 hf_process_interval=5):
        """
        Initialize with frame skipping for better performance.

        Args:
            width: Camera width
            height: Camera height
            fps: Camera FPS
            model_name: HuggingFace model
            hf_process_interval: Process HF model every N frames (default: 5)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.hf_process_interval = hf_process_interval

        print("Initializing RealSense camera...")
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.profile = self.pipeline.start(config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.baseline = 0.055

        print(f"Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Threading for async HF processing
        self.hf_depth = None
        self.hf_disparity = None
        self.hf_processing = False
        self.hf_lock = threading.Lock()
        self.latest_rgb = None
        self.hf_inference_time = 0
        self.frame_count = 0

        print("Ready!")

    def depth_to_disparity(self, depth, method="realsense"):
        """Convert depth to disparity."""
        if method == "realsense":
            valid_mask = depth > 0
            disparity = np.zeros_like(depth, dtype=np.float32)
            disparity[valid_mask] = (self.baseline * self.fx) / depth[valid_mask]
            return disparity
        else:
            valid_mask = depth > 0
            disparity = np.zeros_like(depth, dtype=np.float32)
            if np.any(valid_mask):
                disparity[valid_mask] = 1.0 / (depth[valid_mask] + 1e-6)
            return disparity

    def normalize_for_display(self, disparity, colormap=cv2.COLORMAP_JET, reverse=False):
        """Normalize for visualization.

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

        return cv2.applyColorMap(normalized, colormap)

    def process_huggingface_async(self, rgb_image):
        """Process HuggingFace model in background thread."""
        def _process():
            with self.hf_lock:
                self.hf_processing = True

            start_time = time.perf_counter()

            # Convert and process
            rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            inputs = self.image_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=rgb_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            )

            depth = prediction.squeeze().cpu().numpy()
            disparity = self.depth_to_disparity(depth, method="huggingface")

            inference_time = (time.perf_counter() - start_time) * 1000

            with self.hf_lock:
                self.hf_depth = depth
                self.hf_disparity = disparity
                self.hf_inference_time = inference_time
                self.hf_processing = False

        thread = threading.Thread(target=_process, daemon=True)
        thread.start()

    def get_frames(self):
        """Get aligned frames."""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_meters = depth_image * self.depth_scale

        return color_image, depth_meters

    def run(self):
        """Main loop with async HF processing."""
        print(f"\nRunning (HF processes every {self.hf_process_interval} frames)...")
        print("Press 'q' to quit, 's' to save, '+'/'-' to adjust interval")
        print("-" * 60)

        while True:
            color_image, depth_meters = self.get_frames()

            if color_image is None:
                continue

            self.frame_count += 1

            # RealSense disparity (always computed)
            rs_disparity = self.depth_to_disparity(depth_meters, method="realsense")
            # RealSense: normal colormap (near=red, far=blue)
            rs_colored = self.normalize_for_display(rs_disparity, reverse=False)

            # Trigger HF processing if interval reached and not already processing
            if self.frame_count % self.hf_process_interval == 0:
                with self.hf_lock:
                    if not self.hf_processing:
                        self.process_huggingface_async(color_image)

            # Get latest HF result (or placeholder)
            with self.hf_lock:
                if self.hf_disparity is not None:
                    hf_disparity = self.hf_disparity.copy()
                    hf_time = self.hf_inference_time
                    processing = self.hf_processing
                else:
                    hf_disparity = np.zeros_like(rs_disparity)
                    hf_time = 0
                    processing = self.hf_processing

            # HuggingFace: reversed colormap (near=blue, far=red) to match depth convention
            hf_colored = self.normalize_for_display(hf_disparity, reverse=True)

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Label RealSense
            cv2.putText(rs_colored, "RealSense (Real-time)", (10, 30),
                       font, 0.6, (255, 255, 255), 2)

            # Label HuggingFace
            status = "Processing..." if processing else f"{hf_time:.0f}ms"
            cv2.putText(hf_colored, f"HuggingFace (1/{self.hf_process_interval} frames)", (10, 30),
                       font, 0.6, (255, 255, 255), 2)
            cv2.putText(hf_colored, f"Inference: {status}", (10, 60),
                       font, 0.5, (255, 255, 255), 1)

            # Create comparison
            top = np.hstack([color_image, color_image])
            bottom = np.hstack([rs_colored, hf_colored])

            # Add info overlay on top row
            info_text = [
                f"Frame: {self.frame_count}",
                f"HF Interval: {self.hf_process_interval} (press +/- to adjust)",
                f"Processing: {'Yes' if processing else 'No'}",
                "Press 'q' to quit, 's' to save"
            ]

            y = 30
            for text in info_text:
                cv2.putText(top, text, (self.width + 10, y),
                           font, 0.4, (0, 255, 0), 1)
                y += 20

            combined = np.vstack([top, bottom])

            cv2.imshow("Depth Comparison (Lite Mode)", combined)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"comparison_lite_{timestamp}.png", combined)
                print(f"Saved snapshot: comparison_lite_{timestamp}.png")
            elif key == ord('+') or key == ord('='):
                self.hf_process_interval = max(1, self.hf_process_interval - 1)
                print(f"HF interval: {self.hf_process_interval}")
            elif key == ord('-') or key == ord('_'):
                self.hf_process_interval += 1
                print(f"HF interval: {self.hf_process_interval}")

        self.cleanup()

    def cleanup(self):
        """Cleanup."""
        print("\nCleaning up...")
        self.pipeline.stop()
        cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Lightweight depth comparison")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model", type=str, default="Intel/dpt-hybrid-midas")
    parser.add_argument("--interval", type=int, default=5,
                       help="Process HuggingFace model every N frames")

    args = parser.parse_args()

    print("=" * 60)
    print("Depth Comparison (Lite Mode)")
    print("=" * 60)
    print(f"Camera: {args.width}x{args.height} @ {args.fps}fps")
    print(f"Model: {args.model}")
    print(f"HF Processing Interval: every {args.interval} frames")
    print("=" * 60)

    comparator = DepthComparisonLite(
        width=args.width,
        height=args.height,
        fps=args.fps,
        model_name=args.model,
        hf_process_interval=args.interval
    )

    comparator.run()


if __name__ == "__main__":
    main()
