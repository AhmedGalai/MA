"""
Monocular Depth Estimation using DepthAnythingV2
"""

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
from PIL import Image

class DepthAnythingV2:
    """
    A wrapper for the DepthAnythingV2 model for monocular depth estimation.
    """

    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        """
        Initializes the DepthAnythingV2 model and processor.

        Args:
            model_name (str): The name of the model to load from Hugging Face.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DepthAnythingV2] Using device: {self.device}")

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
        print(f"[DepthAnythingV2] Model '{model_name}' loaded.")

    def estimate_depth(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Estimates the depth map from a single RGB image.

        Args:
            rgb_image (np.ndarray): The input RGB image in (H, W, 3) format.

        Returns:
            np.ndarray: The estimated depth map, normalized to 0-1.
        """
        # Convert numpy array to PIL image
        image = Image.fromarray(rgb_image)

        # Prepare image for the model
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Move to CPU and convert to numpy array
        output = prediction.cpu().numpy()

        # Normalize the output
        output = (output - output.min()) / (output.max() - output.min())

        return output

if __name__ == '__main__':
    # Example usage
    # Create a dummy RGB image
    dummy_rgb = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

    # Initialize the depth estimator
    depth_estimator = DepthAnythingV2()

    # Estimate depth
    estimated_depth = depth_estimator.estimate_depth(dummy_rgb)

    print(f"Estimated depth map shape: {estimated_depth.shape}")
    print(f"Estimated depth map min value: {estimated_depth.min()}")
    print(f"Estimated depth map max value: {estimated_depth.max()}")

    # To visualize the depth map, you can use matplotlib or opencv
    import cv2 as cv
    depth_vis = (estimated_depth * 255).astype(np.uint8)
    depth_vis_color = cv.applyColorMap(depth_vis, cv.COLORMAP_INFERNO)
    cv.imshow("Depth Map", depth_vis_color)
    cv.waitKey(0)
    cv.destroyAllWindows()
