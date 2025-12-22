"""
Client for FoundationPose 6D pose estimation API.

This module provides functionality to send images, depth maps, masks, and mesh data
to a FoundationPose API endpoint and retrieve 6D pose estimates.
"""

import base64
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple

import cv2
import numpy as np
import requests

from config import CONFIG

logger = logging.getLogger(__name__)


def _encode_image_base64(img: np.ndarray, format: str = '.jpg') -> str:
    """
    Encode an image to base64 with data URI prefix.

    Args:
        img: Image array (HxWx3 uint8 for RGB)
        format: Image format ('.jpg' or '.png')

    Returns:
        Base64 encoded image string with data URI prefix
    """
    try:
        success, encoded = cv2.imencode(format, img)
        if not success:
            logger.error(f"Failed to encode image with format {format}")
            return ""

        # Encode directly without .tobytes() and without data URI prefix
        img_base64 = base64.b64encode(encoded).decode('utf-8')
        return img_base64

    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return ""


# def _encode_depth_as_png(depth: np.ndarray) -> str:
#     """
#     Convert depth map to disparity and encode as base64 PNG.

#     Converts depth values (in meters) to disparity using the formula:
#     disparity = 1.0 / (depth + 1e-6)

#     Then normalizes to 0-255 range and encodes as PNG for lossless compression.

#     Args:
#         depth: Depth map (HxW float32 in meters)

#     Returns:
#         Base64 encoded PNG string with data URI prefix
#     """
#     try:
#         # Ensure depth is float32
#         depth = depth.astype(np.float32)

#         # Compute disparity: 1 / depth (with small epsilon to avoid division by zero)
#         disparity = 1.0 / (depth + 1e-6)

#         # Replace inf values with 0
#         disparity[np.isinf(disparity)] = 0

#         # Normalize to 0-255 range
#         disparity_min = np.min(disparity[disparity > 0]) if np.any(disparity > 0) else 1.0
#         disparity_max = np.max(disparity)

#         if disparity_max - disparity_min < 1e-6:
#             # All values are essentially the same
#             disparity_normalized = np.zeros_like(disparity, dtype=np.uint8)
#         else:
#             disparity_normalized = (
#                 (disparity - disparity_min) / (disparity_max - disparity_min) * 255
#             ).astype(np.uint8)

#         # Encode as PNG without .tobytes() and without data URI prefix
#         success, encoded = cv2.imencode('.png', disparity_normalized)
#         if not success:
#             logger.error("Failed to encode depth as PNG")
#             return ""

#         img_base64 = base64.b64encode(encoded).decode('utf-8')
#         return img_base64

#     except Exception as e:
#         logger.error(f"Error encoding depth: {e}")
#         return ""

def _encode_depth_as_png_mm(depth_m: np.ndarray) -> str:
    """
    Encode depth (meters float32) as 16-bit PNG in millimeters.
    0 = invalid.
    """
    try:
        d = depth_m.astype(np.float32)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        depth_mm = (d * 1000.0).round().astype(np.uint16)
        # keep zeros as invalid

        success, encoded = cv2.imencode('.png', depth_mm)  # 16-bit png
        if not success:
            logger.error("Failed to encode depth as 16-bit PNG")
            return ""
        return base64.b64encode(encoded).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding depth: {e}")
        return ""



def _encode_mesh_base64(mesh_path: str) -> str:
    """
    Read and encode a .ply mesh file as base64.

    Args:
        mesh_path: Path to .ply model file

    Returns:
        Base64 encoded mesh file with data URI prefix
    """
    try:
        if not os.path.exists(mesh_path):
            logger.error(f"Mesh file not found: {mesh_path}")
            return ""

        with open(mesh_path, 'rb') as f:
            mesh_data = f.read()

        # Encode without data URI prefix
        mesh_base64 = base64.b64encode(mesh_data).decode('utf-8')
        return mesh_base64

    except Exception as e:
        logger.error(f"Error encoding mesh file: {e}")
        return ""


def estimate_pose(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    mesh_path: str,
    api_url: str
) -> Optional[np.ndarray]:
    """
    Estimate 6D pose using FoundationPose API.

    Sends RGB image, depth map, mask, camera intrinsics, and mesh to the
    FoundationPose API endpoint and retrieves the estimated 4x4 transformation
    matrix.

    Args:
        rgb: RGB image (HxWx3 uint8)
        depth: Depth map (HxW float32 in meters)
        mask: Binary mask (HxW uint8, 0 or 255)
        K: Camera intrinsics (3x3 matrix)
        mesh_path: Path to .ply model file
        api_url: FoundationPose API endpoint URL

    Returns:
        4x4 transformation matrix as numpy array, or None on failure
    """
    try:
        # Validate inputs
        if rgb is None or rgb.size == 0:
            logger.error("RGB image is empty or None")
            return None

        if depth is None or depth.size == 0:
            logger.error("Depth map is empty or None")
            return None

        if mask is None or mask.size == 0:
            logger.error("Mask is empty or None")
            return None

        if K is None or K.shape != (3, 3):
            logger.error(f"Invalid camera matrix shape: {K.shape if K is not None else None}")
            return None

        # Ensure proper data types
        rgb = rgb.astype(np.uint8)
        depth = depth.astype(np.float32)
        mask = mask.astype(np.uint8)

        # Encode images and mesh
        logger.debug("Encoding RGB image...")
        # rgb_b64 = _encode_image_base64(rgb, format='.jpg')
        rgb_b64 = _encode_image_base64(rgb, format='.png')

        if not rgb_b64:
            logger.error("Failed to encode RGB image")
            return None

        logger.debug("Encoding depth map...")
        # depth_b64 = _encode_depth_as_png(depth)
        depth_b64 = _encode_depth_as_png_mm(depth)

        if not depth_b64:
            logger.error("Failed to encode depth map")
            return None

        logger.debug("Encoding mask...")
        mask_b64 = _encode_image_base64(mask, format='.png')
        if not mask_b64:
            logger.error("Failed to encode mask")
            return None

        logger.debug("Encoding mesh file...")
        mesh_b64 = _encode_mesh_base64(mesh_path)
        if not mesh_b64:
            logger.error("Failed to encode mesh file")
            return None

        # Construct JSON payload
        payload = {
            "camera_matrix": K.tolist(),
            "images": [
                {
                    "filename": "frame_rs",
                    "rgb": rgb_b64,
                    "depth": depth_b64
                }
            ],
            "mask": mask_b64,
            "mesh": mesh_b64,
            "sequence": False,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Send POST request
        logger.debug(f"Sending request to {api_url}...")
        response = requests.post(
            api_url,
            json=payload,
            timeout=30
        )

        # Check response status
        if response.status_code != 200:
            logger.error(
                f"API returned status {response.status_code}: {response.text}"
            )
            return None

        # Parse response JSON
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response as JSON: {e}")
            logger.error(f"Response text: {response.text}")
            return None

        # Validate response structure
        if not isinstance(result, dict):
            logger.error(f"Expected dict response, got {type(result)}")
            return None

        if "transformation_matrix" not in result:
            logger.error(f"Missing 'transformation_matrix' in response: {result.keys()}")
            return None

        # Extract and validate transformation matrix
        try:
            T = np.array(result["transformation_matrix"], dtype=np.float32)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert transformation_matrix to array: {e}")
            return None

        # Handle batch dimension if present (squeeze from (1, 4, 4) to (4, 4))
        if T.shape == (1, 4, 4):
            T = T.squeeze(0)

        # Validate shape
        if T.shape != (4, 4):
            logger.error(
                f"Invalid transformation matrix shape: {T.shape}, expected (4, 4)"
            )
            return None

        logger.debug("Successfully estimated pose")
        return T

    except requests.exceptions.Timeout:
        logger.error("Request to FoundationPose API timed out (30s)")
        return None

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to FoundationPose API: {e}")
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error in estimate_pose: {e}", exc_info=True)
        return None
