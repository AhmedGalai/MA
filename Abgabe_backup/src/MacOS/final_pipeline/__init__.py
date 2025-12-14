"""
Final Pipeline - Cleaned Depth Estimation with Pose Correction

A streamlined pipeline for 6D pose estimation using:
- RealSense depth camera (fixed)
- Continuous headset pose streaming
- ArUco-based calibration
- Probabilistic pose correction
"""

from .pipeline_core import FinalPipeline

__version__ = "1.0.0"
__all__ = ["FinalPipeline"]
