# mock_pose_api.py
# pip install fastapi uvicorn numpy

import math, time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# ---------- Schemas ----------
class ImageItem(BaseModel):
    filename: str
    rgb: str
    depth: str

class PoseRequest(BaseModel):
    camera_matrix: List[List[float]]
    images: List[ImageItem]
    mesh: str
    mask: Optional[str] = None
    depthscale: float

# ---------- App ----------
app = FastAPI(title="Mock Pose API", version="4.0")

# Pose generator settings
Z_DIST = 10.0          # "far from the camera" along +Z
YAW_DPS = 25.0         # deg/sec around Z
PITCH_DPS = 60.0       # deg/sec around Y
t0 = time.perf_counter()

def Rz(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[ c,-s, 0.0],
                     [ s, c, 0.0],
                     [0.0,0.0, 1.0]], dtype=float)

def Ry(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[ c, 0.0, s],
                     [0.0, 1.0,0.0],
                     [-s, 0.0, c]], dtype=float)

def pose_at_time(t: float) -> np.ndarray:
    """
    Deterministic pose:
      - centered in view: x=y=0
      - far from camera: z=Z_DIST
      - rotation: yaw (Z) and pitch (Y) at different angular rates
    """
    yaw   = math.radians(YAW_DPS   * t)
    pitch = math.radians(PITCH_DPS * t)
    R = Rz(yaw) @ Ry(pitch)            # intrinsic order: pitch then yaw
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = np.array([0.0, 0.0, Z_DIST], dtype=float)
    return T

@app.post("/pose")
def pose(req: PoseRequest):
    # Time-based "generator" pose (smooth, no randomness)
    t = time.perf_counter() - t0
    T = pose_at_time(t)
    return {
        "status": "Pose generator (centered, far, 2-axis rotation)",
        "transformation_matrix": [T.tolist()],
        "debug": {
            "t_seconds": t,
            "yaw_deg":  YAW_DPS * t,
            "pitch_deg": PITCH_DPS * t,
            "z_dist": Z_DIST
        }
    }
