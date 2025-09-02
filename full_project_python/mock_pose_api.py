# mock_pose_api.py
# pip install fastapi uvicorn numpy

import math, random
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
app = FastAPI(title="Mock Pose API", version="3.0")

def random_pose():
    """Generate a random SE(3) pose matrix (4x4)."""
    # Random Euler angles
    ang_x = random.uniform(-math.pi/6, math.pi/6)  # ±30°
    ang_y = random.uniform(-math.pi/6, math.pi/6)
    ang_z = random.uniform(-math.pi/6, math.pi/6)

    Rx = np.array([[1,0,0],
                   [0,math.cos(ang_x),-math.sin(ang_x)],
                   [0,math.sin(ang_x), math.cos(ang_x)]])
    Ry = np.array([[math.cos(ang_y),0,math.sin(ang_y)],
                   [0,1,0],
                   [-math.sin(ang_y),0,math.cos(ang_y)]])
    Rz = np.array([[math.cos(ang_z),-math.sin(ang_z),0],
                   [math.sin(ang_z), math.cos(ang_z),0],
                   [0,0,1]])

    R = Rz @ Ry @ Rx
    # Random translation in a small cube
    t = np.array([random.uniform(-0.1,0.1),
                  random.uniform(-0.05,0.05),
                  random.uniform(0.5,0.7)])  # Z positive

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

@app.post("/pose")
def pose(req: PoseRequest):
    # Return a fresh random pose each call
    T = random_pose()
    return {
        "status": "Pose estimation complete",
        "transformation_matrix": [T.tolist()],
    }
