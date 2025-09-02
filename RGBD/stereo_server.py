import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from typing import List
import torch  # optional for GPU check
import tempfile
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = FastAPI()

# ---------------------------
# GPU check
# ---------------------------
def has_gpu():
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0 or torch.cuda.is_available()
    except:
        return False

# ---------------------------
# Calibration + disparity
# ---------------------------
def run_calibration_and_disparity(left_imgs, right_imgs):
    # Assume intrinsics (pinhole)
    img0 = cv2.imdecode(np.frombuffer(open(left_imgs[0],"rb").read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = img0.shape
    f = max(h, w)
    K = np.array([[f, 0, w/2],
                  [0, f, h/2],
                  [0, 0,   1]], dtype=np.float64)
    dist = np.zeros(5)

    # Use SIFT
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    all_ptsL, all_ptsR = [], []
    for lpath, rpath in zip(left_imgs, right_imgs):
        imgL = cv2.imdecode(np.frombuffer(open(lpath,"rb").read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imdecode(np.frombuffer(open(rpath,"rb").read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        kpL, desL = sift.detectAndCompute(imgL, None)
        kpR, desR = sift.detectAndCompute(imgR, None)
        if desL is None or desR is None: continue
        matches = bf.knnMatch(desL, desR, k=2)
        good = [m for m,n in matches if m.distance < 0.75*n.distance]
        ptsL = np.float32([kpL[m.queryIdx].pt for m in good])
        ptsR = np.float32([kpR[m.trainIdx].pt for m in good])
        if len(ptsL)>0:
            all_ptsL.append(ptsL)
            all_ptsR.append(ptsR)

    if not all_ptsL:
        return None

    all_ptsL = np.vstack(all_ptsL)
    all_ptsR = np.vstack(all_ptsR)

    E, mask = cv2.findEssentialMat(all_ptsL, all_ptsR, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, T, mask_pose = cv2.recoverPose(E, all_ptsL, all_ptsR, K)

    # Rectify with first pair
    imgL = cv2.imdecode(np.frombuffer(open(left_imgs[0],"rb").read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imdecode(np.frombuffer(open(right_imgs[0],"rb").read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(K,dist,K,dist,(w,h),R,T,alpha=0)
    mapLx,mapLy = cv2.initUndistortRectifyMap(K,dist,R1,P1,(w,h),cv2.CV_32FC1)
    mapRx,mapRy = cv2.initUndistortRectifyMap(K,dist,R2,P2,(w,h),cv2.CV_32FC1)
    rectL = cv2.remap(imgL,mapLx,mapLy,cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR,mapRx,mapRy,cv2.INTER_LINEAR)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*8,
        blockSize=7,
        P1=8*3*7**2,
        P2=32*3*7**2,
        uniquenessRatio=10,
        speckleWindowSize=200,
        speckleRange=32
    )
    disp = stereo.compute(rectL, rectR).astype(np.float32)/16.0

    # Encode disparity as PNG (base64)
    plt.figure(figsize=(6,5))
    plt.imshow(disp, cmap="viridis")
    plt.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    disp_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "K": K.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "Q": Q.tolist(),
        "gpu": has_gpu(),
        "disparity_map": disp_b64
    }

# ---------------------------
# API Endpoints
# ---------------------------
@app.post("/calibrate")
async def calibrate_endpoint(left: List[UploadFile] = File(...), right: List[UploadFile] = File(...)):
    with tempfile.TemporaryDirectory() as tmpdir:
        left_paths, right_paths = [], []
        for i,file in enumerate(left):
            path = os.path.join(tmpdir, f"L{i}.jpg")
            with open(path,"wb") as f: f.write(await file.read())
            left_paths.append(path)
        for i,file in enumerate(right):
            path = os.path.join(tmpdir, f"R{i}.jpg")
            with open(path,"wb") as f: f.write(await file.read())
            right_paths.append(path)

        result = run_calibration_and_disparity(left_paths, right_paths)
        if result is None:
            return {"error":"Calibration failed"}
        return result

