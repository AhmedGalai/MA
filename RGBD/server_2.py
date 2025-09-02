import os, cv2, numpy as np, base64, tempfile
from fastapi import FastAPI, Request
from io import BytesIO
import matplotlib.pyplot as plt
import json

app = FastAPI()

# ---------------------------
# Helper: encode image to base64 PNG
# ---------------------------
def encode_image(img, cmap=None):
    buf = BytesIO()
    if len(img.shape)==2 or cmap:
        plt.imsave(buf, img, cmap=cmap or "gray", format="png")
    else:
        plt.imsave(buf, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ---------------------------
# Stereo rectification + disparity
# ---------------------------
def process_stereo(imgL, imgR):
    grayL, grayR = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    h, w = grayL.shape

    # --- Simple intrinsics
    f = max(h, w)
    K = np.array([[f, 0, w/2],
                  [0, f, h/2],
                  [0, 0,   1]], dtype=np.float64)
    dist = np.zeros(5)

    # --- ORB features + Essential matrix
    orb = cv2.ORB_create(5000)
    kpL, desL = orb.detectAndCompute(grayL, None)
    kpR, desR = orb.detectAndCompute(grayR, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desL, desR)
    matches = sorted(matches, key=lambda x: x.distance)[:200]

    ptsL = np.float32([kpL[m.queryIdx].pt for m in matches])
    ptsR = np.float32([kpR[m.trainIdx].pt for m in matches])

    E, _ = cv2.findEssentialMat(ptsL, ptsR, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, T, _ = cv2.recoverPose(E, ptsL, ptsR, K)

    # --- Rectification
    R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(K, dist, K, dist, (w,h), R, T, alpha=0)
    mapLx,mapLy = cv2.initUndistortRectifyMap(K,dist,R1,P1,(w,h),cv2.CV_32FC1)
    mapRx,mapRy = cv2.initUndistortRectifyMap(K,dist,R2,P2,(w,h),cv2.CV_32FC1)
    rectL = cv2.remap(grayL,mapLx,mapLy,cv2.INTER_LINEAR)
    rectR = cv2.remap(grayR,mapRx,mapRy,cv2.INTER_LINEAR)

    # --- Disparity
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=16*6, blockSize=7,
        P1=8*3*7**2, P2=32*3*7**2,
        uniquenessRatio=10, speckleWindowSize=200, speckleRange=32
    )
    disp = stereo.compute(rectL, rectR).astype(np.float32)/16.0

    return encode_image(rectL), encode_image(disp, cmap="viridis"), K, R, T

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/stereo")
async def stereo_endpoint(request: Request):
    data = await request.json()
    print("Received request keys:", list(data.keys()))  # logging

    left_b64 = data["leftImage"]
    right_b64 = data["rightImage"]

    left_bytes = base64.b64decode(left_b64)
    right_bytes = base64.b64decode(right_b64)

    imgL = cv2.imdecode(np.frombuffer(left_bytes, np.uint8), cv2.IMREAD_COLOR)
    imgR = cv2.imdecode(np.frombuffer(right_bytes, np.uint8), cv2.IMREAD_COLOR)

    rectified, disparity, K, R, T = process_stereo(imgL, imgR)

    return {
        "rectifiedLeft": rectified,
        "disparity": disparity,
        "K": K.tolist(),
        "R": R.tolist(),
        "T": T.tolist()
    }

