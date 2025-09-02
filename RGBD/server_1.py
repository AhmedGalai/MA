import os, cv2, numpy as np, base64, torch, tempfile, json
from fastapi import FastAPI, UploadFile, File, Query
from typing import List
from io import BytesIO
import matplotlib.pyplot as plt

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
# Feature extractors
# ---------------------------
def get_features(img, mode="sift"):
    if mode == "sift":
        return cv2.SIFT_create().detectAndCompute(img, None)
    elif mode == "orb":
        return cv2.ORB_create(5000).detectAndCompute(img, None)
    elif mode == "both":
        sift = cv2.SIFT_create(); orb = cv2.ORB_create(5000)
        kp1, des1 = sift.detectAndCompute(img, None)
        kp2, des2 = orb.detectAndCompute(img, None)
        return [(kp1, des1), (kp2, des2)]
    return None, None

# ---------------------------
# Helper: encode image to base64 PNG
# ---------------------------
def encode_image(img, cmap=None):
    buf = BytesIO()
    if len(img.shape)==2 or cmap:  # grayscale/disparity
        plt.imsave(buf, img, cmap=cmap or "gray", format="png")
    else:  # RGB
        plt.imsave(buf, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ---------------------------
# Stereo calibration + disparity
# ---------------------------
def run_calibration_and_disparity(left_imgs, right_imgs, descriptor="sift"):
    # --- Intrinsics guess
    img0 = cv2.imdecode(np.frombuffer(open(left_imgs[0],"rb").read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = img0.shape
    f = max(h, w)
    K = np.array([[f, 0, w/2],
                  [0, f, h/2],
                  [0, 0,   1]], dtype=np.float64)
    dist = np.zeros(5)

    # --- Collect matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    all_ptsL, all_ptsR = [], []

    for lpath, rpath in zip(left_imgs, right_imgs):
        imgL = cv2.imdecode(np.frombuffer(open(lpath,"rb").read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imdecode(np.frombuffer(open(rpath,"rb").read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        if descriptor == "both":
            featsL = get_features(imgL, "both")
            featsR = get_features(imgR, "both")
            combined = []
            for (kpL, desL), (kpR, desR) in zip(featsL, featsR):
                if desL is None or desR is None: continue
                matches = bf.knnMatch(desL, desR, k=2)
                good = [m for m,n in matches if m.distance < 0.75*n.distance]
                combined.extend([(kpL[m.queryIdx].pt, kpR[m.trainIdx].pt) for m in good])
            if combined:
                ptsL, ptsR = zip(*combined)
                all_ptsL.append(np.float32(ptsL))
                all_ptsR.append(np.float32(ptsR))
        else:
            kpL, desL = get_features(imgL, descriptor)
            kpR, desR = get_features(imgR, descriptor)
            if desL is None or desR is None: continue
            matches = bf.knnMatch(desL, desR, k=2)
            good = [m for m,n in matches if m.distance < 0.75*n.distance]
            ptsL = np.float32([kpL[m.queryIdx].pt for m in good])
            ptsR = np.float32([kpR[m.trainIdx].pt for m in good])
            if len(ptsL)>0:
                all_ptsL.append(ptsL); all_ptsR.append(ptsR)

    if not all_ptsL:
        return None

    all_ptsL = np.vstack(all_ptsL)
    all_ptsR = np.vstack(all_ptsR)

    # --- Pose
    E, mask = cv2.findEssentialMat(all_ptsL, all_ptsR, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, T, _ = cv2.recoverPose(E, all_ptsL, all_ptsR, K)
    _,_,_,_,Q,_,_ = cv2.stereoRectify(K,dist,K,dist,(w,h),R,T,alpha=0)

    # --- Stereo matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=16*8, blockSize=7,
        P1=8*3*7**2, P2=32*3*7**2, uniquenessRatio=10,
        speckleWindowSize=200, speckleRange=32
    )

    images_out = []
    for lpath, rpath in zip(left_imgs, right_imgs):
        imgL = cv2.imdecode(np.frombuffer(open(lpath,"rb").read(), np.uint8), cv2.IMREAD_COLOR)
        imgR = cv2.imdecode(np.frombuffer(open(rpath,"rb").read(), np.uint8), cv2.IMREAD_COLOR)
        grayL, grayR = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(K, dist, K, dist, (w,h), R, T, alpha=0)
        mapLx,mapLy = cv2.initUndistortRectifyMap(K,dist,R1,P1,(w,h),cv2.CV_32FC1)
        mapRx,mapRy = cv2.initUndistortRectifyMap(K,dist,R2,P2,(w,h),cv2.CV_32FC1)
        rectL = cv2.remap(grayL,mapLx,mapLy,cv2.INTER_LINEAR)
        rectR = cv2.remap(grayR,mapRx,mapRy,cv2.INTER_LINEAR)

        disp = stereo.compute(rectL, rectR).astype(np.float32)/16.0

        images_out.append({
            "filename": os.path.basename(lpath),
            "rgb": encode_image(imgL),
            "depth": encode_image(disp, cmap="viridis")
        })

    # --- Fake ROI mask (circle) & mesh (cube placeholder)
    mask_img = np.zeros((h,w), dtype=np.uint8)
    cv2.circle(mask_img,(w//2,h//2),min(h,w)//4,255,-1)
    mask_b64 = encode_image(mask_img)

    mesh_b64 = base64.b64encode(b"cube_mesh_placeholder").decode("utf-8")

    # --- Build JSON
    result = {
        "camera_matrix": K.tolist(),
        "images": images_out,
        "mesh": mesh_b64,
        "mask": mask_b64,
        "depthscale": 1.0
    }
    return result

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/calibrate")
async def calibrate_endpoint(
    left: List[UploadFile] = File(...),
    right: List[UploadFile] = File(...),
    descriptor: str = Query("sift", description="sift, orb, both")
):
    with tempfile.TemporaryDirectory() as tmpdir:
        left_paths, right_paths = [], []
        for i,f in enumerate(left):
            p = os.path.join(tmpdir,f"L{i}.jpg"); open(p,"wb").write(await f.read()); left_paths.append(p)
        for i,f in enumerate(right):
            p = os.path.join(tmpdir,f"R{i}.jpg"); open(p,"wb").write(await f.read()); right_paths.append(p)
        result = run_calibration_and_disparity(left_paths, right_paths, descriptor)
        if result is None:
            return {"error":"Calibration failed"}
        return result

