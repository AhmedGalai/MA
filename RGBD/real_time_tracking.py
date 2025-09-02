import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# -----------------------------
# Load stereo images
# -----------------------------
SAMPLES_DIR = "samples"
all_imgs = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.jpg")))
assert len(all_imgs) >= 50, "Need at least 50 samples!"

# Split into left/right sets (first half = left, second half = right)
N_samples = 2*50
left_imgs = all_imgs[:N_samples//2]
right_imgs = all_imgs[N_samples//2:N_samples]

# Reference: use the 24th pair (index 24 = 25th image, since 0-based)
idx_ref = len(left_imgs) - 2  # 24
imgL_ref = cv2.imread(left_imgs[idx_ref], cv2.IMREAD_GRAYSCALE)
imgR_ref = cv2.imread(right_imgs[idx_ref], cv2.IMREAD_GRAYSCALE)
h, w = imgL_ref.shape

# Camera intrinsics (approx pinhole)
f = max(h, w)
K = np.array([[f, 0, w/2],
              [0, f, h/2],
              [0, 0,   1]], dtype=np.float64)

# Stereo disparity → depth for frame t
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
disp_ref = stereo.compute(imgL_ref, imgR_ref).astype(np.float32)/16.0
baseline = 0.1  # assume 10cm baseline
depth_ref = f * baseline / (disp_ref + 1e-6)

# -----------------------------
# Next RGB(t+1): last left image
# -----------------------------
img_t = cv2.imread(left_imgs[idx_ref], cv2.IMREAD_GRAYSCALE)      # frame t (49th left)
img_tp1 = cv2.imread(left_imgs[idx_ref+1], cv2.IMREAD_GRAYSCALE)  # frame t+1 (50th left)

# -----------------------------
# Feature matching between t and t+1
# -----------------------------
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_t, None)
kp2, des2 = sift.detectAndCompute(img_tp1, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = [m for m,n in matches if m.distance < 0.75*n.distance]

pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

# Estimate relative motion
E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, T, _ = cv2.recoverPose(E, pts1, pts2, K)

# -----------------------------
# Warp depth map from t → t+1
# -----------------------------
u, v = np.meshgrid(np.arange(w), np.arange(h))
Z = depth_ref
X = (u - K[0,2]) * Z / K[0,0]
Y = (v - K[1,2]) * Z / K[1,1]
pts3d = np.stack((X,Y,Z), axis=-1).reshape(-1,3).T  # (3,N)

# transform to new frame
pts3d_new = R @ pts3d + T

# project back
u2 = (K[0,0]*pts3d_new[0]/pts3d_new[2] + K[0,2]).astype(int)
v2 = (K[1,1]*pts3d_new[1]/pts3d_new[2] + K[1,2]).astype(int)
depth_new = np.zeros_like(depth_ref)
mask = (u2>=0)&(u2<w)&(v2>=0)&(v2<h)&(pts3d_new[2]>0)
depth_new[v2[mask], u2[mask]] = pts3d_new[2][mask]

#import cv2.ximgproc as xip
#wls = xip.createDisparityWLSFilter(stereo)
#right_matcher = cv2.ximgproc.createRightMatcher(stereo)
#dispR = right_matcher.compute(imgR_ref, imgL_ref)
#disp_filtered = wls.filter(depth_ref, imgL_ref, None, dispR)
disp_smooth = cv2.morphologyEx(depth_ref, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
# -----------------------------
# Show results
# -----------------------------
plt.figure(figsize=(12,8))
plt.subplot(3,2,1)
plt.imshow(img_t, cmap="gray"); plt.title("RGB(t=49)")
plt.axis("off")
vmin, vmax = np.percentile(depth_ref, [5, 95])  # robust min/max
plt.subplot(3,2,2)
#plt.imshow(depth_ref, cmap="viridis"); plt.title("Depth(t=49)")
plt.imshow(depth_ref, cmap="viridis", vmin=vmin, vmax=vmax)
plt.axis("off")

plt.subplot(3,2,3)
plt.imshow(img_tp1, cmap="gray"); plt.title("RGB(t+1=50)")
#plt.imshow(img_tp1, cmap="viridis", vmin=vmin, vmax=vmax)
plt.axis("off")

plt.subplot(3,2,4)
plt.imshow(depth_new, cmap="viridis"); plt.title("Estimated Depth(t+1=50)")
plt.axis("off")

plt.subplot(3,2,5)
plt.imshow(disp_smooth, cmap="viridis"); plt.title("Estimated Depth(t+1=50) filtered")
plt.axis("off")

plt.tight_layout()
plt.show()

