import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

SAMPLES_DIR = "samples"

# ----------------------
# Load images
# ----------------------
all_imgs = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.jpg")))
assert len(all_imgs) >= 20, "Need at least 20 images!"

N_samples = 20
left_imgs = all_imgs[:N_samples//2]
right_imgs = all_imgs[N_samples//2:N_samples]

print("Left set:", left_imgs)
print("Right set:", right_imgs)

# ----------------------
# Approx intrinsics (pinhole guess)
sample_img = cv2.imread(left_imgs[0], cv2.IMREAD_GRAYSCALE)
h, w = sample_img.shape
f = max(h, w)
K = np.array([[f, 0, w/2],
              [0, f, h/2],
              [0, 0, 1]])
dist = np.zeros(5)

# ----------------------
# Feature matcher setup
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

all_ptsL, all_ptsR = [], []

for i, (left_path, right_path) in enumerate(zip(left_imgs, right_imgs)):
    imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    kpL, desL = sift.detectAndCompute(imgL, None)
    kpR, desR = sift.detectAndCompute(imgR, None)
    if desL is None or desR is None:
        print(f"Skipping pair {i}, no descriptors found")
        continue

    matches = bf.match(desL, desR)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]  # top N

    # Collect points for calibration
    ptsL = np.float32([kpL[m.queryIdx].pt for m in good_matches])
    ptsR = np.float32([kpR[m.trainIdx].pt for m in good_matches])
    all_ptsL.append(ptsL)
    all_ptsR.append(ptsR)

    # --- Show match visualization ---
    match_vis = cv2.drawMatches(imgL, kpL, imgR, kpR, good_matches, None, flags=2)
    cv2.imshow(f"Pair {i} Matches", match_vis)
    cv2.waitKey(500)  # show for 0.5 sec

cv2.destroyAllWindows()

# Concatenate points
if len(all_ptsL) == 0:
    raise RuntimeError("No good matches collected across all pairs!")
all_ptsL = np.vstack(all_ptsL)
all_ptsR = np.vstack(all_ptsR)
print(f"Collected {len(all_ptsL)} total correspondences.")

# ----------------------
# Essential matrix & pose
E, mask = cv2.findEssentialMat(all_ptsL, all_ptsR, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, T, mask_pose = cv2.recoverPose(E, all_ptsL, all_ptsR, K)

print("\n=== Stereo Calibration (multi-image) ===")
print("Rotation (R):\n", R)
print("Translation (T):\n", T.ravel())

# ----------------------
# Rectify first pair as reference
imgL = cv2.imread(left_imgs[0], cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(right_imgs[0], cv2.IMREAD_GRAYSCALE)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K, dist, K, dist, (w, h), R, T, alpha=0
)

mapLx, mapLy = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w, h), cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w, h), cv2.CV_32FC1)

rectifiedL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
rectifiedR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

# ----------------------
# Stereo matching
window_size = 7
min_disp = 0
num_disp = 16*8  # must be divisible by 16

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=32
)

disparity = stereo.compute(rectifiedL, rectifiedR).astype(np.float32) / 16.0

# ----------------------
# Show rectified reference + disparity
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(rectifiedL, cmap="gray")
axes[0].set_title("Reference Image (Rectified Left)")
axes[0].axis("off")

im = axes[1].imshow(disparity, cmap="viridis")
axes[1].set_title("Disparity Map")
axes[1].axis("off")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

plt.show()

