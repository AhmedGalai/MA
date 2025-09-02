import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import Text
from PIL import Image, ImageTk
import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt

# ---------------------------
# Globals
# ---------------------------
SAVE_DIR = "samples"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
left_set, right_set = [], []   # filenames
K, R, T, Q = None, None, None, None

# ---------------------------
# Tkinter setup
# ---------------------------
root = tk.Tk()
root.title("RGB-D Sampler")

# Video preview
video_label = ttk.Label(root)
video_label.grid(row=0, column=0, columnspan=3)

# Controls
n_label = ttk.Label(root, text="N_samples:")
n_label.grid(row=1, column=0, sticky="e")
n_entry = ttk.Entry(root, width=5)
n_entry.insert(0, "10")
n_entry.grid(row=1, column=1, sticky="w")

# Log window
log_text = Text(root, height=10, width=60)
log_text.grid(row=2, column=0, columnspan=3, pady=5)

def log(msg):
    log_text.insert(tk.END, msg + "\n")
    log_text.see(tk.END)
    root.update()

# ---------------------------
# Camera update
# ---------------------------
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    root.after(10, update_frame)

# ---------------------------
# Sampling functions
# ---------------------------
def sample_images(set_type):
    global left_set, right_set
    N_samples = int(n_entry.get())
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    filelist = []
    for i in range(1, N_samples+1):
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(SAVE_DIR, f"{timestamp}_{set_type}_{i:02d}.jpg")
            cv2.imwrite(filename, frame)
            filelist.append(filename)
            log(f"Saved {filename}")
        root.update()
        time.sleep(0.1)

    if set_type == "L":
        left_set = filelist
    else:
        right_set = filelist

# ---------------------------
# Calibration
# ---------------------------
def calibrate():
    global K, R, T, Q
    if not left_set or not right_set:
        log("Capture both Left and Right sets first!")
        return

    sample_img = cv2.imread(left_set[0], cv2.IMREAD_GRAYSCALE)
    h, w = sample_img.shape
    f = max(h, w)
    K = np.array([[f, 0, w/2],
                  [0, f, h/2],
                  [0, 0,   1]])
    dist = np.zeros(5)

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    all_ptsL, all_ptsR = [], []

    cv2.namedWindow("Calibration Matches", cv2.WINDOW_NORMAL)

    for left_path, right_path in zip(left_set, right_set):
        imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        kpL, desL = sift.detectAndCompute(imgL, None)
        kpR, desR = sift.detectAndCompute(imgR, None)
        if desL is None or desR is None:
            continue

        matches = bf.match(desL, desR)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]

        ptsL = np.float32([kpL[m.queryIdx].pt for m in good_matches])
        ptsR = np.float32([kpR[m.trainIdx].pt for m in good_matches])
        all_ptsL.append(ptsL)
        all_ptsR.append(ptsR)

        # update window
        match_vis = cv2.drawMatches(imgL, kpL, imgR, kpR, good_matches, None, flags=2)
        cv2.imshow("Calibration Matches", match_vis)
        cv2.waitKey(100)

    cv2.destroyWindow("Calibration Matches")

    if not all_ptsL:
        log("Calibration failed: no matches!")
        return

    all_ptsL = np.vstack(all_ptsL)
    all_ptsR = np.vstack(all_ptsR)

    E, mask = cv2.findEssentialMat(all_ptsL, all_ptsR, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, T, mask_pose = cv2.recoverPose(E, all_ptsL, all_ptsR, K)

    log("\n=== Stereo Calibration ===")
    log(f"R:\n{R}")
    log(f"T:\n{T.ravel()}")

    # Rectify maps
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, dist, K, dist, (w, h), R, T, alpha=0
    )

    log("Calibration done.")

# ---------------------------
# Disparity estimation
# ---------------------------
def estimate_disparity():
    global K, R, T, Q
    if K is None or R is None:
        log("Run calibration first!")
        return

    # Take first pair
    imgL_raw = cv2.imread(left_set[0])
    imgR_raw = cv2.imread(right_set[0])
    imgL = cv2.cvtColor(imgL_raw, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR_raw, cv2.COLOR_BGR2GRAY)
    h, w = imgL.shape

    dist = np.zeros(5)

    # Rectify
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, dist, K, dist, (w, h), R, T, alpha=0
    )
    mapLx, mapLy = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w, h), cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w, h), cv2.CV_32FC1)

    rectifiedL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    # Stereo matching
    window_size = 7
    num_disp = 16*8
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
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

    # Show raw RGB, rectified gray, disparity
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(imgL_raw, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Raw RGB (Left)")
    axes[0].axis("off")

    axes[1].imshow(rectifiedL, cmap="gray")
    axes[1].set_title("Rectified Left (Gray)")
    axes[1].axis("off")

    im = axes[2].imshow(disparity, cmap="viridis")
    axes[2].set_title("Disparity Map")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.show()

# ---------------------------
# Buttons
# ---------------------------
btn_left = ttk.Button(root, text="Sample Left", command=lambda: sample_images("L"))
btn_left.grid(row=3, column=0, pady=5)

btn_right = ttk.Button(root, text="Sample Right", command=lambda: sample_images("R"))
btn_right.grid(row=3, column=1, pady=5)

btn_calib = ttk.Button(root, text="Calibrate", command=calibrate)
btn_calib.grid(row=3, column=2, pady=5)

btn_disp = ttk.Button(root, text="Estimate Disparity", command=estimate_disparity)
btn_disp.grid(row=4, column=1, pady=10)

# ---------------------------
# Start
# ---------------------------
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()

