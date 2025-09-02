import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------
# Globals
# -----------------------------
cap = cv2.VideoCapture(0)
sampling = False
last_frame = None
disp_img = None
sampling_rate = 2.0  # Hz

# Matplotlib figure
plt.ion()
fig, ax = plt.subplots()

# -----------------------------
# Tkinter setup
# -----------------------------
root = tk.Tk()
root.title("Monocular Depth from Motion")

video_label = ttk.Label(root)
video_label.grid(row=0, column=0, columnspan=3)

ttk.Label(root, text="Sampling Rate (Hz):").grid(row=1, column=0, sticky="e")
rate_entry = ttk.Entry(root, width=5)
rate_entry.insert(0, "2")
rate_entry.grid(row=1, column=1, sticky="w")

# -----------------------------
# Camera preview
# -----------------------------
def update_preview():
    if not sampling:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
    root.after(30, update_preview)

# -----------------------------
# Depth estimation
# -----------------------------
def process_frame(frame):
    global last_frame, disp_img

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if last_frame is None:
        last_frame = gray
        return

    # ORB features
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(last_frame, None)
    kp2, des2 = orb.detectAndCompute(gray, None)
    if des1 is None or des2 is None:
        last_frame = gray
        return

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 20:
        last_frame = gray
        return

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Parallax magnitude as pseudo-depth
    parallax = np.linalg.norm(pts1 - pts2, axis=1)
    depth_map = np.zeros_like(gray, dtype=np.float32)
    for (x,y), d in zip(pts1.astype(int), parallax):
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            depth_map[y, x] = d

    disp_img = cv2.GaussianBlur(depth_map, (15,15), 0)

    # Update matplotlib
    ax.clear()
    ax.imshow(disp_img, cmap="plasma")
    ax.set_title("Depth from Motion (relative)")
    ax.axis("off")
    fig.canvas.draw()
    fig.canvas.flush_events()

    last_frame = gray

# -----------------------------
# Sampling loop (Tkinter after)
# -----------------------------
def sample_step():
    global sampling
    if not sampling:
        return
    ret, frame = cap.read()
    if ret:
        process_frame(frame)
    delay = int(1000 / sampling_rate)  # ms
    root.after(delay, sample_step)

def start_sampling():
    global sampling, sampling_rate, last_frame
    try:
        sampling_rate = float(rate_entry.get())
    except:
        sampling_rate = 2.0
    last_frame = None
    sampling = True
    sample_step()  # schedule first step

def stop_sampling():
    global sampling
    sampling = False

# -----------------------------
# Buttons
# -----------------------------
ttk.Button(root, text="Start Sampling", command=start_sampling).grid(row=2, column=0, pady=5)
ttk.Button(root, text="Stop Sampling", command=stop_sampling).grid(row=2, column=1, pady=5)

# -----------------------------
# Start
# -----------------------------
update_preview()
root.mainloop()
cap.release()
cv2.destroyAllWindows()

