import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import time

# ---------------------------
# Config
# ---------------------------
SAVE_DIR = "samples"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# Camera setup
# ---------------------------
cap = cv2.VideoCapture(0)  # 0 = default camera
N_samples = 20
# ---------------------------
# Tkinter setup
# ---------------------------
root = tk.Tk()
root.title("Camera Sampler")

# Label to show video frames
video_label = ttk.Label(root)
video_label.pack()

# ---------------------------
# Functions
# ---------------------------
def update_frame():
    """Grab a frame from the camera and show in the Tkinter window."""
    ret, frame = cap.read()
    if ret:
        # Convert BGR (OpenCV) → RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    # Call this function again after 10 ms
    root.after(10, update_frame)

def sample_images():
    """Take 10 photos quickly and save them with timestamped names."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for i in range(1, 1+N_samples):
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(SAVE_DIR, f"{timestamp}_{i:02d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
        root.update()  # keep UI responsive
        time.sleep(0.1)  # small delay between frames

# Button
sample_btn = ttk.Button(root, text="Sample", command=sample_images)
sample_btn.pack(pady=10)

# ---------------------------
# Start the video loop
# ---------------------------
update_frame()

# Run the Tkinter main loop
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()

