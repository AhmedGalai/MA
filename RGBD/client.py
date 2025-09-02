import cv2
import tkinter as tk
from tkinter import ttk, Text
from PIL import Image, ImageTk
import os, time, requests, base64
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
SAVE_DIR = "samples"
os.makedirs(SAVE_DIR, exist_ok=True)

API_URL = "http://localhost:8000/calibrate"   # FastAPI server endpoint
N_samples = 10   # default

cap = cv2.VideoCapture(0)
left_set, right_set = [], []

# ---------------------------
# Tkinter setup
# ---------------------------
root = tk.Tk()
root.title("RGB-D Client Sampler")

# Video preview
video_label = ttk.Label(root)
video_label.grid(row=0, column=0, columnspan=3)

# Log window
log_text = Text(root, height=12, width=60)
log_text.grid(row=1, column=0, columnspan=3, pady=5)

def log(msg):
    log_text.insert(tk.END, msg + "\n")
    log_text.see(tk.END)
    root.update()

# ---------------------------
# Camera preview loop
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
# Sampling
# ---------------------------
def sample_images(set_type):
    global left_set, right_set
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
# Send to API
# ---------------------------
def send_to_api():
    global left_set, right_set
    if not left_set or not right_set:
        log("Error: Capture both Left and Right sets first!")
        return

    log("Sending images to API...")
    files = []
    for path in left_set:
        files.append(("left", (os.path.basename(path), open(path, "rb"), "image/jpeg")))
    for path in right_set:
        files.append(("right", (os.path.basename(path), open(path, "rb"), "image/jpeg")))

    r = requests.post(API_URL, files=files)
    if r.status_code != 200:
        log(f"API error: {r.status_code}")
        return

    data = r.json()
    if "error" in data:
        log("Calibration failed: " + data["error"])
        return

    log("\n=== Calibration Results ===")
    log("GPU: " + str(data["gpu"]))
    log("K: " + str(data["K"]))
    log("R: " + str(data["R"]))
    log("T: " + str(data["T"]))

    # Save disparity image
    disp_b64 = data["disparity_map"]
    disparity_bytes = base64.b64decode(disp_b64)
    disp_path = os.path.join(SAVE_DIR, "disparity.png")
    with open(disp_path, "wb") as f:
        f.write(disparity_bytes)
    log(f"Saved disparity image to {disp_path}")

    # Show disparity image
    img = plt.imread(disp_path)
    plt.imshow(img)
    plt.title("Disparity Map from API")
    plt.axis("off")
    plt.show()

# ---------------------------
# Buttons
# ---------------------------
btn_left = ttk.Button(root, text="Sample Left", command=lambda: sample_images("L"))
btn_left.grid(row=2, column=0, pady=5)

btn_right = ttk.Button(root, text="Sample Right", command=lambda: sample_images("R"))
btn_right.grid(row=2, column=1, pady=5)

btn_send = ttk.Button(root, text="Send to API", command=send_to_api)
btn_send.grid(row=2, column=2, pady=5)

# ---------------------------
# Start
# ---------------------------
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()

