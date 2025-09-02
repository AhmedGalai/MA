import cv2, os, time, requests, base64, io, json
import tkinter as tk
from tkinter import ttk, Text
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

SAVE_DIR="samples"; os.makedirs(SAVE_DIR,exist_ok=True)
API_URL="http://localhost:8000/calibrate"
cap=cv2.VideoCapture(0)
left_set,right_set=[],[]

root=tk.Tk(); root.title("RGB-D Client")

video_label=ttk.Label(root); video_label.grid(row=0,column=0,columnspan=3)
log_text=Text(root,height=12,width=60); log_text.grid(row=1,column=0,columnspan=3,pady=5)
def log(msg): log_text.insert(tk.END,msg+"\n"); log_text.see(tk.END); root.update()

# Descriptor selection
descriptor_var=tk.StringVar(value="sift")
ttk.Label(root,text="Descriptor:").grid(row=2,column=0)
for idx,mode in enumerate(["sift","orb","both"]):
    ttk.Radiobutton(root,text=mode.upper(),variable=descriptor_var,value=mode).grid(row=2,column=idx+1)

# N_samples input
ttk.Label(root,text="N_samples:").grid(row=3,column=0)
n_entry=ttk.Entry(root,width=5); n_entry.insert(0,"50"); n_entry.grid(row=3,column=1)

def update_frame():
    ret,frame=cap.read()
    if ret:
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        imgtk=ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        video_label.imgtk=imgtk; video_label.configure(image=imgtk)
    root.after(10,update_frame)

def sample_images(set_type):
    global left_set,right_set
    N=int(n_entry.get()); ts=time.strftime("%Y%m%d_%H%M%S"); files=[]
    for i in range(1,N+1):
        ret,frame=cap.read()
        if ret:
            fn=os.path.join(SAVE_DIR,f"{ts}_{set_type}_{i:02d}.jpg")
            cv2.imwrite(fn,frame); files.append(fn); log(f"Saved {fn}")
        root.update(); time.sleep(0.05)
    if set_type=="L": left_set=files
    else: right_set=files

def send_to_api():
    if not left_set or not right_set: log("Capture Left/Right first!"); return
    files=[]
    for p in left_set: files.append(("left",(os.path.basename(p),open(p,"rb"),"image/jpeg")))
    for p in right_set: files.append(("right",(os.path.basename(p),open(p,"rb"),"image/jpeg")))
    params={"descriptor":descriptor_var.get()}
    log(f"Sending with descriptor={descriptor_var.get()}...")
    r=requests.post(API_URL,files=files,params=params)
    if r.status_code!=200: log("API error:"+str(r.status_code)); return
    data=r.json()
    if "error" in data: log("Error: "+data["error"]); return
    # Save JSON result
    with open("calibration_result.json","w") as f: json.dump(data,f,indent=2)
    log("Saved calibration_result.json")
    # Show first image + depth
    img_b64=data["images"][0]["rgb"]; dep_b64=data["images"][0]["depth"]
    img=plt.imread(io.BytesIO(base64.b64decode(img_b64)),format="png")
    dep=plt.imread(io.BytesIO(base64.b64decode(dep_b64)),format="png")
    fig,ax=plt.subplots(1,2,figsize=(12,6))
    ax[0].imshow(img); ax[0].set_title("RGB"); ax[0].axis("off")
    ax[1].imshow(dep); ax[1].set_title("Depth (disparity)"); ax[1].axis("off")
    plt.show()

ttk.Button(root,text="Sample Left",command=lambda:sample_images("L")).grid(row=4,column=0,pady=5)
ttk.Button(root,text="Sample Right",command=lambda:sample_images("R")).grid(row=4,column=1,pady=5)
ttk.Button(root,text="Send to API",command=send_to_api).grid(row=4,column=2,pady=5)

update_frame(); root.mainloop(); cap.release(); cv2.destroyAllWindows()

