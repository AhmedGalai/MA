from transformers import pipeline, infer_device
import torch
device = infer_device()
checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
pipe = pipeline("depth-estimation", model=checkpoint, device=device)

from PIL import Image
import requests
import matplotlib.pyplot as plt

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)
#image


predictions = pipe(image)


# 1) Load your image (use a raw string so backslashes don't escape)
img_path = r"C:\Users\Lenovo\Desktop\Ahmed\master\MA\sandbox\RGBD\samples\20250831_182241_R_50.jpg"
image = Image.open(img_path).convert("RGB")

# 2) Predict depth
pred = pipe(image)                 # uses your existing pipeline
depth_img = pred["depth"]          # PIL Image (mode "L"), same size as input

# 3) Show RGB (left) and depth (right) in one figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title("RGB")
axes[0].axis("off")

axes[1].imshow(depth_img, cmap="gray")
axes[1].set_title("Predicted Depth")
axes[1].axis("off")

plt.tight_layout()
plt.show()
