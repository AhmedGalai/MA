from datasets import load_dataset
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Step 1: Load the dataset
ds = load_dataset("jasonzhango/SPAR-7M-RGBD", split="train")  # Adjust split if applicable

# Step 2: Pick a random example
idx = random.randint(0, len(ds) - 1)
example = ds[idx]

# Inspect available keys
print("Example keys:", example.keys())

# Step 3: Access image and disparity
# Adjust these keys based on dataset schema:
# Common field names might be "image" or "rgb" for the image and "disparity" or "depth" for the map
rgb_bytes = example.get("image") or example.get("rgb")
disp_array = example.get("disparity") or example.get("depth")

if rgb_bytes is None or disp_array is None:
    raise KeyError("Expected fields 'image' or 'rgb', and 'disparity' or 'depth' not found.")

# Step 4: Convert RGB bytes to numpy array
rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
rgb_np = np.array(rgb_img)

# If disparity is also stored as raw bytes (e.g., PNG), decode similarly; else it's a float array
if isinstance(disp_array, (bytes, bytearray)):
    disp_img = Image.open(io.BytesIO(disp_array))
    disp = np.array(disp_img).astype(float)
else:
    disp = np.array(disp_array, dtype=float)

# Step 5: Display them side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(rgb_np)
axes[0].set_title("RGB Image")
axes[0].axis("off")

im = axes[1].imshow(disp, cmap="plasma")
axes[1].set_title("Disparity Map")
axes[1].axis("off")

fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
plt.show()

