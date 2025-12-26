#!/usr/bin/env python3
"""
Quick test to visualize the colormap reversal effect.
Shows what the reversed colormap looks like for HuggingFace depth.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_gradient():
    """Create a simple gradient to show colormap effect."""
    gradient = np.linspace(0, 255, 256).astype(np.uint8)
    gradient = np.tile(gradient, (50, 1))
    return gradient


def apply_colormap(gradient, reverse=False):
    """Apply colormap with optional reversal."""
    if reverse:
        gradient = 255 - gradient
    return cv2.applyColorMap(gradient, cv2.COLORMAP_JET)


# Create gradients
gradient = create_gradient()

# Apply normal and reversed colormaps
normal = apply_colormap(gradient, reverse=False)
reversed_cm = apply_colormap(gradient, reverse=True)

# Convert BGR to RGB for matplotlib
normal_rgb = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
reversed_rgb = cv2.cvtColor(reversed_cm, cv2.COLOR_BGR2RGB)

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 4))

axes[0].imshow(normal_rgb)
axes[0].set_title('RealSense Disparity - Normal Colormap\n(Near=Red/Yellow, Far=Blue)',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Far ← Distance → Near')
axes[0].set_yticks([])
axes[0].set_xticks([0, 128, 255])
axes[0].set_xticklabels(['Far\n(Blue)', 'Medium\n(Green)', 'Near\n(Red)'])

axes[1].imshow(reversed_rgb)
axes[1].set_title('HuggingFace Depth - Reversed Colormap\n(Near=Blue, Far=Red)',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Far ← Distance → Near')
axes[1].set_yticks([])
axes[1].set_xticks([0, 128, 255])
axes[1].set_xticklabels(['Far\n(Red)', 'Medium\n(Green)', 'Near\n(Blue)'])

plt.tight_layout()
plt.savefig('colormap_comparison.png', dpi=150, bbox_inches='tight')
print("Colormap comparison saved to: colormap_comparison.png")

# Also save as simple image for quick viewing
combined = np.vstack([normal, np.ones((10, normal.shape[1], 3), dtype=np.uint8) * 255, reversed_cm])
cv2.imwrite('colormap_comparison_simple.png', combined)
print("Simple comparison saved to: colormap_comparison_simple.png")

print("\n" + "=" * 60)
print("Colormap Reversal Effect")
print("=" * 60)
print("\nRealSense (Normal):")
print("  Near objects → Red/Yellow")
print("  Far objects  → Blue")
print("\nHuggingFace (Reversed):")
print("  Near objects → Blue")
print("  Far objects  → Red/Yellow")
print("\nThis makes HuggingFace depth maps follow standard")
print("depth visualization conventions.")
print("=" * 60)
