import json
import numpy as np
import matplotlib.pyplot as plt

# 1) Load your moving‐satellite data
with open('sphere_moving_1m_1min.json') as f:
    data = json.load(f)

# Extract center points of fused bounding boxes
centers = []
for record in data:
    for rect in record['fused_rectangles']:
        (x1, y1), (x2, y2) = rect
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))

# Separate x and y coordinates
xs = [c[0] for c in centers]
ys = [c[1] for c in centers]

# 4) Static std dev thresholds
static_std_x_sphere = 11.14
static_std_y_sphere = 10.73

# 5) Plot observed vs. fitted and errorbars
fig, ax = plt.subplots(figsize=(6, 4))

# error bars
ax.errorbar(xs, ys,
             xerr=static_std_x_sphere, yerr=static_std_y_sphere,
             fmt='none', ecolor='gray', alpha=0.5,
             elinewidth=1, capsize=2,
             label='Static ±1σ', zorder=1)

# observed centers
ax.scatter(xs, ys, s=5, c='blue', alpha=0.7,
           label='Fused bounding boxes Centers', zorder=2)

# annotate frame count
n_frames = len(data)
ax.text(
    0.98, 0.02,
    f'Frames: {n_frames}',
    transform=ax.transAxes,
    ha='right', va='bottom',
    fontsize=10,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
)

# set axis limits
ax.set_xlim(200, 500)
ax.set_ylim(220, 420)

# optional: invert y if your image origin is top‐left
ax.invert_yaxis()

ax.grid()
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.set_title('Moving Sphere: Center displacement vs Static σ')
ax.legend(loc='upper right', fontsize='small', markerscale=0.7)

plt.tight_layout()
plt.show()
