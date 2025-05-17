import json
import numpy as np
import matplotlib.pyplot as plt

# Load normalized rectangles data
with open('rocket_body_static_1m_1min.json') as f:
    data = json.load(f)

# Compute number of frames
num_frames = len(data)

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

# Compute mean and standard deviation
mean_x = np.mean(xs)
mean_y = np.mean(ys)
std_x = np.std(xs)
std_y = np.std(ys)

# Print metrics
print(f"Number of frames: {num_frames}")
print(f"Mean X: {mean_x:.2f}")
print(f"Mean Y: {mean_y:.2f}")
print(f"Std Dev X: {std_x:.2f}")
print(f"Std Dev Y: {std_y:.2f}")

# Plotting the dispersion
fig, ax = plt.subplots()
ax.scatter(xs, ys, s=30, alpha=0.6)
ax.set_xlim(0, 640)
ax.set_ylim(0, 480)
ax.invert_yaxis()  # Origin at top-left

# Annotate frame count in the top-left corner (axis coords)
ax.text(
    0.02, 0.98,
    f'Frames: {num_frames}',
    transform=ax.transAxes,
    fontsize=12,
    va='top',
    bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=0.8)
)

ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.set_title('Dispersion of Fused Bounding Box Centers for Static Rocket Body')
plt.show()
