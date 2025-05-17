import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load normalized rectangles data
with open('multidetection_diff_lighting.json') as f:
    data = json.load(f)

# Compute and print number of frames
num_frames = len(data)
print(f"Number of frames: {num_frames}")

# Extract center points of fused bounding boxes
pts = []
for record in data:
    for rect in record['fused_rectangles']:
        (x1, y1), (x2, y2) = rect
        pts.append([(x1 + x2) / 2, (y1 + y2) / 2])
pts = np.float32(pts)

# Cluster into 3 groups using k-means
K = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(
    pts, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

# Compute per-cluster mean and std dev
cluster_stats = []
for i in range(K):
    cluster_pts = pts[labels.ravel() == i]
    mean = np.mean(cluster_pts, axis=0)
    std  = np.std(cluster_pts, axis=0)
    cluster_stats.append((mean, std))
    print(
        f"Cluster {i+1}: "
        f"mean=({mean[0]:.2f}, {mean[1]:.2f}), "
        f"std=({std[0]:.2f}, {std[1]:.2f})"
    )

# Plot clusters with means and frame count annotation
fig, ax = plt.subplots()
colors = ['r', 'g', 'b']
for i in range(K):
    cluster_pts = pts[labels.ravel() == i]
    ax.scatter(
        cluster_pts[:,0], cluster_pts[:,1],
        c=colors[i], label=f'Cluster {i+1}', alpha=0.6
    )
    ax.scatter(
        cluster_stats[i][0][0], cluster_stats[i][0][1],
        c=colors[i], marker='x', s=200, linewidths=2
    )

# Annotate frame count
ax.text(
    0.02, 0.98,
    f'Frames: {num_frames}',
    transform=ax.transAxes,
    fontsize=12,
    va='top',
    bbox=dict(boxstyle='round,pad=0.3',
              edgecolor='black',
              facecolor='white',
              alpha=0.8)
)

ax.set_title('Dispersion of Fused Bounding Box Centers for Multiple Objects')
ax.set_xlim(0, 640)
ax.set_ylim(0, 480)
ax.invert_yaxis()
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.legend()
plt.show()
