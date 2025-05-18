import json                             # For loading the JSON file containing bounding‐box records
import numpy as np                      # For numerical operations (arrays, mean, std, type conversion)
import cv2                              # OpenCV for k-means clustering and image coordinate utilities
import matplotlib.pyplot as plt         # For plotting the clustered points

# --- Load normalized rectangles data from JSON ---
with open('multidetection_diff_lighting.json') as f:
    data = json.load(f)                # 'data' is a list of dicts, one per frame

# --- Compute and print number of frames processed ---
num_frames = len(data)                 # Total number of frame entries in the JSON
print(f"Number of frames: {num_frames}")

# --- Extract center points of all fused bounding boxes across frames ---
pts = []                               # Will collect each center as [x, y]
for record in data:                    # Iterate through each frame record
    for rect in record['fused_rectangles']:  # Each fused bounding box in this frame
        (x1, y1), (x2, y2) = rect      # Unpack top-left and bottom-right corners
        # Compute center coordinates and append as a list
        pts.append([(x1 + x2) / 2, (y1 + y2) / 2])
# Convert list of centers to a float32 NumPy array for k-means
pts = np.float32(pts)

# --- Cluster the centers into K=3 groups using OpenCV's k-means ---
K = 3                                 # Number of clusters desired
# Define convergence criteria: either 100 iterations or epsilon=0.2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# Run k-means: returns compactness (ignored), labels per point, and cluster centers
_, labels, centers = cv2.kmeans(
    pts,                               # Data to cluster
    K,                                 # Number of clusters
    None,                              # Initial labels (let OpenCV choose)
    criteria,                          # Convergence criteria
    10,                                # Number of attempts with different initial centers
    cv2.KMEANS_RANDOM_CENTERS          # Use random initial center selection
)

# --- Compute and print per-cluster statistics (mean and standard deviation) ---
cluster_stats = []                     # Will hold (mean, std) tuples for each cluster
for i in range(K):                     # For each cluster index 0..K-1
    # Select the points assigned to cluster i
    cluster_pts = pts[labels.ravel() == i]
    mean = np.mean(cluster_pts, axis=0)  # Mean X and Y for this cluster
    std  = np.std(cluster_pts, axis=0)   # Std dev in X and Y
    cluster_stats.append((mean, std))
    # Print summary for this cluster: 1-based index for readability
    print(
        f"Cluster {i+1}: "
        f"mean=({mean[0]:.2f}, {mean[1]:.2f}), "
        f"std=({std[0]:.2f}, {std[1]:.2f})"
    )

# --- Plot the clustered centers, cluster means, and annotate frame count ---
fig, ax = plt.subplots()               # Create a figure and single axes
colors = ['r', 'g', 'b']               # Colors for each of the 3 clusters

for i in range(K):
    # Extract this cluster's points again
    cluster_pts = pts[labels.ravel() == i]
    # Plot the points for cluster i
    ax.scatter(
        cluster_pts[:, 0], cluster_pts[:, 1],
        c=colors[i],                        # Use the i-th color
        label=f'Cluster {i+1}',             # Legend label
        alpha=0.6                           # Semi-transparent points
    )
    # Mark the cluster mean with a large 'x'
    ax.scatter(
        cluster_stats[i][0][0], cluster_stats[i][0][1],
        c=colors[i], marker='x',
        s=200, linewidths=2                # Size and line width of the marker
    )

# Annotate the total number of frames in the top-left corner (axis coords)
ax.text(
    0.02, 0.98,                           # Position at 2% from left, 98% from bottom
    f'Frames: {num_frames}',              # Text showing number of frames
    transform=ax.transAxes,               # Use axis-coordinate system
    fontsize=12,                          # Font size for annotation
    va='top',                             # Vertical alignment at the top of text
    bbox=dict(                            # Draw a white background box for readability
        boxstyle='round,pad=0.3',
        edgecolor='black',
        facecolor='white',
        alpha=0.8
    )
)

# Configure plot appearance
ax.set_title('Dispersion of Fused Bounding Box Centers for Multiple Objects')
ax.set_xlim(0, 640)                     # X-axis pixel range assuming 640px width
ax.set_ylim(0, 480)                     # Y-axis pixel range assuming 480px height
ax.invert_yaxis()                       # Flip Y-axis so origin is top-left if desired
ax.set_xlabel('X (pixels)')             # Label for X-axis
ax.set_ylabel('Y (pixels)')             # Label for Y-axis
ax.legend()                             # Show legend indicating cluster colors

plt.show()                              # Display the resulting scatter plot
