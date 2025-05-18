import json                             # For loading the JSON file containing normalized rectangles
import numpy as np                      # For numerical operations (mean, std, array manipulations)
import matplotlib.pyplot as plt         # For plotting dispersion of centers

# --- Load normalized rectangles data from JSON ---
with open('rocket_body_static_1m_1min.json') as f:
    data = json.load(f)                 # 'data' is a list of per-frame records

# --- Compute number of frames processed ---
num_frames = len(data)                 # Total count of frame records

# --- Extract center points of all fused bounding boxes across frames ---
centers = []                           # Will hold (cx, cy) for each fused rectangle
for record in data:                    # Iterate over each frame's record
    for rect in record['fused_rectangles']:  # Each fused bounding box in this frame
        (x1, y1), (x2, y2) = rect      # Unpack top-left and bottom-right corners
        cx = (x1 + x2) / 2             # Compute center X-coordinate
        cy = (y1 + y2) / 2             # Compute center Y-coordinate
        centers.append((cx, cy))       # Append center tuple to list

# --- Separate X and Y coordinates into individual lists ---
xs = [c[0] for c in centers]           # All center X-values
ys = [c[1] for c in centers]           # All center Y-values

# --- Compute mean and standard deviation of the center coordinates ---
mean_x = np.mean(xs)                   # Average X position
mean_y = np.mean(ys)                   # Average Y position
std_x  = np.std(xs)                    # Spread (std dev) in X
std_y  = np.std(ys)                    # Spread (std dev) in Y

# --- Print out the computed metrics ---
print(f"Number of frames: {num_frames}")  
print(f"Mean X: {mean_x:.2f}")         # Formatted to two decimal places
print(f"Mean Y: {mean_y:.2f}")
print(f"Std Dev X: {std_x:.2f}")
print(f"Std Dev Y: {std_y:.2f}")

# --- Plotting the dispersion of all fused-box centers ---
fig, ax = plt.subplots()               # Create a figure and axis
ax.scatter(xs, ys, s=30, alpha=0.6)    # Scatter plot of centers, size=30, semi-transparent
ax.set_xlim(0, 640)                    # X-axis range: 0 to image width
ax.set_ylim(0, 480)                    # Y-axis range: 0 to image height
ax.invert_yaxis()                      # Flip Y-axis so origin (0,0) is top-left

# Annotate the total frame count in the top-left of the plot (normalized coords)
ax.text(
    0.02, 0.98,                        # Position at 2% from left, 98% from bottom
    f'Frames: {num_frames}',           # Text showing number of frames
    transform=ax.transAxes,            # Use axis coordinate system
    fontsize=12,                       # Font size for annotation
    va='top',                          # Vertically align text at its top
    bbox=dict(                         # Draw a background box around text
        boxstyle='round,pad=0.3',
        edgecolor='black',
        facecolor='white',
        alpha=0.8
    )
)

# Label axes and add a title for clarity
ax.set_xlabel('X (pixels)')            # X-axis label
ax.set_ylabel('Y (pixels)')            # Y-axis label
ax.set_title('Dispersion of Fused Bounding Box Centers for Static Rocket Body')

plt.show()                             # Display the plot window
