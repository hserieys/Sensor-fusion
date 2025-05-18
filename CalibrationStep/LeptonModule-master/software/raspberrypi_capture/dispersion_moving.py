import json                            # For loading JSON data files
import numpy as np                     # For numerical operations (mean, std, arrays)
import matplotlib.pyplot as plt        # For creating plots and visualizations

# --- 1) Load the moving-sphere experiment data from JSON ---
with open('sphere_moving_1m_1min.json') as f:
    data = json.load(f)                # 'data' is a list of records, one per frame

# --- 2) Extract center points of fused bounding boxes across all frames ---
centers = []                           # Will collect (cx, cy) for every fused rectangle
for record in data:                    # Iterate over each frame's record
    for rect in record['fused_rectangles']:  # Each fused bounding box in this frame
        (x1, y1), (x2, y2) = rect      # Unpack the top-left and bottom-right corners
        cx = (x1 + x2) / 2             # Compute center X-coordinate
        cy = (y1 + y2) / 2             # Compute center Y-coordinate
        centers.append((cx, cy))       # Append the center tuple to the list

# --- 3) Separate the X and Y coordinates into separate lists for plotting ---
xs = [c[0] for c in centers]           # All center X-values
ys = [c[1] for c in centers]           # All center Y-values

# --- 4) Define the static frame's standard deviation thresholds (pixels) ---
static_std_x_sphere = 11.14            # 1σ variation in X for the static sphere experiment
static_std_y_sphere = 10.73            # 1σ variation in Y for the static sphere experiment

# --- 5) Plot observed centers with error bars reflecting static σ thresholds ---
fig, ax = plt.subplots(figsize=(6, 4)) # Create a figure and axis with specified size

# Plot error bars (horizontal and vertical) for the static σ around each observed point
ax.errorbar(
    xs, ys,
    xerr=static_std_x_sphere,                 # Horizontal error = static σ in X
    yerr=static_std_y_sphere,                 # Vertical error = static σ in Y
    fmt='none',                               # No marker for the errorbar points
    ecolor='gray',                            # Color of the error bars
    alpha=0.5,                                # Semi-transparent error bars
    elinewidth=1,                             # Line width of error bar caps
    capsize=2,                                # Cap size at ends of error bars
    label='Static ±1σ',                       # Legend label
    zorder=1                                  # Draw behind the scatter points
)

# Plot the observed fused bounding-box centers
ax.scatter(
    xs, ys,
    s=5,                                      # Marker size
    c='blue',                                 # Marker color
    alpha=0.7,                                # Semi-transparent markers
    label='Fused bounding boxes Centers',     # Legend label
    zorder=2                                  # Draw on top of error bars
)

# Annotate the total number of frames in the bottom-right corner of the plot
n_frames = len(data)                        # Total frame count
ax.text(
    0.98, 0.02,                              # Position in axis coordinates (right, bottom)
    f'Frames: {n_frames}',                   # Text showing frame count
    transform=ax.transAxes,                  # Interpret coordinates in axis-relative terms
    ha='right',                              # Right-align the text horizontally
    va='bottom',                             # Bottom-align the text vertically
    fontsize=10,                             # Font size for annotation
    bbox=dict(                               # Draw a background box for readability
        boxstyle='round,pad=0.3',
        facecolor='white',
        alpha=0.7
    )
)

# --- 6) Configure axis limits to focus on the region of interest ---
ax.set_xlim(200, 500)                       # X-axis spans from pixel 200 to 500
ax.set_ylim(220, 420)                       # Y-axis spans from pixel 220 to 420

# If the image coordinate origin is at the top-left, invert the Y-axis
ax.invert_yaxis()

# Add grid lines for easier reading
ax.grid(True)

# Label the axes
ax.set_xlabel('X (pixels)')                 # X-axis label
ax.set_ylabel('Y (pixels)')                 # Y-axis label

# Add a descriptive title
ax.set_title('Moving Sphere: Center displacement vs Static σ')

# Display the legend in the upper-right corner
ax.legend(loc='upper right', fontsize='small', markerscale=0.7)

# Adjust layout to prevent clipping of labels/annotations
plt.tight_layout()

# Render the plot to the screen
plt.show()
