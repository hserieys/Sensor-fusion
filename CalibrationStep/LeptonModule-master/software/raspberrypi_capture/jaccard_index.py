import json                                  # For reading JSON files
import numpy as np                           # For numerical operations (means, arrays)
import matplotlib.pyplot as plt              # For plotting bar charts, histograms, etc.

# --- Load the per-frame statistics from a JSON file ---
with open("jaccard_indices_rocket_body_static_novote.json", "r") as f:
    frame_stats = json.load(f)               # frame_stats: list of dicts, one per frame

# --- Compute number of frames for use in titles/annotations ---
n_frames = len(frame_stats)

# --- 1) Compute mean contributions for each sensor (including FLIR) ---
sensors = ["red", "green", "blue", "depth", "flir"]
# For each sensor, gather its contribution across all frames and take the mean
mean_contrib = [
    np.mean([fs["contribution"][sensor] for fs in frame_stats])
    for sensor in sensors
]

# --- 2) Compute mean Jaccard index for each agreement threshold ---
threshold_keys = [
    ">=1_sensor",
    ">=2_sensors",
    ">=3_sensors",
    ">=4_sensors",
    ">=5_sensors"
]
# Human‐friendly labels matching the threshold keys  
j_labels = ["J ≥ 1", "J ≥ 2", "J ≥ 3", "J ≥ 4", "J ≥ 5"]
# Compute mean Jaccard value at each threshold
mean_jaccard = [
    np.mean([fs["jaccard"][key] for fs in frame_stats])
    for key in threshold_keys
]

# --- Plot 2: Mean Jaccard Indices as a bar chart ---
plt.figure()                              # Start a new figure
bars = plt.bar(j_labels, mean_jaccard)    # Draw bars for each threshold
plt.xlabel("Jaccard Indices")             # X-axis label
plt.ylabel("Mean Jaccard Index")          # Y-axis label
# Include the number of frames in the title for context
plt.title(f"Mean Jaccard Indices for Rocket Body (n={n_frames} frames)")

# Annotate each bar with its numerical value
for bar in bars:
    height = bar.get_height()             # Height of this bar
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # X-position: center of the bar
        height,                            # Y-position: top of the bar
        f"{height:.3f}",                   # Text: height formatted to three decimals
        ha="center",                       # Horizontal alignment: centered
        va="bottom"                        # Vertical alignment: just above the bar
    )

plt.tight_layout()                        # Adjust layout to prevent clipping
plt.show()                                # Display the plot

# -------------------------------------------------------------------
# The following code blocks are alternative visualizations that are
# currently commented out. Uncomment to use them.
# -------------------------------------------------------------------

# # Histogram of the four Jaccard distributions
# j1 = [fs["jaccard"][">=1_sensor"] for fs in frame_stats]
# j2 = [fs["jaccard"][">=2_sensors"] for fs in frame_stats]
# j3 = [fs["jaccard"][">=3_sensors"] for fs in frame_stats]
# j4 = [fs["jaccard"][">=4_sensors"] for fs in frame_stats]
# plt.figure()
# plt.hist(j1, bins=20, alpha=0.6, label="J ≥ 1")
# plt.hist(j2, bins=20, alpha=0.6, label="J ≥ 2")
# plt.hist(j3, bins=20, alpha=0.6, label="J ≥ 3")
# plt.hist(j4, bins=20, alpha=0.6, label="J ≥ 4")
# plt.xlabel("Jaccard Index")
# plt.ylabel("Frequency")
# plt.title("Histogram of Jaccard Distributions")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Boxplots of Jaccard at different k and variance of contributions
# variance = [
#     np.var([
#         fs["contribution"]["red"],
#         fs["contribution"]["green"],
#         fs["contribution"]["blue"],
#         fs["contribution"]["depth"]
#     ])
#     for fs in frame_stats
# ]
# j_labels_box = [fs["jaccard"][k] for k in [">=1_sensor",">=2_sensors",">=3_sensors",">=4_sensors"]]
# plt.figure(figsize=(8,4))
# # Boxplot of Jaccard distributions
# plt.subplot(1,2,1)
# plt.boxplot([j1, j2, j3, j4], labels=["J≥1","J≥2","J≥3","J≥4"])
# plt.ylabel("Jaccard")
# plt.title("Agreement Improves with Stricter Fusion")
# # Boxplot of variance of contributions
# plt.subplot(1,2,2)
# plt.boxplot(variance)
# plt.xticks([1], ["Var(contributions)"])
# plt.ylabel("Variance of [r,g,b,d]")
# plt.title("Per‐Frame Sensor Variance (Uncertainty)")
# plt.tight_layout()
# plt.show()

# # Concentric circles plot: size ∝ mean Jaccard
# from matplotlib.patches import Circle
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
# mean_jacc = mean_jaccard[:-1]  # omit the >=5 threshold if desired
# max_j = max(mean_jacc)
# radii = [mj / max_j for mj in mean_jacc]
# fig, ax = plt.subplots(figsize=(6,6))
# for radius, color, label in zip(radii, colors, j_labels[:-1]):
#     circle = Circle(
#         (0,0),
#         radius,
#         edgecolor=color,
#         facecolor='none',
#         linewidth=3,
#         label=f"{label}: {radius*max_j:.2f}"
#     )
#     ax.add_patch(circle)
# ax.set_aspect('equal')
# ax.axis('off')
# ax.legend(loc='upper right')
# plt.title("Concentric Circles (Size ∝ Mean Jaccard)")
# plt.show()
