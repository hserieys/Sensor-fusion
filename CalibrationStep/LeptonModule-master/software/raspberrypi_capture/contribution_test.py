import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open("jaccard_indices_rocket_body_static_novote.json", "r") as f:
    frame_stats = json.load(f)

n_frames = len(frame_stats)

# 1) Compute mean contributions for each sensor (omit FLIR)
sensors = ["red", "green", "blue", "depth", "flir"]
mean_contrib = [
    np.mean([fs["contribution"][sensor] for fs in frame_stats])
    for sensor in sensors
]

# 2) Compute mean Jaccard index for each agreement threshold
threshold_keys = [">=1_sensor", ">=2_sensors", ">=3_sensors", ">=4_sensors", ">=5_sensors"]
j_labels = ["J ≥ 1", "J ≥ 2", "J ≥ 3", "J ≥ 4", "J ≥ 5"]
mean_jaccard = [
    np.mean([fs["jaccard"][key] for fs in frame_stats])
    for key in threshold_keys
]

# Plot 2: Mean Jaccard Indices
plt.figure()
bars = plt.bar(j_labels, mean_jaccard)
plt.xlabel("Jaccard Indices")
plt.ylabel("Mean Jaccard Index")
plt.title(f"Mean Jaccard Indices for Rocket Body (n={n_frames} frames)")
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h, f"{h:.3f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# import json
# import numpy as np
# import matplotlib.pyplot as plt
# 
# # Load data
# with open("sensor_contributions_multidetection.json", "r") as f:
#     frame_stats = json.load(f)
# 
# # Extract Jaccard distributions
# j1 = [fs["jaccard"][">=1_sensor"]  for fs in frame_stats]
# j2 = [fs["jaccard"][">=2_sensors"] for fs in frame_stats]
# j3 = [fs["jaccard"][">=3_sensors"] for fs in frame_stats]
# j4 = [fs["jaccard"][">=4_sensors"] for fs in frame_stats]
# 
# # Plot histogram of the four Jaccard distributions
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

# import json
# import numpy as np
# import matplotlib.pyplot as plt
# 
# # Load your fused results
# with open("sensor_contributions.json","r") as f:
#     stats = json.load(f)
# 
# # Pre-allocate lists
# j1, j2, j3, j4 = [], [], [], []
# variance = []
# 
# for fs in stats:
#     # Jaccard thresholds
#     j1.append( fs["jaccard"][">=1_sensor"] )
#     j2.append( fs["jaccard"][">=2_sensors"] )
#     j3.append( fs["jaccard"][">=3_sensors"] )
#     j4.append( fs["jaccard"][">=4_sensors"] )
#     # Contribution variance
#     vals = [
#         fs["contribution"]["red"],
#         fs["contribution"]["green"],
#         fs["contribution"]["blue"],
#         fs["contribution"]["depth"]
#     ]
#     variance.append(np.var(vals))
# 
# # 1) Boxplot of Jaccard at different k
# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.boxplot([j1,j2,j3,j4], labels=["J≥1","J≥2","J≥3","J≥4"])
# plt.ylabel("Jaccard")
# plt.title("Agreement Improves with Stricter Fusion")
# 
# # 2) Boxplot of variance
# plt.subplot(1,2,2)
# plt.boxplot(variance)
# plt.xticks([1],["Var(cont.)"])
# plt.ylabel("Variance of [r,g,b,d]")
# plt.title("Per‐Frame Sensor Variance (Uncertainty)")
# 
# plt.tight_layout()
# plt.show()

# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
# 
# # Load data
# with open("sensor_contributions.json", "r") as f:
#     frame_stats = json.load(f)
# 
# # Jaccard thresholds and labels
# threshold_keys = [">=1_sensor", ">=2_sensors", ">=3_sensors", ">=4_sensors"]
# labels = ["J ≥ 1", "J ≥ 2", "J ≥ 3", "J ≥ 4"]
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Matplotlib default colors
# 
# # Compute mean Jaccard for each threshold
# mean_jaccard = [
#     np.mean([fs["jaccard"][key] for fs in frame_stats])
#     for key in threshold_keys
# ]
# 
# # Normalize radii so max J corresponds to radius 1
# max_j = max(mean_jaccard)
# radii = [mj / max_j for mj in mean_jaccard]
# 
# fig, ax = plt.subplots(figsize=(6, 6))
# center = (0, 0)
# 
# # Draw concentric circles
# for radius, color, label in zip(radii, colors, labels):
#     circle = Circle(center, radius, 
#                     edgecolor=color, facecolor='none', linewidth=3, label=f"{label}: {radius*max_j:.2f}")
#     ax.add_patch(circle)
# 
# # Formatting
# ax.set_aspect('equal', 'box')
# ax.set_xlim(-1.1, 1.1)
# ax.set_ylim(-1.1, 1.1)
# ax.axis('off')
# ax.legend(loc='upper right')
# 
# plt.title("Concentric Circles (Size ∝ Mean Jaccard)")
# plt.show()

