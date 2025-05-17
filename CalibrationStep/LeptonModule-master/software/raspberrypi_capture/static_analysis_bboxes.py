import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1) Load your JSON log
with open("satellite_static_1m_1min.json", "r") as f:
    data = json.load(f)
    
# 2) Collect per-frame envelope boxes for each channel/sensor
envs = {k: [] for k in ["red", "green", "blue", "lidar", "flir", "fused"]}

for rec in data:
    # RGB channels separately
    for ch in ["red", "green", "blue"]:
        rects = rec["rectangles"]["rgb"][ch]
        if rects:
            x1s = [r[0][0] for r in rects]
            y1s = [r[0][1] for r in rects]
            x2s = [r[1][0] for r in rects]
            y2s = [r[1][1] for r in rects]
            envs[ch].append((min(x1s), min(y1s), max(x2s), max(y2s)))
    # LiDAR
    rects = rec["rectangles"]["lidar"]
    if rects:
        x1s = [r[0][0] for r in rects]
        y1s = [r[0][1] for r in rects]
        x2s = [r[1][0] for r in rects]
        y2s = [r[1][1] for r in rects]
        envs["lidar"].append((min(x1s), min(y1s), max(x2s), max(y2s)))
    # FLIR
    rects = rec["rectangles"]["flir"]
    if rects:
        x1s = [r[0][0] for r in rects]
        y1s = [r[0][1] for r in rects]
        x2s = [r[1][0] for r in rects]
        y2s = [r[1][1] for r in rects]
        envs["flir"].append((min(x1s), min(y1s), max(x2s), max(y2s)))
    # Fused
    rects = rec["fused_rectangles"]
    if rects:
        x1s = [r[0][0] for r in rects]
        y1s = [r[0][1] for r in rects]
        x2s = [r[1][0] for r in rects]
        y2s = [r[1][1] for r in rects]
        envs["fused"].append((min(x1s), min(y1s), max(x2s), max(y2s)))

# 3) Compute the mean envelope box across all frames for each key
mean_env = {}
for k, boxes in envs.items():
    if not boxes:
        continue
    x1m = sum(b[0] for b in boxes) / len(boxes)
    y1m = sum(b[1] for b in boxes) / len(boxes)
    x2m = sum(b[2] for b in boxes) / len(boxes)
    y2m = sum(b[3] for b in boxes) / len(boxes)
    mean_env[k] = (x1m, y1m, x2m, y2m)

# 4) Plot the six mean bounding boxes
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 640)
ax.set_ylim(480, 0)  # origin top-left
ax.set_aspect('equal')
ax.set_title("Mean Envelope Bounding Boxes per Sensor/Channel")
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")

colors = {
    "red": "red",
    "green": "green",
    "blue": "blue",
    "lidar": "orange",
    "flir": "purple",
    "fused": "black"
}

for k, (x1, y1, x2, y2) in mean_env.items():
    w = x2 - x1
    h = y2 - y1
    lw = 2 if k == "fused" else 1
    rect = patches.Rectangle(
        (x1, y1), w, h,
        linewidth=lw,
        edgecolor=colors.get(k, "gray"),
        facecolor='none',
        label=k.capitalize()
    )
    ax.add_patch(rect)

ax.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()