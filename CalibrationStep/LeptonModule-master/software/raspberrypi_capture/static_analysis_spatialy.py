
import json
import matplotlib.pyplot as plt

# 1) Load your JSON log
with open("rocket_body_static_1m_1min.json", "r") as f:
    data = json.load(f)

# 2) Extract centers for each sensor and fused
centers_rgb = []
centers_lidar = []
centers_flir = []
centers_fused = []

for rec in data:
    # RGB: combine red, green, blue channels
    rgb_rects = (rec["rectangles"]["rgb"]["red"]
                 + rec["rectangles"]["rgb"]["green"]
                 + rec["rectangles"]["rgb"]["blue"])
    for r in rgb_rects:
        cx = (r[0][0] + r[1][0]) / 2.0
        cy = (r[0][1] + r[1][1]) / 2.0
        centers_rgb.append((cx, cy))
    # LiDAR
    for r in rec["rectangles"]["lidar"]:
        cx = (r[0][0] + r[1][0]) / 2.0
        cy = (r[0][1] + r[1][1]) / 2.0
        centers_lidar.append((cx, cy))
    # FLIR
    for r in rec["rectangles"]["flir"]:
        cx = (r[0][0] + r[1][0]) / 2.0
        cy = (r[0][1] + r[1][1]) / 2.0
        centers_flir.append((cx, cy))
    # Fused
    for r in rec["fused_rectangles"]:
        cx = (r[0][0] + r[1][0]) / 2.0
        cy = (r[0][1] + r[1][1]) / 2.0
        centers_fused.append((cx, cy))

# 3) Unpack coordinates (handle empty lists)
xr, yr = zip(*centers_rgb) if centers_rgb else ([], [])
xl, yl = zip(*centers_lidar) if centers_lidar else ([], [])
xf, yf = zip(*centers_flir) if centers_flir else ([], [])
x_fu, y_fu = zip(*centers_fused) if centers_fused else ([], [])

# 4) Plot
plt.figure(figsize=(7,7))
plt.scatter(xr, yr, s=100, alpha=1, label="RGB")
plt.scatter(xl, yl, s=30, alpha=1, label="LiDAR")
plt.scatter(xf, yf, s=10, alpha=1, label="FLIR")
plt.scatter(x_fu, y_fu, s=15, alpha=0.8, marker='x', label="Fused")
plt.axis([0, 640, 480, 0])
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.title("Spatial Scatter of Detection Centers for All Sensors")
# plt.gca().invert_yaxis()  # origin at top-left
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

'''
import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON log
with open("normalized_rects1m_static_draw_bboxes.json", "r") as f:
    data = json.load(f)

# Helper to compute mean center of a list of rects
def mean_center(rects):
    if not rects:
        return None, None
    xs = [(r[0][0] + r[1][0]) / 2.0 for r in rects]
    ys = [(r[0][1] + r[1][1]) / 2.0 for r in rects]
    return sum(xs) / len(xs), sum(ys) / len(ys)

# Collect centers for each sensor per frame
centers = {"RGB": [], "LiDAR": [], "FLIR": [], "Fused": []}

for rec in data:
    # RGB combined channels
    rgb_rects = rec["rectangles"]["rgb"]["red"] + \
                rec["rectangles"]["rgb"]["green"] + \
                rec["rectangles"]["rgb"]["blue"]
    cx, cy = mean_center(rgb_rects)
    if cx is not None:
        centers["RGB"].append((cx, cy))

    # LiDAR
    cx, cy = mean_center(rec["rectangles"]["lidar"])
    if cx is not None:
        centers["LiDAR"].append((cx, cy))

    # FLIR
    cx, cy = mean_center(rec["rectangles"]["flir"])
    if cx is not None:
        centers["FLIR"].append((cx, cy))

    # Fused
    cx, cy = mean_center(rec["fused_rectangles"])
    if cx is not None:
        centers["Fused"].append((cx, cy))

# Compute variances for X and Y
vars_x = {s: np.var([c[0] for c in lst]) for s, lst in centers.items() if lst}
vars_y = {s: np.var([c[1] for c in lst]) for s, lst in centers.items() if lst}

# 1) Plot X variance per sensor
plt.figure()
plt.bar(list(vars_x.keys()), list(vars_x.values()))
plt.xlabel("Sensor")
plt.ylabel("Variance of Center X (pixels²)")
plt.title("Center X Variance per Sensor")
plt.tight_layout()

# 2) Plot Y variance per sensor
plt.figure()
plt.bar(list(vars_y.keys()), list(vars_y.values()))
plt.xlabel("Sensor")
plt.ylabel("Variance of Center Y (pixels²)")
plt.title("Center Y Variance per Sensor")
plt.tight_layout()

plt.show()
'''
'''
import json
import matplotlib.pyplot as plt

# --- Helper functions ---

def envelope(rects):
    """Compute the envelope bounding box of a list of rectangles."""
    x1s = [r[0][0] for r in rects]
    y1s = [r[0][1] for r in rects]
    x2s = [r[1][0] for r in rects]
    y2s = [r[1][1] for r in rects]
    return (min(x1s), min(y1s)), (max(x2s), max(y2s))

def intersection_area(boxA, boxB):
    """Compute intersection area between two boxes ((x1,y1),(x2,y2))."""
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])
    if xB <= xA or yB <= yA:
        return 0
    return (xB - xA) * (yB - yA)

def area(box):
    """Area of box ((x1,y1),(x2,y2))."""
    return max(0, box[1][0] - box[0][0]) * max(0, box[1][1] - box[0][1])

# --- Load fusion log ---
with open("normalized_rects1m_static_draw_bboxes.json", "r") as f:
    data = json.load(f)

times = []
p_rgb_list = []
p_lidar_list = []
p_flir_list = []
belief_list = []
plaus_list = []
t0 = data[0]["timestamps"]["rgb"]

for rec in data:
    # build envelopes…
    rgb_rects   = rec["rectangles"]["rgb"]["red"]   + rec["rectangles"]["rgb"]["green"]   + rec["rectangles"]["rgb"]["blue"]
    lidar_rects = rec["rectangles"]["lidar"]
    flir_rects  = rec["rectangles"]["flir"]
    fused_rects = rec["fused_rectangles"]

    # only proceed if *all* are non-empty
    if not (rgb_rects and lidar_rects and flir_rects and fused_rects):
        continue

    # now get relative time *after* the check
    ts_rel = rec["timestamps"]["rgb"] - t0
    times.append(ts_rel)

    # compute your overlap-ratios
    env_rgb   = envelope(rgb_rects)
    env_lidar = envelope(lidar_rects)
    env_flir  = envelope(flir_rects)
    env_fused = envelope(fused_rects)

    p_rgb   = intersection_area(env_rgb, env_fused)   / area(env_rgb)
    p_lidar = intersection_area(env_lidar, env_fused) / area(env_lidar)
    p_flir  = intersection_area(env_flir, env_fused)  / area(env_flir)

    # Min/Max‐Rule
    belief = min(p_rgb, p_lidar, p_flir)
    plaus  = max(p_rgb, p_lidar, p_flir)

    # append everything
    p_rgb_list.append(p_rgb)
    p_lidar_list.append(p_lidar)
    p_flir_list.append(p_flir)
    belief_list.append(belief)
    plaus_list.append(plaus)

# --- Plotting ---

# 1) Probabilities per sensor over time
plt.figure()
plt.plot(times, p_rgb_list,   label='p_rgb',   marker='o')
plt.plot(times, p_lidar_list, label='p_lidar', marker='s')
plt.plot(times, p_flir_list,  label='p_flir',  marker='^')
plt.xlabel("Time (s)")
plt.ylabel("Overlap Ratio")
plt.title("Sensor Support (Overlap Ratios) Over Time")
plt.legend()
plt.tight_layout()

# 2) Belief and Plausibility over time
plt.figure()
plt.plot(times, belief_list, label='Belief (min)',    linewidth=2)
plt.plot(times, plaus_list,  label='Plausibility (max)', linestyle='--', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title("Belief vs. Plausibility Over Time")
plt.legend()
plt.tight_layout()

# 3) Histogram of Belief and Plausibility
plt.figure()
plt.hist(belief_list, bins=20, alpha=0.6, label='Belief')
plt.hist(plaus_list,  bins=20, alpha=0.6, label='Plausibility')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Distribution of Belief and Plausibility")
plt.legend()
plt.tight_layout()

plt.show()'''