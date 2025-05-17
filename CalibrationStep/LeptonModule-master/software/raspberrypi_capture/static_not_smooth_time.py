import json
import numpy as np
import matplotlib.pyplot as plt

# ─── 1) Load data and build time axis ────────────────────────────────────────
with open("rocket_body_different_light_static.json", "r") as f:
    data = json.load(f)

t0    = data[0]["timestamps"]["rgb"]
times = np.array([rec["timestamps"]["rgb"] - t0 for rec in data])

# ─── 2) Helper to get mean center of a list of rects ───────────────────────
def mean_center(rects):
    if not rects:
        return np.nan, np.nan
    xs = [(r[0][0] + r[1][0]) / 2.0 for r in rects]
    ys = [(r[0][1] + r[1][1]) / 2.0 for r in rects]
    return float(np.mean(xs)), float(np.mean(ys))

# ─── 3) Extract raw centers (no fill / no smoothing) ───────────────────────
cx = {"Red":[], "Green":[], "Blue":[], "LiDAR":[], "FLIR":[], "Fused":[]}
cy = {k:[] for k in cx}

for rec in data:
    # individual RGB channels
    for ch in ("red","green","blue"):
        mx, my = mean_center(rec["rectangles"]["rgb"][ch])
        cx[ch.capitalize()].append(mx)
        cy[ch.capitalize()].append(my)
    # LiDAR
    mx, my = mean_center(rec["rectangles"]["lidar"])
    cx["LiDAR"].append(mx); cy["LiDAR"].append(my)
    # FLIR
    mx, my = mean_center(rec["rectangles"]["flir"])
    cx["FLIR"].append(mx);  cy["FLIR"].append(my)
    # Fused
    mx, my = mean_center(rec["fused_rectangles"])
    cx["Fused"].append(mx); cy["Fused"].append(my)

# convert to numpy arrays
for k in cx:
    cx[k] = np.array(cx[k], dtype=float)
    cy[k] = np.array(cy[k], dtype=float)

# ─── 4) Compute a single “RGB” curve as the mean of the three channels ──────
rgb_x = np.vstack([cx["Red"], cx["Green"], cx["Blue"]])
rgb_y = np.vstack([cy["Red"], cy["Green"], cy["Blue"]])
cx["RGB"] = np.nanmean(rgb_x, axis=0)
cy["RGB"] = np.nanmean(rgb_y, axis=0)

# ─── 5) Plot X‐centroid vs time (raw curves, NaNs break the line) ─────────
plt.figure(figsize=(10,4))
for name, style in [("RGB","-"), ("LiDAR","-"), ("FLIR","-"), ("Fused","--")]:
    y = np.ma.masked_invalid(cx[name])
    plt.plot(times, y, linestyle=style, linewidth=2, label=name)

plt.xlabel("Time (s) since start")
plt.ylabel("Mean Center X (px)")
plt.title("Raw Center-X Trajectories Over Time")
plt.legend()
plt.grid(ls="--", alpha=0.4)
plt.tight_layout()

# ─── 6) Plot Y‐centroid vs time ─────────────────────────────────────────────
plt.figure(figsize=(10,4))
for name, style in [("RGB","-"), ("LiDAR","-"), ("FLIR","-"), ("Fused","--")]:
    y = np.ma.masked_invalid(cy[name])
    plt.plot(times, y, linestyle=style, linewidth=2, label=name)

plt.xlabel("Time (s) since start")
plt.ylabel("Mean Center Y (px)")
plt.title("Raw Center-Y Trajectories Over Time")
plt.legend()
plt.grid(ls="--", alpha=0.4)
plt.tight_layout()

plt.show()
