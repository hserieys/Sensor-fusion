import json
import numpy as np
import matplotlib.pyplot as plt

# ─── 1) Chargement et time axis ─────────────────────────────────────────────
with open("satellite_different_light_static.json", "r") as f:
    data = json.load(f)

t0    = data[0]["timestamps"]["rgb"]
times = np.array([rec["timestamps"]["rgb"] - t0 for rec in data])

# ─── 2) Helper pour centre moyen ────────────────────────────────────────────
def mean_center(rects):
    if not rects:
        return np.nan, np.nan
    xs = [(r[0][0] + r[1][0]) / 2.0 for r in rects]
    ys = [(r[0][1] + r[1][1]) / 2.0 for r in rects]
    return float(np.mean(xs)), float(np.mean(ys))

# ─── 3) Extraction des centres bruts ────────────────────────────────────────
cx = {"Red":[], "Green":[], "Blue":[], "LiDAR":[], "FLIR":[], "Fused":[]}
cy = {k:[] for k in cx}

for rec in data:
    for ch in ("red","green","blue"):
        mx, my = mean_center(rec["rectangles"]["rgb"][ch])
        cx[ch.capitalize()].append(mx)
        cy[ch.capitalize()].append(my)
    mx, my = mean_center(rec["rectangles"]["lidar"])
    cx["LiDAR"].append(mx); cy["LiDAR"].append(my)
    mx, my = mean_center(rec["rectangles"]["flir"])
    cx["FLIR"].append(mx);  cy["FLIR"].append(my)
    mx, my = mean_center(rec["fused_rectangles"])
    cx["Fused"].append(mx); cy["Fused"].append(my)

# ─── 4) Passage en numpy ─────────────────────────────────────────────────────
for k in cx:
    cx[k] = np.array(cx[k], dtype=float)
    cy[k] = np.array(cy[k], dtype=float)

# ─── 5) Calcul de la courbe RGB combinée ────────────────────────────────────
rgb_x = np.vstack([cx["Red"], cx["Green"], cx["Blue"]])
rgb_y = np.vstack([cy["Red"], cy["Green"], cy["Blue"]])
cx["RGB"] = np.nanmean(rgb_x, axis=0)
cy["RGB"] = np.nanmean(rgb_y, axis=0)

# ─── 6) Smoothing + limité aux petits trous ─────────────────────────────────
def fill_and_smooth(arr, window=11, max_gap=5):
    idx, mask = np.arange(len(arr)), ~np.isnan(arr)
    if mask.sum() < 2:
        return arr.copy()
    filled = np.interp(idx, idx[mask], arr[mask])
    for i in idx[~mask]:
        left  = idx[mask & (idx < i)]
        right = idx[mask & (idx > i)]
        dmin  = np.inf
        if left.size:  dmin = min(dmin, i-left.max())
        if right.size: dmin = min(dmin, right.min()-i)
        if dmin > max_gap:
            filled[i] = np.nan
    kernel   = np.ones(window)/window
    return np.convolve(filled, kernel, mode="same")

to_plot    = ["RGB","LiDAR","FLIR","Fused"]
cx_smooth  = {n: fill_and_smooth(cx[n]) for n in to_plot}
cy_smooth  = {n: fill_and_smooth(cy[n]) for n in to_plot}

# ─── define your event times (seconds) and labels ───────────────────────────
event1, event2, event3, event4, event5 = 8.3, 35.6, 62.7, 92.1, 79.3 
events = [("Halogen Light Off", event1), ("Room Light Off: RGB loss", event2), ("Lidar loss", event3), ("Lidar available", event4), ("Room Light On: RGB available", event5)]

# ─── 7) Tracé X vs temps + event markers ────────────────────────────────────
plt.figure(figsize=(10,4))
for n in to_plot:
    y = np.ma.masked_invalid(cx_smooth[n])
    style = "--" if n=="Fused" else "-."
    if n=="RGB": style = "-"
    plt.plot(times, y, label=n, linestyle=style, linewidth=2)

# add vertical lines
for label, t in events:
    plt.axvline(t, color="k", linestyle=":", linewidth=1)
    plt.text(t + 0.5, plt.ylim()[1]*0.95, label,
             rotation=90, va="top", fontsize="small")

plt.xlabel("Time (s)")
plt.ylabel("Mean Center X (px)")
plt.title("Satellite Center X Trajectories Over Time")
plt.legend(ncol=2, fontsize="small")
plt.grid(True, ls="--", alpha=0.4)
plt.tight_layout()

# ─── 8) Tracé Y vs temps + event markers ────────────────────────────────────
plt.figure(figsize=(9,4))
for n in to_plot:
    y = np.ma.masked_invalid(cy_smooth[n])
    style = "--" if n=="Fused" else "-."
    if n=="RGB": style = "-"
    plt.plot(times, y, label=n, linestyle=style, linewidth=2)

for label, t in events:
    plt.axvline(t, color="k", linestyle=":", linewidth=1)
    plt.text(t + 0.5, plt.ylim()[1]*0.72, label,
             rotation=90, va="top", fontsize="small")

plt.xlabel("Time (s)")
plt.ylabel("Mean Center Y (px)")
plt.title("Satellite Center Y Trajectories Over Time")
plt.legend(ncol=1, fontsize="small")
plt.grid(True, ls="--", alpha=0.4)
plt.tight_layout()

plt.show()
