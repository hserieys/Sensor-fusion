import json                           # For loading JSON data
import numpy as np                    # For numerical operations and arrays
import matplotlib.pyplot as plt       # For creating plots

# ─── 1) Load data and build time axis ───────────────────────────────────────
with open("satellite_different_light_static.json", "r") as f:
    data = json.load(f)               # 'data' is a list of records, each with timestamps and rectangles

# Reference time (first RGB timestamp) for relative timing
t0    = data[0]["timestamps"]["rgb"]
# Compute elapsed time (in seconds) for each record relative to t0
times = np.array([rec["timestamps"]["rgb"] - t0 for rec in data])

# ─── 2) Helper to compute mean center of a list of rectangles ──────────────
def mean_center(rects):
    """
    Given a list of rectangles ((x1,y1),(x2,y2)), return the mean center (cx, cy).
    If the list is empty, return (nan, nan).
    """
    if not rects:
        return np.nan, np.nan
    # Compute each rectangle's center
    xs = [(r[0][0] + r[1][0]) / 2.0 for r in rects]
    ys = [(r[0][1] + r[1][1]) / 2.0 for r in rects]
    # Return the mean of those centers
    return float(np.mean(xs)), float(np.mean(ys))

# ─── 3) Extract raw centers for each sensor/channel ─────────────────────────
# Prepare dictionaries to accumulate x- and y-centroids
cx = {"Red": [], "Green": [], "Blue": [], "LiDAR": [], "FLIR": [], "Fused": []}
cy = {k: [] for k in cx}

for rec in data:
    # RGB individual channels
    for ch in ("red", "green", "blue"):
        mx, my = mean_center(rec["rectangles"]["rgb"][ch])
        cx[ch.capitalize()].append(mx)
        cy[ch.capitalize()].append(my)
    # LiDAR channel
    mx, my = mean_center(rec["rectangles"]["lidar"])
    cx["LiDAR"].append(mx)
    cy["LiDAR"].append(my)
    # FLIR channel
    mx, my = mean_center(rec["rectangles"]["flir"])
    cx["FLIR"].append(mx)
    cy["FLIR"].append(my)
    # Combined fused detections
    mx, my = mean_center(rec["fused_rectangles"])
    cx["Fused"].append(mx)
    cy["Fused"].append(my)

# ─── 4) Convert lists to numpy arrays for easier handling ────────────────────
for k in cx:
    cx[k] = np.array(cx[k], dtype=float)
    cy[k] = np.array(cy[k], dtype=float)

# ─── 5) Compute combined RGB trajectory by averaging the three channels ─────
# Stack the three color-channel arrays and take the mean across them
rgb_x = np.vstack([cx["Red"], cx["Green"], cx["Blue"]])
rgb_y = np.vstack([cy["Red"], cy["Green"], cy["Blue"]])
cx["RGB"] = np.nanmean(rgb_x, axis=0)  # Ignore NaNs when computing mean
cy["RGB"] = np.nanmean(rgb_y, axis=0)

# ─── 6) Smoothing function: fill small NaN gaps and apply moving average ────
def fill_and_smooth(arr, window=11, max_gap=5):
    """
    Interpolate small gaps in 'arr' (replacing NaNs up to max_gap long),
    then apply a uniform moving average with given window size.
    """
    idx, mask = np.arange(len(arr)), ~np.isnan(arr)
    # If fewer than two valid points, return a copy without smoothing
    if mask.sum() < 2:
        return arr.copy()
    # Linear interpolation over valid indices
    filled = np.interp(idx, idx[mask], arr[mask])
    # Re-introduce NaNs for gaps larger than max_gap
    for i in idx[~mask]:
        left  = idx[mask & (idx < i)]
        right = idx[mask & (idx > i)]
        dmin  = np.inf
        if left.size:   dmin = min(dmin, i - left.max())
        if right.size:  dmin = min(dmin, right.min() - i)
        if dmin > max_gap:
            filled[i] = np.nan
    # Moving average kernel
    kernel = np.ones(window) / window
    # Convolve and return same-length array
    return np.convolve(filled, kernel, mode="same")

# Apply smoothing to the selected trajectories
to_plot   = ["RGB", "LiDAR", "FLIR", "Fused"]
cx_smooth = {n: fill_and_smooth(cx[n]) for n in to_plot}
cy_smooth = {n: fill_and_smooth(cy[n]) for n in to_plot}

# ─── 7) Define event times (in seconds) and their labels ────────────────────
events = [
    ("Halogen Light Off",              8.3),
    ("Room Light Off: RGB loss",      35.6),
    ("Lidar loss",                    62.7),
    ("Lidar available",               92.1),
    ("Room Light On: RGB available",  79.3)
]

# ─── 8) Plot mean center X over time with event markers ─────────────────────
plt.figure(figsize=(10, 4))
for n in to_plot:
    y = np.ma.masked_invalid(cx_smooth[n])  # Mask NaNs to break line
    # Use solid line for RGB, dash-dot for others, dashed for fused
    style = "--" if n == "Fused" else ("-" if n == "RGB" else "-.")
    plt.plot(times, y, label=n, linestyle=style, linewidth=2)

# Add vertical lines and labels for each event
for label, t in events:
    plt.axvline(t, color="k", linestyle=":", linewidth=1)
    plt.text(
        t + 0.5, plt.ylim()[1] * 0.95, label,
        rotation=90, va="top", fontsize="small"
    )

plt.xlabel("Time (s)")
plt.ylabel("Mean Center X (px)")
plt.title("Satellite Center X Trajectories Over Time")
plt.legend(ncol=2, fontsize="small")
plt.grid(ls="--", alpha=0.4)
plt.tight_layout()

# ─── 9) Plot mean center Y over time with event markers ─────────────────────
plt.figure(figsize=(9, 4))
for n in to_plot:
    y = np.ma.masked_invalid(cy_smooth[n])
    style = "--" if n == "Fused" else ("-" if n == "RGB" else "-.")
    plt.plot(times, y, label=n, linestyle=style, linewidth=2)

for label, t in events:
    plt.axvline(t, color="k", linestyle=":", linewidth=1)
    plt.text(
        t + 0.5, plt.ylim()[1] * 0.72, label,
        rotation=90, va="top", fontsize="small"
    )

plt.xlabel("Time (s)")
plt.ylabel("Mean Center Y (px)")
plt.title("Satellite Center Y Trajectories Over Time")
plt.legend(ncol=1, fontsize="small")
plt.grid(ls="--", alpha=0.4)
plt.tight_layout()

plt.show()  # Display all figures
