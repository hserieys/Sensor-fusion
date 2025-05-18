# --- Import necessary libraries ---
import cv2                             # OpenCV for image I/O and processing
import csv                             # CSV module (imported but not used in this script)
import os                              # Filesystem path operations
import numpy as np                    # Numerical array operations
import matplotlib.pyplot as plt       # Plotting library for visualization
from scipy.optimize import minimize   # Optimization routines for calibration
import json                            # JSON read/write for saving calibration data
from matplotlib.patches import Patch  # Legend patches for matplotlib plots

# Global flag to control whether to show matplotlib plots
global plt_flag
plt_flag = True

# -------------------------------------------------------------------
# Sensor Helper Functions
# -------------------------------------------------------------------
def get_rgb(path):
    """
    Extract calibration contours from an RGB image saved as a NumPy array.
    
    Parameters:
        path (str): Folder containing 'captured_image1m.npy'.
    Returns:
        ell_r, ell_g, ell_b: lists of ellipse parameters for each channel.
        r_contours, g_contours, b_contours: raw contour lists per channel.
    """
    # Build full filename and load image array
    fn = os.path.join(path, "captured_image1m.npy")
    img = np.load(fn)

    # Split into B, G, R channels
    b, g, r = cv2.split(img)

    # Structuring element for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # --- Blue channel processing ---
    b_blur = cv2.GaussianBlur(b, (5, 5), 0)                     # Smooth noise
    _, edges_blue = cv2.threshold(b_blur, 0, 255, cv2.THRESH_OTSU)  # Otsu threshold
    edges_blue = cv2.dilate(edges_blue, kernel, iterations=2)    # Close gaps
    # Create a display image: B in blue plane, zeros in G/R
    blue_display = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])
    b_contours = []
    # Find contours on binary edge image
    contours_blue, _ = cv2.findContours(edges_blue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours_blue):
        area = cv2.contourArea(c)
        # Filter by area to remove noise and overly large shapes
        if area < 1e1 or area > 5e4:
            continue
        b_contours.append(c)
        cv2.drawContours(blue_display, contours_blue, i, (255, 0, 0), 2)
    # Save the visualization for blue channel
    cv2.imwrite(os.path.join(path, "blue.jpg"), blue_display)
    # Merge nearby contours and fit ellipses
    b_contours, ell_b = reduce_near_contours(b_contours, b)

    # --- Green channel processing ---
    g_blur = cv2.GaussianBlur(g, (5, 5), 0)
    _, edges_green = cv2.threshold(g_blur, 0, 255, cv2.THRESH_OTSU)
    edges_green = cv2.dilate(edges_green, kernel, iterations=2)
    green_display = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
    g_contours = []
    contours_green, _ = cv2.findContours(edges_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours_green):
        area = cv2.contourArea(c)
        if area < 1e1 or area > 5e4:
            continue
        g_contours.append(c)
        cv2.drawContours(green_display, contours_green, i, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(path, "green.jpg"), green_display)
    g_contours, ell_g = reduce_near_contours(g_contours, g)

    # --- Red channel processing ---
    r_blur = cv2.GaussianBlur(r, (5, 5), 0)
    _, edges_red = cv2.threshold(r_blur, 0, 255, cv2.THRESH_OTSU)
    edges_red = cv2.dilate(edges_red, kernel, iterations=2)
    red_display = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])
    r_contours = []
    contours_red, _ = cv2.findContours(edges_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours_red):
        area = cv2.contourArea(c)
        # Note: red channel uses a higher minimum area (1e3)
        if area < 1e3 or area > 5e4:
            continue
        r_contours.append(c)
        cv2.drawContours(red_display, contours_red, i, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(path, "red.jpg"), red_display)
    r_contours, ell_r = reduce_near_contours(r_contours, red_display)

    # Return fitted ellipses and raw contours for each channel
    return ell_r, ell_g, ell_b, r_contours, g_contours, b_contours


def get_flir(path):
    """
    Extract calibration contours from a FLIR thermal image (PNG).
    
    Parameters:
        path (str): Folder containing 'thermal_raw1m.png'.
    Returns:
        flir_contours: list of raw contours.
        ell_flir: list of fitted ellipse parameters.
    """
    thermal_img = "thermal_raw1m.png"
    full_fn = os.path.join(path, thermal_img)
    flir = cv2.imread(full_fn, cv2.IMREAD_UNCHANGED)        # Load in original bit-depth
    flir = cv2.rotate(flir, cv2.ROTATE_180)                  # Rotate for correct orientation

    # Denoise with bilateral filter (preserves edges)
    denoised_frame = cv2.bilateralFilter(flir, 7, 150, 150)
    # Simple binary threshold to segment warm regions
    ret, thresh = cv2.threshold(denoised_frame, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Convert to BGR for drawing
    display_frame = cv2.cvtColor(denoised_frame, cv2.COLOR_GRAY2BGR)

    flir_contours = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 1e1 or area > 1e3:
            continue
        flir_contours.append(c)
        cv2.drawContours(display_frame, contours, i, (0, 255, 0), 1)
    # Save visualization
    cv2.imwrite(os.path.join(path, "thermal.jpg"), display_frame)
    # Merge near contours and fit ellipses
    flir_contours, ell_flir = reduce_near_contours(flir_contours, display_frame)

    return flir_contours, ell_flir


def get_lidar(path):
    """
    Extract calibration contours from a LiDAR depth array.
    
    Parameters:
        path (str): Folder containing 'depth_raw1m.npy'.
    Returns:
        dpt_contours: list of raw contours.
        ell_dpt: list of fitted ellipse parameters.
        depth_frame: normalized depth image (uint8 BGR).
    """
    fn = os.path.join(path, "depth_raw1m.npy")
    dpt_img = np.load(fn)                                   # Load 16-bit depth array
    depth_img_16 = np.fliplr(dpt_img)                       # Flip horizontally
    # Replace zero (no-data) pixels with max depth
    depth_img_16[depth_img_16 == 0] = np.max(depth_img_16)

    # Normalize to 8-bit for display
    depth_frame = cv2.normalize(
        depth_img_16, None, 0, 255,
        cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    # Threshold & contour detection
    ret, thresh = cv2.threshold(depth_frame, 0, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    depth_frame_color = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
    contour_frame = depth_frame_color.copy()

    dpt_contours = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 1e1 or area > 1e4:
            continue
        dpt_contours.append(c)
        cv2.drawContours(contour_frame, contours, i, (0, 0, 255), 2)
    # Save visualization
    cv2.imwrite(os.path.join(path, "depth.jpg"), contour_frame)
    # Merge near contours and fit ellipses
    dpt_contours, ell_dpt = reduce_near_contours(dpt_contours, contour_frame)

    return dpt_contours, ell_dpt, depth_frame_color


# -------------------------------------------------------------------
# Utility Functions for Calibration
# -------------------------------------------------------------------
def reduce_near_contours(contours, img):
    """
    Merge contours whose ellipse centers are within 1/10 of image width,
    then filter by area ratio, and fit ellipses.
    
    Parameters:
        contours (list): raw contour arrays.
        img (ndarray): reference image for size.
    Returns:
        contour2: merged and filtered contours.
        ell2: list of fitted ellipses for each contour2.
    """
    # Fit ellipse to each initial contour if possible
    ell = [cv2.fitEllipse(c) for c in contours if len(c) >= 5]
    nr_contour = len(contours)
    dist_border = img.shape[1] / 10   # merge threshold distance
    pic_area = img.shape[0] * img.shape[1]
    contour2 = []
    remove = []

    for i in range(nr_contour):
        if i in remove:
            continue
        contour = contours[i]
        for j in range(nr_contour):
            if j <= i or j in remove:
                continue
            x1, y1 = ell[i][0]
            x2, y2 = ell[j][0]
            dist = np.hypot(x2 - x1, y2 - y1)
            if dist < dist_border and i != j:
                # Merge contours and refit ellipse
                contour = np.concatenate((contour, contours[j]))
                ell[i] = cv2.fitEllipse(contour)
                remove.append(j)
        # Keep contour if its area ratio is within limits
        area = cv2.contourArea(contour)
        rate = area / pic_area
        if 0.003 < rate < 100:
            contour2.append(contour)

    # Fit ellipses to final contours
    ell2 = [cv2.fitEllipse(c) for c in contour2 if len(c) >= 5]
    return contour2, ell2


def near_point(ellip, ellip_ref):
    """
    Pair each ellipse in `ellip` with the nearest ellipse in `ellip_ref`.
    
    Returns:
        np.ndarray of shape (n,4): [x_ref, x, y_ref, y] for each pair.
    """
    near_pt = []
    for e in ellip:
        min_d = float('inf')
        best = None
        for r in ellip_ref:
            x1, y1 = r[0]
            x2, y2 = e[0]
            d = np.hypot(x2 - x1, y2 - y1)
            if d < min_d:
                min_d = d
                best = [x1, x2, y1, y2]
        near_pt.append(best)
    return np.array(near_pt)


def min_dist_rt(x):
    """
    Objective function for scale+translate+rotate calibration.
    Expects global `points` as Nx4 array.
    """
    return np.sum(np.sqrt(
        (points[:, 0] - x[0] * ((points[:, 1] * np.cos(x[3]) - points[:, 3] * np.sin(x[3])) - x[1]))**2 +
        (points[:, 2] - x[0] * ((points[:, 1] * np.sin(x[3]) + points[:, 3] * np.cos(x[3])) - x[2]))**2
    ))


def min_dist(x):
    """
    Objective function for scale+translate calibration (no rotation).
    Expects global `points` as Nx4 array.
    """
    return np.sum(np.sqrt(
        (points[:, 0] - x[0] * (points[:, 1] - x[1]))**2 +
        (points[:, 2] - x[0] * (points[:, 3] - x[2]))**2
    ))


def dilatation(val):
    """
    Apply morphological dilation with a 7-pixel radius structuring element.
    """
    dil_size = 7
    element = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2*dil_size + 1, 2*dil_size + 1),
        (dil_size, dil_size)
    )
    return cv2.dilate(val, None, element)


# -------------------------------------------------------------------
# Calibration Routine
# -------------------------------------------------------------------
def calibrate(path):
    """
    Compute calibration parameters between RGB, FLIR, and LiDAR sensors
    and produce a post-calibration visualization image.

    Returns:
        cal_sdd, cal_sgb, cal_srb, cal_sdb, cal_sfb: lists [scale, dx, dy, rot]
        post_calibration: BGR image with overlaid ellipses and centers
    """
    global points

    # Retrieve data from each sensor
    ell_r, ell_g, ell_b, _, _, _ = get_rgb(path)
    flir_contours, ell_flir = get_flir(path)
    dpt_contours, ell_dpt, depth_frame = get_lidar(path)

    # Pre-calibration visualization (white background)
    pre_cal = np.full((480, 640, 3), 255, dtype=np.uint8)
    for e in ell_b:
        cv2.ellipse(pre_cal, e, (255, 0, 0), 5)
    for e in ell_g:
        cv2.ellipse(pre_cal, e, (0, 255, 0), 3)
    for e in ell_r:
        cv2.ellipse(pre_cal, e, (0, 0, 255), 2)
    for e in ell_dpt:
        cv2.ellipse(pre_cal, e, (0, 0, 0), 2)

    # Scale FLIR ellipses by 8× for visualization
    ell_flir_p = []
    for e in ell_flir:
        e_p = ((8*e[0][0], 8*e[0][1]), (8*e[1][0], 8*e[1][1]), e[2])
        ell_flir_p.append(e_p)
        cv2.ellipse(pre_cal, e_p, (255, 127, 127), 2)

    # Optionally display pre-calibration figure
    if plt_flag:
        fig, ax = plt.subplots()
        ax.imshow(pre_cal)
        ax.set_title("Pre-Calibration Visualization")
        legend_elems = [
            Patch(color='blue', label='Blue ellipses'),
            Patch(color='green', label='Green ellipses'),
            Patch(color='red', label='Red ellipses'),
            Patch(color='black', label='Depth ellipses'),
            Patch(color=(1.0,0.5,0.5), label='FLIR ellipses'),
        ]
        ax.legend(handles=legend_elems, loc='upper right', fontsize='small')
        plt.show()

    # Compute nearest-point correspondences for each sensor vs LiDAR
    near_rb = near_point(ell_r, ell_dpt)
    near_gb = near_point(ell_g, ell_dpt)
    near_db = near_point(ell_b, ell_dpt)
    near_fb = near_point(ell_flir_p, ell_dpt)

    x0 = [1, 0, 0, 0]  # Initial guess [scale, dx, dy, rot]

    # Calibrate red
    points = near_rb
    if points.size == 0:
        cal_srb = [1, 0, 0, 0]
    else:
        cal_srb = minimize(min_dist, x0, method='TNC', options={'disp':False}).x

    # Calibrate green
    points = near_gb
    if points.size == 0:
        cal_sgb = [1, 0, 0, 0]
    else:
        cal_sgb = minimize(min_dist, x0, method='TNC', options={'disp':False}).x

    # Calibrate blue
    points = near_db
    if points.size == 0:
        cal_sdb = [1, 0, 0, 0]
    else:
        cal_sdb = minimize(min_dist, x0, method='TNC', options={'disp':False}).x

    # Calibrate FLIR (with rotation)
    points = near_fb
    if points.size == 0:
        cal_sfb = [1, 0, 0, 0]
    else:
        cal_sfb = minimize(min_dist_rt, x0, method='TNC', options={'disp':False}).x
        # refine with additional iterations
        for _ in range(4):
            cal_sfb = minimize(min_dist_rt, cal_sfb, method='TNC', options={'disp':False}).x
        cal_sfb = minimize(min_dist, cal_sfb, method='TNC', options={'disp':False}).x

    # Dummy calibration for depth sensor
    cal_sdd = [1.0, 0.0, 0.0, 0.0]

    # Post-calibration visualization (slightly gray background)
    post_cal = np.full((480, 640, 3), 245, dtype=np.uint8)
    for e in ell_dpt:
        cv2.ellipse(post_cal, e, (0, 0, 0), 5)
    # Overlay transformed sensor ellipses
    for e in ell_g:
        ep = (
            (cal_sgb[0]*((e[0][0]*np.cos(cal_sgb[3]) - e[0][1]*np.sin(cal_sgb[3])) - cal_sgb[1]),
             cal_sgb[0]*((e[0][0]*np.sin(cal_sgb[3]) + e[0][1]*np.cos(cal_sgb[3])) - cal_sgb[2])),
            (cal_sgb[0]*e[1][0], cal_sgb[0]*e[1][1]), e[2]
        )
        cv2.ellipse(post_cal, ep, (0, 255, 0), 3)
    for e in ell_r:
        ep = (
            (cal_srb[0]*((e[0][0]*np.cos(cal_srb[3]) - e[0][1]*np.sin(cal_srb[3])) - cal_srb[1]),
             cal_srb[0]*((e[0][0]*np.sin(cal_srb[3]) + e[0][1]*np.cos(cal_srb[3])) - cal_srb[2])),
            (cal_srb[0]*e[1][0], cal_srb[0]*e[1][1]), e[2]
        )
        cv2.ellipse(post_cal, ep, (0, 0, 255), 2)
    for e in ell_b:
        ep = (
            (cal_sdb[0]*((e[0][0]*np.cos(cal_sdb[3]) - e[0][1]*np.sin(cal_sdb[3])) - cal_sdb[1]),
             cal_sdb[0]*((e[0][0]*np.sin(cal_sdb[3]) + e[0][1]*np.cos(cal_sdb[3])) - cal_sdb[2])),
            (cal_sdb[0]*e[1][0], cal_sdb[0]*e[1][1]), e[2]
        )
        cv2.ellipse(post_cal, ep, (255, 0, 0), 2)
    for e in ell_flir_p:
        ep = (
            (cal_sfb[0]*((e[0][0]*np.cos(cal_sfb[3]) - e[0][1]*np.sin(cal_sfb[3])) - cal_sfb[1]),
             cal_sfb[0]*((e[0][0]*np.sin(cal_sfb[3]) + e[0][1]*np.cos(cal_sfb[3])) - cal_sfb[2])),
            (cal_sfb[0]*e[1][0], cal_sfb[0]*e[1][1]), e[2]
        )
        cv2.ellipse(post_cal, ep, (255, 127, 127), 2)

    # Draw ellipse centers on post_cal
    for e in ell_dpt:
        cx, cy = map(int, e[0]); cv2.circle(post_cal, (cx, cy), 3, (0, 0, 0), -1)
    for e in ell_g:
        cx = int(cal_sgb[0]*((e[0][0]*np.cos(cal_sgb[3]) - e[0][1]*np.sin(cal_sgb[3])) - cal_sgb[1]))
        cy = int(cal_sgb[0]*((e[0][0]*np.sin(cal_sgb[3]) + e[0][1]*np.cos(cal_sgb[3])) - cal_sgb[2]))
        cv2.circle(post_cal, (cx, cy), 3, (0, 255, 0), -1)
    for e in ell_r:
        cx = int(cal_srb[0]*((e[0][0]*np.cos(cal_srb[3]) - e[0][1]*np.sin(cal_srb[3])) - cal_srb[1]))
        cy = int(cal_srb[0]*((e[0][0]*np.sin(cal_srb[3]) + e[0][1]*np.cos(cal_srb[3])) - cal_srb[2]))
        cv2.circle(post_cal, (cx, cy), 3, (0, 0, 255), -1)
    for e in ell_b:
        cx = int(cal_sdb[0]*((e[0][0]*np.cos(cal_sdb[3]) - e[0][1]*np.sin(cal_sdb[3])) - cal_sdb[1]))
        cy = int(cal_sdb[0]*((e[0][0]*np.sin(cal_sdb[3]) + e[0][1]*np.cos(cal_sdb[3])) - cal_sdb[2]))
        cv2.circle(post_cal, (cx, cy), 3, (255, 0, 0), -1)
    for e in ell_flir_p:
        cx = int(cal_sfb[0]*((e[0][0]*np.cos(cal_sfb[3]) - e[0][1]*np.sin(cal_sfb[3])) - cal_sfb[1]))
        cy = int(cal_sfb[0]*((e[0][0]*np.sin(cal_sfb[3]) + e[0][1]*np.cos(cal_sfb[3])) - cal_sfb[2]))
        cv2.circle(post_cal, (cx, cy), 3, (255, 127, 127), -1)

    # Optionally display post-calibration figure
    if plt_flag:
        fig, ax = plt.subplots()
        ax.imshow(post_cal)
        ax.set_title("Post-Calibration Visualization")
        legend_elems = [
            Patch(color='black', label='Depth'),
            Patch(color='green', label='Green'),
            Patch(color='red', label='Red'),
            Patch(color='blue', label='Blue'),
            Patch(color=(1.0,0.5,0.5), label='FLIR'),
        ]
        ax.legend(handles=legend_elems, loc='upper right', fontsize='small')
        plt.show()

    return cal_sdd, cal_sgb, cal_srb, cal_sdb, cal_sfb, post_cal


# -------------------------------------------------------------------
# Main Workflow: Iterate Calibration Folders & Save Results
# -------------------------------------------------------------------
if __name__ == "__main__":
    root = os.getcwd()
    needed = ["captured_image1m.npy", "depth_raw1m.npy", "thermal_raw1m.png"]
    # Identify folders containing all required calibration files
    folder_list = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and
           all(os.path.exists(os.path.join(root, d, fn)) for fn in needed)
    ]

    print("Working directory:", root)
    print("Found calibration folders:", folder_list)

    # Arrays to collect calibration parameters
    cal_sdd_ar, cal_sgb_ar, cal_srb_ar, cal_sdb_ar, cal_sfb_ar = [], [], [], [], []
    last_depth_frame = None

    for folder in folder_list:
        folder_path = os.path.join(root, folder)
        print("Processing:", folder_path)
        # Run calibration routine on each folder
        cal_sdd, cal_sgb, cal_srb, cal_sdb, cal_sfb, post_img = calibrate(folder_path)
        print("Results:", cal_sdd, cal_sgb, cal_srb, cal_sdb, cal_sfb)

        cal_sdd_ar.append(cal_sdd)
        cal_sgb_ar.append(cal_sgb)
        cal_srb_ar.append(cal_srb)
        cal_sdb_ar.append(cal_sdb)
        cal_sfb_ar.append(cal_sfb)
        last_depth_frame = post_img

    if not cal_sdd_ar:
        raise ValueError("No calibration data found: check folders.")

    # Convert lists to numpy arrays for JSON-friendly structure
    cal_sdd_ar = np.atleast_2d(np.array(cal_sdd_ar))
    cal_sgb_ar = np.atleast_2d(np.array(cal_sgb_ar))
    cal_srb_ar = np.atleast_2d(np.array(cal_srb_ar))
    cal_sdb_ar = np.atleast_2d(np.array(cal_sdb_ar))
    cal_sfb_ar = np.atleast_2d(np.array(cal_sfb_ar))

    # Build dictionaries of parameter series
    def to_dict(arr):
        return {
            "scale":    list(arr[:,0]),
            "dx":       list(arr[:,1]),
            "dy":       list(arr[:,2]),
            "rotation": list(arr[:,3])
        }

    cal_sdd_dict = to_dict(cal_sdd_ar)
    cal_sgb_dict = to_dict(cal_sgb_ar)
    cal_srb_dict = to_dict(cal_srb_ar)
    cal_sdb_dict = to_dict(cal_sdb_ar)
    cal_sfb_dict = to_dict(cal_sfb_ar)

    # JSON serialization helper for numpy types
    def numpy_converter(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Type {type(o)} not serializable")

    # Save each calibration JSON
    with open('srb.json', "w") as f: json.dump(cal_srb_dict, f, default=numpy_converter, indent=4)
    with open('sdd.json', "w") as f: json.dump(cal_sdd_dict, f, default=numpy_converter, indent=4)
    with open('sgb.json', "w") as f: json.dump(cal_sgb_dict, f, default=numpy_converter, indent=4)
    with open('sdb.json', "w") as f: json.dump(cal_sdb_dict, f, default=numpy_converter, indent=4)
    with open('sfb.json', "w") as f: json.dump(cal_sfb_dict, f, default=numpy_converter, indent=4)

    print("Calibration JSON files saved successfully.")

    # Real-time display of last post-calibration depth frame
    if last_depth_frame is not None:
        while True:
            cv2.imshow("Depth Frame with Ellipse Centers", last_depth_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
