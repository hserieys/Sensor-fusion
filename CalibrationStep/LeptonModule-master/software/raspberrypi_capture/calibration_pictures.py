import cv2
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
from matplotlib.patches import Patch

# Global flag for plotting
global plt_flag
plt_flag = True

# -------------------------------------------------------------------
# Sensor Helper Functions
# -------------------------------------------------------------------
def get_rgb(path):
    """
    Extract the calibration contours from the rgb camera
    """
    # Load from the specified folder
    fn = os.path.join(path, "captured_image1m.npy")
    img = np.load(fn)

    # Split the channels
    b, g, r = cv2.split(img)
    kernel = np.ones((5, 5), np.uint8)
    
    # Process Blue channel
    b_blur = cv2.GaussianBlur(b, (5, 5), 0)
    _, edges_blue = cv2.threshold(b_blur, 0, 255, cv2.THRESH_OTSU)
    edges_blue = cv2.dilate(edges_blue, kernel, iterations=2)
    blue_display = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])
    b_contours = []
    contours_blue, _ = cv2.findContours(edges_blue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, c in enumerate(contours_blue):
        area = cv2.contourArea(c)
        if area < 1e1 or 5e4 < area:
            continue
        b_contours.append(c)
        cv2.drawContours(blue_display, contours_blue, i, (255, 0, 0), 2)
    
    cv2.imwrite(os.path.join(path, "blue.jpg"), blue_display)
    b_contours, ell_b = reduce_near_contours(b_contours, b)
    
    # Process Green channel
    g_blur = cv2.GaussianBlur(g, (5, 5), 0)
    _, edges_green = cv2.threshold(g_blur, 0, 255, cv2.THRESH_OTSU)
    edges_green = cv2.dilate(edges_green, kernel, iterations=2)
    green_display = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
    g_contours = []
    contours_green, _ = cv2.findContours(edges_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours_green):
        area = cv2.contourArea(c)
        if area < 1e1 or 5e4 < area:
            continue
        g_contours.append(c)
        cv2.drawContours(green_display, contours_green, i, (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(path, "green.jpg"), green_display)
    g_contours, ell_g = reduce_near_contours(g_contours, g)
        
    # Process Red channel
    r_blur = cv2.GaussianBlur(r, (5, 5), 0)
    _, edges_red = cv2.threshold(r_blur, 0, 255, cv2.THRESH_OTSU)
    edges_red = cv2.dilate(edges_red, kernel, iterations=2)
    red_display = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])
    r_contours = []
    contours_red, _ = cv2.findContours(edges_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours_red):
        area = cv2.contourArea(c)
        if area < 1e3 or 5e4 < area:
            continue
        r_contours.append(c)
        cv2.drawContours(red_display, contours_red, i, (0, 0, 255), 2)
    
    cv2.imwrite(os.path.join(path, "red.jpg"), red_display)
    r_contours, ell_r = reduce_near_contours(r_contours, red_display)

    return ell_r, ell_g, ell_b, r_contours, g_contours, b_contours


def get_flir(path):
    """
    Extract the calibration contours from the Flir thermal camera
    """
    thermal_img = "thermal_raw1m.png"
    full_fn = os.path.join(path, thermal_img)
    flir = cv2.imread(full_fn, cv2.IMREAD_UNCHANGED)
    flir = cv2.rotate(flir, cv2.ROTATE_180)
     
    denoised_frame = cv2.bilateralFilter(flir, 7, 150, 150)
    ret, thresh = cv2.threshold(denoised_frame, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    display_frame = cv2.cvtColor(denoised_frame, cv2.COLOR_GRAY2BGR)
    
    flir_contours = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 1e1 or 1e3 < area:
            continue
        flir_contours.append(c)
        cv2.drawContours(display_frame, contours, i, (0, 255, 0), 1)
    
    cv2.imwrite(os.path.join(path, "thermal.jpg"), display_frame)
    flir_contours, ell_flir = reduce_near_contours(flir_contours, display_frame)
    
    return flir_contours, ell_flir


def get_lidar(path):
    """
    Extract the calibration contours from the LIDAR camera
    """
    fn = os.path.join(path, "depth_raw1m.npy")
    dpt_img = np.load(fn)
    dpt_contours = []
    depth_img_16 = dpt_img
    depth_img_16 = np.fliplr(depth_img_16)  
    depth_img_16[depth_img_16 == 0] = np.max(depth_img_16)
    
    depth_frame = cv2.normalize(depth_img_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ret, thresh = cv2.threshold(depth_frame, 0, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
    contour_frame = depth_frame.copy()

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 1e1 or 1e4 < area:
            continue
        dpt_contours.append(c)
        cv2.drawContours(contour_frame, contours, i, (0, 0, 255), 2)
    
    cv2.imwrite(os.path.join(path, "depth.jpg"), contour_frame)
    dpt_contours, ell_dpt = reduce_near_contours(dpt_contours, contour_frame)

    return dpt_contours, ell_dpt, depth_frame

# -------------------------------------------------------------------
# Utility Functions for Calibration
# -------------------------------------------------------------------
def reduce_near_contours(contours, img):
    """
    Combines nearby contours into a single contour and fits ellipses.
    
    Returns:
      contour2: filtered list of contours.
      ell2: list of ellipse objects fitted to the filtered contours.
    """
    ell = []
    for c in contours:
        if len(c) >= 5:
            ell.append(cv2.fitEllipse(c))
    nr_contour = len(contours)
    dist_border = img.shape[1] / 10
    pic_area = img.shape[0] * img.shape[1]
    contour2 = []
    remove = []
    for i in range(nr_contour):
        if i in remove:
            continue
        contour = contours[i]
        for j in range(nr_contour):
            if j < i or j in remove:
                continue
            x1, y1 = ell[i][0]
            x2, y2 = ell[j][0]
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dist < dist_border and i != j:
                contour = np.concatenate((contour, contours[j]))
                ell[i] = cv2.fitEllipse(contour)
                remove.append(j)
        area = cv2.contourArea(contour)
        rate = area / pic_area
        if 0.003 < rate < 100:
            contour2.append(contour)
    ell2 = []
    for c in contour2:
        if len(c) >= 5:
            ell2.append(cv2.fitEllipse(c))
    return contour2, ell2

def near_point(ellip, ellip_ref):
    """
    Finds the nearest points between two sets of ellipse objects.
    
    Returns:
      An array of points [x1, x2, y1, y2] for each pairing.
    """
    near_pt = []
    for i in ellip:
        min_d = 1e3
        for j in ellip_ref:
            x1, y1 = j[0]
            x2, y2 = i[0]
            d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if d < min_d:
                min_d = d
                points = [x1, x2, y1, y2]
        near_pt.append(points)
    return np.array(near_pt)

def min_dist_rt(x):
    """
    Optimization function used when rotation is included.
    x = [scale, dx, dy, rotation]
    """
    return np.sum(np.sqrt(
        (points[:, 0] - x[0] * ((points[:, 1] * np.cos(x[3]) - points[:, 3] * np.sin(x[3])) - x[1])) ** 2 +
        (points[:, 2] - x[0] * ((points[:, 1] * np.sin(x[3]) + points[:, 3] * np.cos(x[3])) - x[2])) ** 2
    ))

def min_dist(x):
    """
    Optimization function used when only scale and translation are considered.
    x = [scale, dx, dy]
    """
    return np.sum(np.sqrt(
        (points[:, 0] - x[0] * (points[:, 1] - x[1])) ** 2 +
        (points[:, 2] - x[0] * (points[:, 3] - x[2])) ** 2
    ))

def dilatation(val):
    """
    Dilates the image to enhance calibration features.
    """
    dilatation_size = 7
    element = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(val, None, element)
    return dilatation_dst

# -------------------------------------------------------------------
# Calibration Routine (with real-time depth display)
# -------------------------------------------------------------------
def calibrate(path):
    """
    Computes calibration constants comparing the different sensor sources and
    produces a post-calibration (depth) frame with overlaid ellipses and their centers.
    
    Returns:
      cal_sdd, cal_sgb, cal_srb, cal_sdb, cal_sfb – calibration parameters.
      post_calibration – the depth frame with overlaid sensor data.
    """
    global points
    
    # Retrieve calibration data from each sensor.
    ell_r, ell_g, ell_b, r_contours, g_contours, b_contours = get_rgb(path)
    flir_contours, ell_flir = get_flir(path)
    dpt_contours, ell_dpt, depth_frame = get_lidar(path)
    
    # Create a pre-calibration visualization image on a white background.
    pre_calibration = np.full((480, 640, 3), 255, dtype=np.uint8)
    for i in ell_b:
        cv2.ellipse(pre_calibration, i, (255, 0, 0), 5)
    for i in ell_g:
        cv2.ellipse(pre_calibration, i, (0, 255, 0), 3)
    for i in ell_r:
        cv2.ellipse(pre_calibration, i, (0, 0, 255), 2)
    for i in ell_dpt:
        cv2.ellipse(pre_calibration, i, (0, 0, 0), 2)
    
    # Process FLIR ellipses: scale up for visualization
    ell_flir_p = []
    for i in ell_flir:
        i_p = ((8 * i[0][0], 8 * i[0][1]), (8 * i[1][0], 8 * i[1][1]), i[2])
        ell_flir_p.append(i_p)
        cv2.ellipse(pre_calibration, i_p, (255, 127, 127), 2)
    
    if plt_flag:
        fig, ax = plt.subplots()
        ax.matshow(pre_calibration)
        ax.set_title("Pre-Calibration Visualization")
        # Legend for sensor ellipses
        legend_elements = [
            Patch(facecolor='blue', edgecolor='blue', label='Blue Ellipses'),
            Patch(facecolor='green', edgecolor='green', label='Green Ellipses'),
            Patch(facecolor='red', edgecolor='red', label='Red Ellipses'),
            Patch(facecolor='black', edgecolor='black', label='Depth Ellipses'),
            Patch(facecolor=(1.0,0.5,0.5), edgecolor=(1.0,0.5,0.5), label='FLIR Ellipses'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
        plt.show()
    # Determine nearest points for calibration between the depth sensor and each other sensor
    near_rb = near_point(ell_r, ell_dpt)
    near_gb = near_point(ell_g, ell_dpt)
    near_db = near_point(ell_b, ell_dpt)
    near_fb = near_point(ell_flir_p, ell_dpt)

    x0 = [1, 0, 0, 0]

    # For red channel calibration
    points = near_rb
    if points.size == 0:
        cal_srb = [1, 0, 0, 0]
    else:
        response = minimize(min_dist, x0, method='TNC', options={'disp': False})
        cal_srb = response.x

    # For green channel calibration
    points = near_gb
    if points.size == 0:
        cal_sgb = [1, 0, 0, 0]
    else:
        response = minimize(min_dist, x0, method='TNC', options={'disp': False})
        cal_sgb = response.x

    # For blue channel calibration
    points = near_db
    if points.size == 0:
        cal_sdb = [1, 0, 0, 0]
    else:
        response = minimize(min_dist, x0, method='TNC', options={'disp': False})
        cal_sdb = response.x

    points = near_fb
    if points.size == 0:
        cal_sfb = [1, 0, 0, 0]
    else:
        response = minimize(min_dist_rt, x0, method='TNC', options={'disp': False})
        cal_sfb = response.x
        for _ in range(4):
            response = minimize(min_dist_rt, cal_sfb, method='TNC', options={'disp': False})
            cal_sfb = response.x
        response = minimize(min_dist, cal_sfb, method='TNC', options={'disp': False})
        cal_sfb = response.x

    # For demonstration, the depth sensor calibration is a dummy constant.
    cal_sdd = [1.0, 0.0, 0.0, 0.0]
    
    # Build a post-calibration visualization image (start with a nearly white background).
    post_calibration = np.full((480, 640, 3), 245, dtype=np.uint8)
    for i in ell_dpt:
        cv2.ellipse(post_calibration, i, (0, 0, 0), 5)
    for i in ell_g:
        i_p = ((cal_sgb[0] * ((i[0][0] * np.cos(cal_sgb[3]) - i[0][1] * np.sin(cal_sgb[3])) - cal_sgb[1]),
                cal_sgb[0] * ((i[0][0] * np.sin(cal_sgb[3]) + i[0][1] * np.cos(cal_sgb[3])) - cal_sgb[2])),
               (cal_sgb[0] * i[1][0], cal_sgb[0] * i[1][1]), i[2])
        cv2.ellipse(post_calibration, i_p, (0, 255, 0), 3)
    for i in ell_r:
        i_p = ((cal_srb[0] * ((i[0][0] * np.cos(cal_srb[3]) - i[0][1] * np.sin(cal_srb[3])) - cal_srb[1]),
                cal_srb[0] * ((i[0][0] * np.sin(cal_srb[3]) + i[0][1] * np.cos(cal_srb[3])) - cal_srb[2])),
               (cal_srb[0] * i[1][0], cal_srb[0] * i[1][1]), i[2])
        cv2.ellipse(post_calibration, i_p, (0, 0, 255), 2)
    for i in ell_b:
        i_p = ((cal_sdb[0] * ((i[0][0] * np.cos(cal_sdb[3]) - i[0][1] * np.sin(cal_sdb[3])) - cal_sdb[1]),
                cal_sdb[0] * ((i[0][0] * np.sin(cal_sdb[3]) + i[0][1] * np.cos(cal_sdb[3])) - cal_sdb[2])),
               (cal_sdb[0] * i[1][0], cal_sdb[0] * i[1][1]), i[2])
        cv2.ellipse(post_calibration, i_p, (255, 0, 0), 2)
    for i in ell_flir_p:
        i_p = ((cal_sfb[0] * ((i[0][0] * np.cos(cal_sfb[3]) - i[0][1] * np.sin(cal_sfb[3])) - cal_sfb[1]),
                cal_sfb[0] * ((i[0][0] * np.sin(cal_sfb[3]) + i[0][1] * np.cos(cal_sfb[3])) - cal_sfb[2])),
               (cal_sfb[0] * i[1][0], cal_sfb[0] * i[1][1]), i[2])
        cv2.ellipse(post_calibration, i_p, (255, 127, 127), 2)
    
    # ---- Draw the centers of each ellipse on the depth frame ----
    # Depth sensor centers
    for e in ell_dpt:
        center = (int(e[0][0]), int(e[0][1]))
        cv2.circle(post_calibration, center, 3, (0, 0, 0), -1)
    # Green sensor centers
    for e in ell_g:
        center = (
            int(cal_sgb[0] * ((e[0][0] * np.cos(cal_sgb[3]) - e[0][1] * np.sin(cal_sgb[3])) - cal_sgb[1])),
            int(cal_sgb[0] * ((e[0][0] * np.sin(cal_sgb[3]) + e[0][1] * np.cos(cal_sgb[3])) - cal_sgb[2]))
        )
        cv2.circle(post_calibration, center, 3, (0, 255, 0), -1)
    # Red sensor centers
    for e in ell_r:
        center = (
            int(cal_srb[0] * ((e[0][0] * np.cos(cal_srb[3]) - e[0][1] * np.sin(cal_srb[3])) - cal_srb[1])),
            int(cal_srb[0] * ((e[0][0] * np.sin(cal_srb[3]) + e[0][1] * np.cos(cal_srb[3])) - cal_srb[2]))
        )
        cv2.circle(post_calibration, center, 3, (0, 0, 255), -1)
    # Blue sensor centers
    for e in ell_b:
        center = (
            int(cal_sdb[0] * ((e[0][0] * np.cos(cal_sdb[3]) - e[0][1] * np.sin(cal_sdb[3])) - cal_sdb[1])),
            int(cal_sdb[0] * ((e[0][0] * np.sin(cal_sdb[3]) + e[0][1] * np.cos(cal_sdb[3])) - cal_sdb[2]))
        )
        cv2.circle(post_calibration, center, 3, (255, 0, 0), -1)
    # FLIR sensor centers
    for e in ell_flir_p:
        center = (
            int(cal_sfb[0] * ((e[0][0] * np.cos(cal_sfb[3]) - e[0][1] * np.sin(cal_sfb[3])) - cal_sfb[1])),
            int(cal_sfb[0] * ((e[0][0] * np.sin(cal_sfb[3]) + e[0][1] * np.cos(cal_sfb[3])) - cal_sfb[2]))
        )
        cv2.circle(post_calibration, center, 3, (255, 127, 127), -1)
    
    if plt_flag:
        fig, ax = plt.subplots()
        ax.matshow(post_calibration)
        ax.set_title("Post-Calibration Visualization")
        # Legend for centers
        legend_elements = [
            Patch(facecolor='black', edgecolor='black', label='Depth Ellipses'),
            Patch(facecolor='green', edgecolor='green', label='Green Ellipses'),
            Patch(facecolor='red', edgecolor='red', label='Red Ellipses'),
            Patch(facecolor='blue', edgecolor='blue', label='Blue Ellipses'),
            Patch(facecolor=(1.0,0.5,0.5), edgecolor=(1.0,0.5,0.5), label='FLIR Ellipses'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
        plt.show()
    
    return cal_sdd, cal_sgb, cal_srb, cal_sdb, cal_sfb, post_calibration

# -------------------------------------------------------------------
# Main Workflow: Cycle Through Calibration Folders, Save Results, and Display Depth Frame in Real Time
# -------------------------------------------------------------------
if __name__ == "__main__":
    root = os.getcwd()
    # Define which files must exist
    needed = ["captured_image1m.npy", "depth_raw1m.npy", "thermal_raw1m.png"]
    # Find folders containing all needed files
    folder_list = [d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))
                   and all(os.path.exists(os.path.join(root, d, fn)) for fn in needed)]

    print("Working directory:", root)
    print("Found Calibration folders:", folder_list)
    
    cal_sdd_ar, cal_sgb_ar, cal_srb_ar, cal_sdb_ar, cal_sfb_ar = [], [], [], [], []
    depth_frame_display = None

    for folder in folder_list:
        folder_path = os.path.join(root, folder)
        print("Processing calibration folder:", folder_path)
        # Retrieve calibration parameters and the post_calibration image
        cal_sdd, cal_sgb, cal_srb, cal_sdb, cal_sfb, post_calib_img = calibrate(folder_path)
        print("Calibration data for", folder, ":", cal_sdd, cal_sgb, cal_srb, cal_sdb, cal_sfb)

        cal_sdd_ar.append(cal_sdd)
        cal_sgb_ar.append(cal_sgb)
        cal_srb_ar.append(cal_srb)
        cal_sdb_ar.append(cal_sdb)
        cal_sfb_ar.append(cal_sfb)
        depth_frame_display = post_calib_img

    if not cal_sdd_ar:
        raise ValueError("No calibration data found. Please check your Calibration folders.")
    
    os.chdir(root)
    
    cal_sdd_ar = np.array(cal_sdd_ar)
    cal_sgb_ar = np.array(cal_sgb_ar)
    cal_srb_ar = np.array(cal_srb_ar)
    cal_sdb_ar = np.array(cal_sdb_ar)
    cal_sfb_ar = np.array(cal_sfb_ar)
    
    cal_sdd_ar = np.atleast_2d(cal_sdd_ar)
    cal_sgb_ar = np.atleast_2d(cal_sgb_ar)
    cal_srb_ar = np.atleast_2d(cal_srb_ar)
    cal_sdb_ar = np.atleast_2d(cal_sdb_ar)
    cal_sfb_ar = np.atleast_2d(cal_sfb_ar)
    
    if cal_sdd_ar.ndim < 2 or cal_sdd_ar.shape[1] == 0:
        raise ValueError("cal_sdd_ar is empty. Calibration data indexing failed.")
    
    cal_sdd_ar = np.atleast_2d(cal_sdd_ar)
    
    cal_sdd_dict = {"scale": list(cal_sdd_ar[:, 0]),
                    "dx": list(cal_sdd_ar[:, 1]),
                    "dy": list(cal_sdd_ar[:, 2]),
                    "rotation": list(cal_sdd_ar[:, 3])}
    
    cal_sgb_dict = {"scale": list(cal_sgb_ar[:, 0]),
                    "dx": list(cal_sgb_ar[:, 1]),
                    "dy": list(cal_sgb_ar[:, 2]),
                    "rotation": list(cal_sgb_ar[:, 3])}
    
    cal_srb_dict = {"scale": list(cal_srb_ar[:, 0]),
                    "dx": list(cal_srb_ar[:, 1]),
                    "dy": list(cal_srb_ar[:, 2]),
                    "rotation": list(cal_srb_ar[:, 3])}
    
    cal_sdb_dict = {"scale": list(cal_sdb_ar[:, 0]),
                    "dx": list(cal_sdb_ar[:, 1]),
                    "dy": list(cal_sdb_ar[:, 2]),
                    "rotation": list(cal_sdb_ar[:, 3])}
    
    cal_sfb_dict = {"scale": list(cal_sfb_ar[:, 0]),
                    "dx": list(cal_sfb_ar[:, 1]),
                    "dy": list(cal_sfb_ar[:, 2]),
                    "rotation": list(cal_sfb_ar[:, 3])}
    
    def numpy_converter(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    
    with open('srb.json', "w") as outfile:
        json.dump(cal_srb_dict, outfile, default=numpy_converter)
    
    with open('sdd.json', "w") as outfile:
        json.dump(cal_sdd_dict, outfile, default=numpy_converter)
    
    with open('sgb.json', "w") as outfile:
        json.dump(cal_sgb_dict, outfile, default=numpy_converter)
    
    with open('sdb.json', "w") as outfile:
        json.dump(cal_sdb_dict, outfile, default=numpy_converter)
    
    with open('sfb.json', "w") as outfile:
        json.dump(cal_sfb_dict, outfile, default=numpy_converter)
    
    print("Calibration JSON files saved successfully.")
    
    # ---- Real-Time Depth Frame Display ----
    # Display the most recent depth frame (post_calibration image) with overlaid ellipse centers.
    if depth_frame_display is not None:
        while True:
            cv2.imshow("Depth Frame with Ellipse Centers", depth_frame_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
