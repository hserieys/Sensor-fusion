# --- Import necessary libraries ---
import cv2                           # OpenCV: image capture & processing
import csv                           # CSV I/O (not used currently but imported)
import os                            # Filesystem operations
import numpy as np                  # Numerical array operations
import matplotlib.pyplot as plt      # Plotting library (not used here but imported)
from scipy.optimize import minimize # Nonlinear optimization for calibration
import json                          # JSON read/write for saving calibration data
import ctypes                        # Interface to C shared libraries (for FLIR)
import threading                     # Thread support for concurrent sensor reads
import concurrent.futures            # ThreadPoolExecutor for timeouts
from primesense import openni2       # OpenNI2 interface for depth (LiDAR) sensor
from primesense import _openni2 as c_api  # Underlying C API types/constants

# -------------------------------------------------------------------
# Sensor Initialization Functions
# -------------------------------------------------------------------
def init_rgb():
    """Initializes the webcam and returns the cv2.VideoCapture object."""
    cap = cv2.VideoCapture(0, cv2.CAP_V4L)  # Open default camera using V4L backend
    if not cap.isOpened():
        print("Error: Could not access the RGB camera.")
        exit()  # Terminate if camera not available
    print("RGB camera opened successfully.")
    return cap

def init_flir():
    """
    Initializes the FLIR thermal camera via a C shared library.
    Returns a dict containing:
      - lib: the loaded C library
      - buffer_ptr: pointer to the raw frame buffer
      - frame_buffer: numpy byte array backing the buffer
      - FRAME_WIDTH, FRAME_HEIGHT: dimensions of the thermal image
    """
    # Load the FLIR capture shared library
    lib = ctypes.CDLL(
        '/home/pi/Sensor-fusion/CalibrationStep/'
        'LeptonModule-master/software/raspberrypi_capture/'
        'libraspberrypi_capture.so'
    )
    # Specify argument and return types for the C function
    lib.main.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
    lib.main.restype = ctypes.c_int

    # Thermal frame dimensions
    FRAME_WIDTH = 80
    FRAME_HEIGHT = 60
    BYTES_PER_PIXEL = 2
    FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * BYTES_PER_PIXEL

    # Allocate a raw byte buffer for the C library to fill
    frame_buffer = np.zeros(FRAME_SIZE, dtype=np.uint8)
    buffer_ptr = frame_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    return {
        "lib": lib,
        "buffer_ptr": buffer_ptr,
        "frame_buffer": frame_buffer,
        "FRAME_WIDTH": FRAME_WIDTH,
        "FRAME_HEIGHT": FRAME_HEIGHT
    }

def init_lidar():
    """
    Initializes OpenNI2 for the depth (LiDAR) stream.
    Returns a dict containing:
      - depth_stream: the depth data stream
      - device: the OpenNI2 device handle
    """
    # Initialize OpenNI2 with plugin path
    openni2.initialize(
        "/home/pi/Desktop/AstraSDK-v2.1.3-Linux-arm/"
        "AstraSDK-v2.1.3-94bca0f52e-20210611T022735Z-Linux-arm/"
        "lib/Plugins/openni2"
    )
    dev = openni2.Device.open_any()       # Open any connected depth device
    depth_stream = dev.create_depth_stream()
    # Configure depth stream: 640×480 px @ 30 fps, 1 mm precision
    depth_stream.set_video_mode(c_api.OniVideoMode(
        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
        resolutionX=640, resolutionY=480, fps=30
    ))
    depth_stream.start()
    return {"depth_stream": depth_stream, "device": dev}

def call_lib_main_with_timeout(lib, buffer_ptr, timeout_sec=1.0):
    """
    Calls the C library's main() function with a timeout.
    Returns the result code, or -1 on timeout.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lib.main, buffer_ptr)
        try:
            result = future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            print("lib.main() call timed out!")
            result = -1
    return result

# -------------------------------------------------------------------
# Sensor “get” Functions
# -------------------------------------------------------------------
def get_rgb(cap):
    """
    Captures one frame from the RGB camera, processes each color channel
    to detect contours, fits ellipses, and returns:
      ell_r, ell_g, ell_b: lists of ellipse parameters for red, green, blue
      r_contours, g_contours, b_contours: raw contour lists
      red_display, green_display, blue_display: BGR images with drawn contours
    """
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        cap.release()
        return

    # Split into B, G, R channels
    b, g, r = cv2.split(frame)
    kernel = np.ones((5, 5), np.uint8)  # Structuring element for dilation

    # --- Blue channel processing ---
    b_blur = cv2.GaussianBlur(b, (5, 5), 0)
    _, edges_blue = cv2.threshold(b_blur, 0, 255, cv2.THRESH_OTSU)
    edges_blue = cv2.dilate(edges_blue, kernel, iterations=2)
    blue_display = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])
    b_contours = []
    contours_blue, _ = cv2.findContours(
        edges_blue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    for i, c in enumerate(contours_blue):
        area = cv2.contourArea(c)
        if area < 1e1 or area > 5e4:
            continue
        b_contours.append(c)
        cv2.drawContours(blue_display, contours_blue, i, (255, 0, 0), 2)
    b_contours, ell_b = reduce_near_contours(b_contours, b)

    # --- Green channel processing ---
    g_blur = cv2.GaussianBlur(g, (5, 5), 0)
    _, edges_green = cv2.threshold(g_blur, 0, 255, cv2.THRESH_OTSU)
    edges_green = cv2.dilate(edges_green, kernel, iterations=2)
    green_display = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
    g_contours = []
    contours_green, _ = cv2.findContours(
        edges_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    for i, c in enumerate(contours_green):
        area = cv2.contourArea(c)
        if area < 1e1 or area > 5e4:
            continue
        g_contours.append(c)
        cv2.drawContours(green_display, contours_green, i, (0, 255, 0), 2)
    g_contours, ell_g = reduce_near_contours(g_contours, g)

    # --- Red channel processing ---
    r_blur = cv2.GaussianBlur(r, (5, 5), 0)
    _, edges_red = cv2.threshold(r_blur, 0, 255, cv2.THRESH_OTSU)
    edges_red = cv2.dilate(edges_red, kernel, iterations=2)
    red_display = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])
    r_contours = []
    contours_red, _ = cv2.findContours(
        edges_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    for i, c in enumerate(contours_red):
        area = cv2.contourArea(c)
        if area < 1e3 or area > 5e4:
            continue
        r_contours.append(c)
        cv2.drawContours(red_display, contours_red, i, (0, 0, 255), 2)
    r_contours, ell_r = reduce_near_contours(r_contours, red_display)

    return (
        ell_r, ell_g, ell_b,
        r_contours, g_contours, b_contours,
        red_display, green_display, blue_display
    )

def get_flir(flir_params):
    """
    Captures one thermal frame via the FLIR library with timeout,
    normalizes, denoises, edge-detects, finds contours,
    fits ellipses, and returns:
      flir_contours: list of contours
      ell_flir: list of ellipse parameters
      flir_frame: BGR image with drawn contours
    """
    lib = flir_params["lib"]
    buffer_ptr = flir_params["buffer_ptr"]
    frame_buffer = flir_params["frame_buffer"]
    FRAME_WIDTH = flir_params["FRAME_WIDTH"]
    FRAME_HEIGHT = flir_params["FRAME_HEIGHT"]

    # Call C function with timeout
    result = call_lib_main_with_timeout(lib, buffer_ptr, timeout_sec=1.0)

    # Interpret raw buffer as 16-bit image
    frame_data = frame_buffer.view(dtype=np.uint16).reshape((FRAME_HEIGHT, FRAME_WIDTH))
    # Normalize to 0–255 and convert to uint8
    frame_norm = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX)
    flr_as_pic = frame_norm.astype('uint8')

    # Histogram-based contrast adjustment
    histr_flr = cv2.calcHist([flr_as_pic], [0], None, [256], [0,256])
    min_flr, max_flr = 0, 255
    while histr_flr[min_flr] < 5 and min_flr < 255:
        min_flr += 1
    while histr_flr[max_flr] < 5 and max_flr > 0:
        max_flr -= 1
    flr_m = 255.0 / (max_flr - min_flr)
    flr_n = -flr_m * min_flr
    flr_ir = ((np.clip(flr_as_pic, min_flr, max_flr) * flr_m) + flr_n).astype('uint8')

    # Denoise and rotate for correct orientation
    denoised_frame = cv2.bilateralFilter(flr_ir, 7, 170, 170)
    denoised_frame = cv2.rotate(denoised_frame, cv2.ROTATE_180)

    # Edge detection
    thresh = cv2.Canny(denoised_frame, 75, 200)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # Convert to BGR to draw colored contours
    flir_frame = cv2.cvtColor(denoised_frame, cv2.COLOR_GRAY2BGR)
    flir_contours = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 1e1 or area > 1e3:
            continue
        flir_contours.append(c)
        cv2.drawContours(flir_frame, contours, i, (0, 255, 0), 1)

    flir_contours, ell_flir = reduce_near_contours(flir_contours, flir_frame)
    return flir_contours, ell_flir, flir_frame

def get_lidar(lidar_params):
    """
    Captures one depth frame from the LiDAR, normalizes, thresholds,
    finds contours, draws them, fits ellipses, and returns:
      dpt_contours, ell_dpt, depth_frame (BGR)
    """
    depth_stream = lidar_params["depth_stream"]
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    depth_img_16 = np.frombuffer(frame_data, dtype=np.uint16).reshape((480, 640))

    # Flip horizontally and replace zeros with max depth
    depth_img_16 = np.fliplr(depth_img_16)
    depth_img_16[depth_img_16 == 0] = np.max(depth_img_16)

    # Normalize to 8-bit
    display_frame = cv2.normalize(
        depth_img_16, None, 0, 255,
        cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Threshold & find contours
    _, thresh = cv2.threshold(display_frame, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    depth_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
    dpt_contours = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 1e1 or area > 1e4:
            continue
        dpt_contours.append(c)
        cv2.drawContours(depth_frame, contours, i, (0, 0, 255), 2)

    dpt_contours, ell_dpt = reduce_near_contours(dpt_contours, depth_frame)
    return dpt_contours, ell_dpt, depth_frame

# -------------------------------------------------------------------
# Utility Functions for Contour Reduction & Calibration Helpers
# -------------------------------------------------------------------
def reduce_near_contours(contours, img):
    """
    Merges contours whose fitted ellipses centers lie within
    1/10 of image width, and filters by area ratio.
    Returns:
      contour2: merged contours list
      ell2: fitted ellipses list
    """
    # Pair each contour with its fitted ellipse (if ≥5 pts)
    pairs = [(c, cv2.fitEllipse(c)) for c in contours if len(c) >= 5]
    nr_pair = len(pairs)
    dist_border = img.shape[1] / 10  # merge threshold
    pic_area = img.shape[0] * img.shape[1]
    contour2 = []
    remove = set()

    for i in range(nr_pair):
        if i in remove:
            continue
        c_i, ell_i = pairs[i]
        for j in range(i+1, nr_pair):
            if j in remove:
                continue
            c_j, ell_j = pairs[j]
            x1, y1 = ell_i[0]
            x2, y2 = ell_j[0]
            d = np.hypot(x2 - x1, y2 - y1)
            if d < dist_border:
                # Merge close contours
                merged = np.concatenate((c_i, c_j))
                ell_i = cv2.fitEllipse(merged)
                pairs[i] = (merged, ell_i)
                remove.add(j)
        # Filter by area ratio
        area = cv2.contourArea(pairs[i][0])
        rate = area / pic_area
        if 0.0 < rate < 10:  # reasonable area proportion
            contour2.append(pairs[i][0])

    ell2 = [cv2.fitEllipse(c) for c in contour2 if len(c) >= 5]
    return contour2, ell2

def near_point(ellip, ellip_ref):
    """
    For each ellipse in `ellip`, find nearest ellipse center in `ellip_ref`.
    Returns an array of [x_ref, x, y_ref, y].
    """
    near_pt = []
    for e in ellip:
        min_d = float('inf')
        best_point = None
        for r in ellip_ref:
            x1, y1 = r[0]
            x2, y2 = e[0]
            d = np.hypot(x2 - x1, y2 - y1)
            if d < min_d:
                min_d, best_point = d, [x1, x2, y1, y2]
        near_pt.append(best_point or [0, 0, 0, 0])
    return np.array(near_pt)

def min_dist_rt(x):
    """
    Objective for rotating+translating FLIR against LiDAR points.
    Uses global `points` array.
    """
    return np.sum(np.hypot(
        points[:,0] - x[0]*((points[:,1]*np.cos(x[3]) - points[:,3]*np.sin(x[3])) - x[1]),
        points[:,2] - x[0]*((points[:,1]*np.sin(x[3]) + points[:,3]*np.cos(x[3])) - x[2])
    ))

def min_dist(x):
    """
    Objective for scaling+translating RGB/other sensors against LiDAR.
    Uses global `points` array.
    """
    return np.sum(np.hypot(
        points[:,0] - x[0]*(points[:,1] - x[1]),
        points[:,2] - x[0]*(points[:,3] - x[2])
    ))

def dilatation(val):
    """
    Applies morphological dilation with a 7-pixel radius structuring element.
    """
    dil_size = 7
    elem = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2*dil_size+1, 2*dil_size+1),
        (dil_size, dil_size)
    )
    return cv2.dilate(val, None, elem)

# -------------------------------------------------------------------
# Calibration Routine
# -------------------------------------------------------------------
def calibrate_sync(rgb_data, flir_data, lidar_data):
    """
    Performs synchronous calibration of RGB, FLIR, and LiDAR sensors.
    Returns calibration parameters [scale, dx, dy, rot] for each sensor
    and a visualization image `post_calibration`.
    """
    global points
    ell_r, ell_g, ell_b, r_contours, g_contours, b_contours, \
        red_display, green_display, blue_display = rgb_data
    flir_contours, ell_flir, flir_frame = flir_data
    dpt_contours, ell_dpt, depth_frame   = lidar_data

    # Pre-calibration visualization: white background 480×640
    pre_cal = np.full((480, 640, 3), 255, dtype=np.uint8)
    for e in ell_b:
        cv2.ellipse(pre_cal, e, (255,0,0), 5)
    for e in ell_g:
        cv2.ellipse(pre_cal, e, (0,255,0), 3)
    for e in ell_r:
        cv2.ellipse(pre_cal, e, (0,0,255), 2)
    for e in ell_dpt:
        cv2.ellipse(pre_cal, e, (0,0,0), 2)

    # Scale FLIR ellipses by 8× for visualization
    ell_flir_p = []
    for e in ell_flir:
        e_p = ((8*e[0][0], 8*e[0][1]), (8*e[1][0], 8*e[1][1]), e[2])
        ell_flir_p.append(e_p)
        cv2.ellipse(pre_cal, e_p, (255,127,127), 2)

    # Compute nearest-point correspondences
    near_rb = near_point(ell_r, ell_dpt)
    near_gb = near_point(ell_g, ell_dpt)
    near_db = near_point(ell_b, ell_dpt)
    near_fb = near_point(ell_flir_p, ell_dpt)
    x0 = [1, 0, 0, 0]

    # Optimize for each sensor
    points = near_rb
    if points.size == 0:
        cal_rb = [1, 0, 0, 0]
    else:
        cal_rb = minimize(min_dist, x0, method='TNC', options={'disp':False}).x

    points = near_gb
    if points.size == 0:
        cal_gb = [1, 0, 0, 0]
    else:
        cal_gb = minimize(min_dist, x0, method='TNC', options={'disp':False}).x

    points = near_db
    if points.size == 0:
        cal_db = [1, 0, 0, 0]
    else:
        cal_db = minimize(min_dist, x0, method='TNC', options={'disp':False}).x

    points = near_fb
    if points.size == 0:
        cal_fb = [1, 0, 0, 0]
    else:
        cal_fb = minimize(min_dist_rt, x0, method='TNC', options={'disp':False}).x

    # Post-calibration visualization
    post_cal = np.full((480, 640, 3), 245, dtype=np.uint8)
    # Draw LiDAR ellipses
    for e in ell_dpt:
        cv2.ellipse(post_cal, e, (0,0,0), 5)
    # Draw transformed sensor ellipses
    for e in ell_g:
        ep = (
            (cal_gb[0]*((e[0][0]*np.cos(cal_gb[3]) - e[0][1]*np.sin(cal_gb[3])) - cal_gb[1]),
             cal_gb[0]*((e[0][0]*np.sin(cal_gb[3]) + e[0][1]*np.cos(cal_gb[3])) - cal_gb[2])),
            (cal_gb[0]*e[1][0], cal_gb[0]*e[1][1]),
            e[2]
        )
        cv2.ellipse(post_cal, ep, (0,255,0), 3)
    for e in ell_r:
        ep = (
            (cal_rb[0]*((e[0][0]*np.cos(cal_rb[3]) - e[0][1]*np.sin(cal_rb[3])) - cal_rb[1]),
             cal_rb[0]*((e[0][0]*np.sin(cal_rb[3]) + e[0][1]*np.cos(cal_rb[3])) - cal_rb[2])),
            (cal_rb[0]*e[1][0], cal_rb[0]*e[1][1]),
            e[2]
        )
        cv2.ellipse(post_cal, ep, (0,0,255), 2)
    for e in ell_b:
        ep = (
            (cal_db[0]*((e[0][0]*np.cos(cal_db[3]) - e[0][1]*np.sin(cal_db[3])) - cal_db[1]),
             cal_db[0]*((e[0][0]*np.sin(cal_db[3]) + e[0][1]*np.cos(cal_db[3])) - cal_db[2])),
            (cal_db[0]*e[1][0], cal_db[0]*e[1][1]),
            e[2]
        )
        cv2.ellipse(post_cal, ep, (255,0,0), 2)
    for e in ell_flir_p:
        ep = (
            (cal_fb[0]*((e[0][0]*np.cos(cal_fb[3]) - e[0][1]*np.sin(cal_fb[3])) - cal_fb[1]),
             cal_fb[0]*((e[0][0]*np.sin(cal_fb[3]) + e[0][1]*np.cos(cal_fb[3])) - cal_fb[2])),
            (cal_fb[0]*e[1][0], cal_fb[0]*e[1][1]),
            e[2]
        )
        cv2.ellipse(post_cal, ep, (255,127,127), 2)

    return [1.0,0.0,0.0,0.0], cal_gb, cal_rb, cal_db, cal_fb, post_cal

def transform_point(point, calib):
    """
    Transform a point (x,y) from a sensor coordinate frame
    into the LiDAR frame using calibration [s, dx, dy, θ].
    Returns integer (x_new, y_new).
    """
    s, dx, dy, theta = calib
    x, y = point
    x_new = s * (x * np.cos(theta) - y * np.sin(theta)) - dx
    y_new = s * (x * np.sin(theta) + y * np.cos(theta)) - dy
    return int(x_new), int(y_new)

# -------------------------------------------------------------------
# Main Workflow: Real-Time Calibration & Visualization
# -------------------------------------------------------------------
if __name__ == "__main__":
    use_flir = True                              # Flag for using FLIR data
    cap = init_rgb()                             # Initialize RGB camera
    flir_params = init_flir()                    # Initialize FLIR camera
    lidar_params = init_lidar()                  # Initialize LiDAR device

    # Lists to accumulate calibration constants over frames
    cal_depth_list = []  # dummy depth
    cal_green_list = []
    cal_red_list = []
    cal_blue_list = []
    cal_flir_list = []

    while True:
        # Acquire FLIR data if enabled
        print("Acquiring FLIR frame...")
        flir_result = get_flir(flir_params) if use_flir else None
        if not flir_result:
            print("Disabling FLIR for subsequent calibrations.")
            use_flir = False
            flir_data = ([], [], None)
        else:
            flir_contours, ell_flir, flir_frame = flir_result
            if flir_frame is None or len(ell_flir) < 6:
                print(f"Insufficient FLIR ellipses ({len(ell_flir)}), disabling FLIR.")
                use_flir = False
                flir_data = ([], [], flir_frame)
            else:
                flir_data = (flir_contours, ell_flir, flir_frame)
                use_flir = False  # only one FLIR capture per run

        # Acquire LiDAR data
        print("Acquiring LiDAR frame...")
        dpt_contours, ell_dpt, depth_frame = get_lidar(lidar_params)
        if depth_frame is None or depth_frame.size == 0:
            print("Invalid LiDAR frame, retrying...")
            continue
        print("LiDAR frame acquired successfully.")

        # Acquire RGB data
        print("Acquiring RGB frames...")
        rgb_result = get_rgb(cap)
        if not rgb_result:
            print("Failed to acquire RGB frames, retrying...")
            continue
        (
            ell_r, ell_g, ell_b,
            r_contours, g_contours, b_contours,
            red_display, green_display, blue_display
        ) = rgb_result
        print("RGB frames acquired successfully.")

        # Pack sensor data
        rgb_data = (
            ell_r, ell_g, ell_b,
            r_contours, g_contours, b_contours,
            red_display, green_display, blue_display
        )
        lidar_data = (dpt_contours, ell_dpt, depth_frame)

        # Run calibration sync routine
        cal_dd, cal_gb, cal_rb, cal_db, cal_fb, post_calib_img = calibrate_sync(
            rgb_data, flir_data, lidar_data
        )

        # Prepare LiDAR display with calibrated centers overlaid
        lidar_display = depth_frame.copy()
        # Draw LiDAR ellipse centers (black)
        for e in ell_dpt:
            cx, cy = map(int, e[0]); cv2.circle(lidar_display, (cx, cy), 5, (0,0,0), -1)
        # Draw each sensor's transformed ellipse centers
        for e in ell_r:
            cx, cy = map(int, e[0]); pt = transform_point((cx, cy), cal_rb)
            cv2.circle(lidar_display, pt, 5, (0,0,255), -1)
        for e in ell_g:
            cx, cy = map(int, e[0]); pt = transform_point((cx, cy), cal_gb)
            cv2.circle(lidar_display, pt, 5, (0,255,0), -1)
        for e in ell_b:
            cx, cy = map(int, e[0]); pt = transform_point((cx, cy), cal_db)
            cv2.circle(lidar_display, pt, 5, (255,0,0), -1)
        for e in ell_flir:
            cx, cy = e[0][0]*8, e[0][1]*8  # scale FLIR coords
            pt = transform_point((cx, cy), cal_fb)
            cv2.circle(lidar_display, pt, 5, (255,0,127), -1)

        # Append calibration results
        cal_depth_list.append(cal_dd)
        cal_green_list.append(cal_gb)
        cal_red_list.append(cal_rb)
        cal_blue_list.append(cal_db)
        cal_flir_list.append(cal_fb)

        # Display results
        cv2.imshow("LiDAR with Calibrated Sensor Centers", lidar_display)
        if flir_frame is not None:
            cv2.imshow("FLIR Frame", flir_frame)
        # Debug print
        print("Calibration results:")
        print("Depth (dummy):", cal_dd)
        print("Green:", cal_gb)
        print("Red:", cal_rb)
        print("Blue:", cal_db)
        print("FLIR:", cal_fb)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup windows
    cv2.destroyAllWindows()

    # Save all collected calibration constants to JSON
    calibration_data = {
        "depth": cal_depth_list,
        "green": cal_green_list,
        "red":   cal_red_list,
        "blue":  cal_blue_list,
        "flir":  cal_flir_list
    }
    with open("calibration_data.json", "w") as jf:
        json.dump(
            calibration_data, jf, indent=4,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o
        )
    print("Calibration constants saved to calibration_data.json")
