import cv2
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
import ctypes
import threading
import concurrent.futures
from primesense import openni2
from primesense import _openni2 as c_api

# -------------------------------------------------------------------
# Sensor Initialization Functions
# -------------------------------------------------------------------
def init_rgb():
    """Initializes the webcam and returns the capture object."""
    cap = cv2.VideoCapture(0, cv2.CAP_V4L)
    if not cap.isOpened():
        print("Error: Could not access the RGB camera.")
        exit()
    print("RGB camera opened successfully.")
    return cap

def init_flir():
    """
    Initializes the thermal camera.
    
    Returns:
        dict: A dictionary containing the library, buffer pointer, frame buffer, 
              and frame dimensions.
    """
    lib = ctypes.CDLL('/home/pi/Documents/CalibrationStep/LeptonModule-master/software/raspberrypi_capture/libraspberrypi_capture.so')
    lib.main.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
    lib.main.restype = ctypes.c_int

    FRAME_WIDTH = 80
    FRAME_HEIGHT = 60
    BYTES_PER_PIXEL = 2
    FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * BYTES_PER_PIXEL

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
    Initializes OpenNI2 for the depth stream.
    
    Returns:
        dict: A dictionary containing the depth_stream and device.
    """
    openni2.initialize("/home/pi/Desktop/AstraSDK-v2.1.3-Linux-arm/AstraSDK-v2.1.3-94bca0f52e-20210611T022735Z-Linux-arm/lib/Plugins/openni2")
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    depth_stream.set_video_mode(c_api.OniVideoMode(
        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
        resolutionX=640, resolutionY=480, fps=30
    ))
    depth_stream.start()
    return {"depth_stream": depth_stream, "device": dev}

def call_lib_main_with_timeout(lib, buffer_ptr, timeout_sec=1.0):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lib.main, buffer_ptr)
        try:
            result = future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            print("lib.main() call timed out!")
            result = -1  # use an error code indicating a timeout
    return result

# -------------------------------------------------------------------
# Sensor Get Functions (Using the Initialized Objects)
# -------------------------------------------------------------------
def get_rgb(cap):
    """
    Extract the calibration contours from the RGB camera.
    
    Returns:
        ell_r, ell_g, ell_b : lists of ellipse objects for red, green, blue.
        r_contours, g_contours, b_contours : contour lists.
        red_display, green_display, blue_display : frames with drawn contours.
    """    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        cap.release()
        return

    b, g, r = cv2.split(frame)
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
    r_contours, ell_r = reduce_near_contours(r_contours, red_display)     

    return ell_r, ell_g, ell_b, r_contours, g_contours, b_contours, red_display, green_display, blue_display

def get_flir(flir_params):
    lib = flir_params["lib"]
    buffer_ptr = flir_params["buffer_ptr"]
    frame_buffer = flir_params["frame_buffer"]
    FRAME_WIDTH = flir_params["FRAME_WIDTH"]
    FRAME_HEIGHT = flir_params["FRAME_HEIGHT"]

    result = call_lib_main_with_timeout(lib, buffer_ptr, timeout_sec=1.0)
  
    # Interpret raw data as a 16-bit image and reshape it
    frame_data = frame_buffer.view(dtype=np.uint16).reshape((FRAME_HEIGHT, FRAME_WIDTH))
    
    # Normalize the 16-bit image into 8-bit and convert its scale
    # Assuming 'frame_data' is your original 16-bit frame (after reshaping)
    frame_norm = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX)
    flr_as_pic = frame_norm.astype('uint8')
    
    histr_flr = cv2.calcHist([flr_as_pic],[0],None,[256],[0,256])
    
    #FLIR
    min_flr = 0
    max_flr = 255
    while histr_flr[min_flr] < 5 or min_flr == 255:
        min_flr += 1
    
    while histr_flr[max_flr] < 5  or max_flr == 0:
        max_flr -= 1
    
    flr_m = 255/(max_flr-min_flr)
    flr_n = -flr_m*min_flr
    flr_ir = ((np.clip(flr_as_pic,min_flr,max_flr)*flr_m)+flr_n).astype('uint8')


    # Apply bilateral filtering for denoising
    denoised_frame = cv2.bilateralFilter(flr_ir, 7, 170, 170)
    denoised_frame = cv2.rotate(denoised_frame, cv2.ROTATE_180)
    # Threshold the image to create a binary image for contour detection
    #ret, thresh = cv2.threshold(denoised_frame, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.Canny(denoised_frame,75,200)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Convert the denoised image to BGR for drawing colored bounding boxes
    flir_frame = cv2.cvtColor(denoised_frame, cv2.COLOR_GRAY2BGR)
    flir_contours = []
    
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 1e1 or 1e3 < area:
            continue
        # Draw each contour only for visualisation purposes
        flir_contours.append(c)
        cv2.drawContours(flir_frame, contours, i, (0, 255, 0), 1)
    
    flir_contours, ell_flir = reduce_near_contours(flir_contours, flir_frame)
    
    return flir_contours, ell_flir, flir_frame

def get_lidar(lidar_params):
    depth_stream = lidar_params["depth_stream"]
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    depth_img_16 = np.frombuffer(frame_data, dtype=np.uint16).reshape((480, 640))
    dpt_contours = []
    depth_img_16 = np.fliplr(depth_img_16)
    depth_img_16[depth_img_16 == 0] = np.max(depth_img_16)
    display_frame = cv2.normalize(depth_img_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ret, thresh = cv2.threshold(display_frame, 0, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    depth_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 1e1 or 1e4 < area:
            continue
        dpt_contours.append(c)
        cv2.drawContours(depth_frame, contours, i, (0, 0, 255), 2)
    dpt_contours, ell_dpt = reduce_near_contours(dpt_contours, depth_frame)
    return dpt_contours, ell_dpt, depth_frame

# -------------------------------------------------------------------
# Utility Functions for Calibration (reduce_near_contours, near_point, etc.)
# -------------------------------------------------------------------

def reduce_near_contours(contours, img):
    # Create a list of pairs (contour, ellipse) for contours with enough points.
    pairs = [(c, cv2.fitEllipse(c)) for c in contours if len(c) >= 5]
    nr_pair = len(pairs)
    dist_border = img.shape[1] / 10
    pic_area = img.shape[0] * img.shape[1]
    contour2 = []
    remove = set()
    for i in range(nr_pair):
        if i in remove:
            continue
        c_i, ell_i = pairs[i]
        # Merge nearby contours.
        for j in range(i+1, nr_pair):
            if j in remove:
                continue
            c_j, ell_j = pairs[j]
            x1, y1 = ell_i[0]
            x2, y2 = ell_j[0]
            d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if d < dist_border:
                merged_contour = np.concatenate((c_i, c_j))
                ell_i = cv2.fitEllipse(merged_contour)
                pairs[i] = (merged_contour, ell_i)
                remove.add(j)
        area = cv2.contourArea(pairs[i][0])
        rate = area / pic_area
        if 0.0 < rate < 10:
            contour2.append(pairs[i][0])
    ell2 = [cv2.fitEllipse(c) for c in contour2 if len(c) >= 5]
    return contour2, ell2

def near_point(ellip, ellip_ref):
    near_pt = []
    for i in ellip:
        min_d = float('inf')
        best_point = None
        for j in ellip_ref:
            x1, y1 = j[0]
            x2, y2 = i[0]
            d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if d < min_d:
                min_d = d
                best_point = [x1, x2, y1, y2]
        if best_point is None:
            best_point = [0, 0, 0, 0]
        near_pt.append(best_point)
    return np.array(near_pt)


def min_dist_rt(x):
    return np.sum(np.sqrt(
        (points[:, 0] - x[0] * ((points[:, 1] * np.cos(x[3]) - points[:, 3] * np.sin(x[3])) - x[1])) ** 2 +
        (points[:, 2] - x[0] * ((points[:, 1] * np.sin(x[3]) + points[:, 3] * np.cos(x[3])) - x[2])) ** 2
    ))

def min_dist(x):
    return np.sum(np.sqrt(
        (points[:, 0] - x[0] * (points[:, 1] - x[1])) ** 2 +
        (points[:, 2] - x[0] * (points[:, 3] - x[2])) ** 2
    ))

def dilatation(val):
    dilatation_size = 7
    element = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(val, None, element)
    return dilatation_dst

# -------------------------------------------------------------------
# Calibration Routine
# -------------------------------------------------------------------
def calibrate_sync(rgb_data, flir_data, lidar_data):
    global points
    ell_r, ell_g, ell_b, r_contours, g_contours, b_contours, red_display, green_display, blue_display = rgb_data
    flir_contours, ell_flir, flir_frame = flir_data
    dpt_contours, ell_dpt, depth_frame = lidar_data
    
    # Pre-calibration visualization image
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
    
    # Determine nearest points for calibration between sensors and LiDAR
    near_rb = near_point(ell_r, ell_dpt)
    near_gb = near_point(ell_g, ell_dpt)
    near_db = near_point(ell_b, ell_dpt)
    near_fb = near_point(ell_flir_p, ell_dpt)
    x0 = [1, 0, 0, 0]

    points = near_rb
    if points.size == 0:
        cal_rb = [1, 0, 0, 0]
    else:
        response = minimize(min_dist, x0, method='TNC', options={'disp': False})
        cal_rb = response.x

    points = near_gb
    if points.size == 0:
        cal_gb = [1, 0, 0, 0]
    else:
        response = minimize(min_dist, x0, method='TNC', options={'disp': False})
        cal_gb = response.x

    points = near_db
    if points.size == 0:
        cal_db = [1, 0, 0, 0]
    else:
        response = minimize(min_dist, x0, method='TNC', options={'disp': False})
        cal_db = response.x

    points = near_fb
    if points.size == 0:
        cal_fb = [1, 0, 0, 0]
    else:
        response = minimize(min_dist_rt, x0, method='TNC', options={'disp': False})
        cal_fb = response.x
    #    for _ in range(4):
    #        response = minimize(min_dist_rt, cal_fb, method='TNC', options={'disp': False})
    #        cal_fb = response.x
    #    response = minimize(min_dist, cal_fb, method='TNC', options={'disp': False})
    #    cal_fb = response.x

    # Post-calibration visualization image
    post_calibration = np.full((480, 640, 3), 245, dtype=np.uint8)
    for i in ell_dpt:
        cv2.ellipse(post_calibration, i, (0, 0, 0), 5)
    for i in ell_g:
        i_p = ((cal_gb[0] * ((i[0][0] * np.cos(cal_gb[3]) - i[0][1] * np.sin(cal_gb[3])) - cal_gb[1]),
                cal_gb[0] * ((i[0][0] * np.sin(cal_gb[3]) + i[0][1] * np.cos(cal_gb[3])) - cal_gb[2])),
               (cal_gb[0] * i[1][0], cal_gb[0] * i[1][1]), i[2])
        cv2.ellipse(post_calibration, i_p, (0, 255, 0), 3)
    for i in ell_r:
        i_p = ((cal_rb[0] * ((i[0][0] * np.cos(cal_rb[3]) - i[0][1] * np.sin(cal_rb[3])) - cal_rb[1]),
                cal_rb[0] * ((i[0][0] * np.sin(cal_rb[3]) + i[0][1] * np.cos(cal_rb[3])) - cal_rb[2])),
               (cal_rb[0] * i[1][0], cal_rb[0] * i[1][1]), i[2])
        cv2.ellipse(post_calibration, i_p, (0, 0, 255), 2)
    for i in ell_b:
        i_p = ((cal_db[0] * ((i[0][0] * np.cos(cal_db[3]) - i[0][1] * np.sin(cal_db[3])) - cal_db[1]),
                cal_db[0] * ((i[0][0] * np.sin(cal_db[3]) + i[0][1] * np.cos(cal_db[3])) - cal_db[2])),
               (cal_db[0] * i[1][0], cal_db[0] * i[1][1]), i[2])
        cv2.ellipse(post_calibration, i_p, (255, 0, 0), 2)
    for i in ell_flir_p:
        i_p = ((cal_fb[0] * ((i[0][0] * np.cos(cal_fb[3]) - i[0][1] * np.sin(cal_fb[3])) - cal_fb[1]),
                cal_fb[0] * ((i[0][0] * np.sin(cal_fb[3]) + i[0][1] * np.cos(cal_fb[3])) - cal_fb[2])),
               (cal_fb[0] * i[1][0], cal_fb[0] * i[1][1]), i[2])
        cv2.ellipse(post_calibration, i_p, (255, 127, 127), 2)
    
    return [1.0, 0.0, 0.0, 0.0], cal_gb, cal_rb, cal_db, cal_fb, post_calibration

def transform_point(point, calib):
    """
    Transform a point (x, y) from a sensor coordinate system to the LiDAR coordinate system.
    
    Parameters:
      point: tuple (x, y)
      calib: list [scale, dx, dy, rotation]
      
    Returns:
      Transformed point (x_new, y_new) as integers.
    """
    s, dx, dy, theta = calib
    x, y = point
    x_new = s * (x * np.cos(theta) - y * np.sin(theta)) - dx
    y_new = s * (x * np.sin(theta) + y * np.cos(theta)) - dy
    return int(x_new), int(y_new)

# -------------------------------------------------------------------
# Main Workflow: Real-Time Calibration, Live Visualization, and Saving
# -------------------------------------------------------------------
if __name__ == "__main__":
    use_flir = True  # Flag to indicate whether to use FLIR for calibration
    cap = init_rgb()               # For RGB sensor
    flir_params = init_flir()        # For FLIR sensor
    lidar_params = init_lidar()      # For LiDAR sensor

    # Initialize lists for calibration constants
    cal_depth_list = []   # For dummy depth calibration constant
    cal_green_list = []   # For the green sensor
    cal_red_list = []     # For the red sensor
    cal_blue_list = []    # For the blue sensor
    cal_flir_list = []    # For the FLIR sensor
    
    
    while True:
        # Acquire FLIR frame only if use_flir is still True
        print("Acquiring FLIR frame...")
        flir_result = get_flir(flir_params)
        use_flir = True
        if flir_result is None:
            print("Failed to acquire FLIR frame, disabling FLIR for subsequent calibrations.")
            use_flir = False
            flir_data = ([], [], None)
        else:
            flir_contours, ell_flir, flir_frame = flir_result
            if flir_frame is None or flir_frame.size == 0:
                print("FLIR frame invalid, disabling FLIR for subsequent calibrations.")
                use_flir = False
                flir_data = ([], [], None)
            elif len(ell_flir) < 6:
                print(f"Not enough FLIR ellipses detected ({len(ell_flir)} found). Disabling FLIR for subsequent calibrations.")
                use_flir = False
                flir_data = ([], [], flir_frame)
            else:
                flir_data = (flir_contours, ell_flir, flir_frame)
                use_flir = False

        
        # Acquire LiDAR frame
        print("Acquiring LiDAR frame...")
        dpt_contours, ell_dpt, depth_frame = get_lidar(lidar_params)
        if depth_frame is None or depth_frame.size == 0:
            print("Failed to acquire a valid LiDAR frame.")
            continue
        print("LiDAR frame acquired successfully.")
        
        # Acquire RGB frames
        print("Acquiring RGB frames...")
        rgb_result = get_rgb(cap)
        if rgb_result is None:
            print("Failed to acquire valid RGB frames.")
            continue
        ell_r, ell_g, ell_b, r_contours, g_contours, b_contours, red_display, green_display, blue_display = rgb_result
        print("RGB frames acquired successfully.")
        
        # Pack sensor data into tuples
        flir_data = (flir_contours, ell_flir, flir_frame)
        rgb_data = (ell_r, ell_g, ell_b, r_contours, g_contours, b_contours, red_display, green_display, blue_display)
        lidar_data = (dpt_contours, ell_dpt, depth_frame)

        # Calibrate using synchronized sensor data; also get the post-calibration image.
        cal_dd, cal_gb, cal_rb, cal_db, cal_fb, post_calib_img = calibrate_sync(rgb_data, flir_data, lidar_data)
        
        # Create a copy of the LiDAR frame for drawing
        lidar_display = depth_frame.copy()
        
        # Draw LiDAR sensor ellipse centers (already in LiDAR coordinates) in black
        for ellipse in ell_dpt:
            center = ellipse[0]
            center_int = (int(center[0]), int(center[1]))
            cv2.circle(lidar_display, center_int, radius=5, color=(0, 0, 0), thickness=-1)
        
        # For each sensor, transform each ellipse center using the corresponding calibration constants
        # Red sensor (draw in red)
        for ellipse in ell_r:
            center = ellipse[0]
            center_calib = transform_point(center, cal_rb)
            cv2.circle(lidar_display, center_calib, radius=5, color=(0, 0, 255), thickness=-1)
        
        # Green sensor (draw in green)
        for ellipse in ell_g:
            center = ellipse[0]
            center_calib = transform_point(center, cal_gb)
            cv2.circle(lidar_display, center_calib, radius=5, color=(0, 255, 0), thickness=-1)
        
        # Blue sensor (draw in blue)
        for ellipse in ell_b:
            center = ellipse[0]
            center_calib = transform_point(center, cal_db)
            cv2.circle(lidar_display, center_calib, radius=5, color=(255, 0, 0), thickness=-1)
        
        # FLIR sensor (draw in yellow/cyan)
        for ellipse in ell_flir:
            center = ellipse[0]
            # Apply the scaling factor of 8 before transformation:
            scaled_center = (center[0] * 8, center[1] * 8)
            center_calib = transform_point(scaled_center, cal_fb)
            cv2.circle(lidar_display, center_calib, radius=5, color=(255, 0, 127), thickness=-1)
        
        # Append calibration constants to their corresponding lists
        cal_depth_list.append(cal_dd)
        cal_green_list.append(cal_gb)
        cal_red_list.append(cal_rb)
        cal_blue_list.append(cal_db)
        cal_flir_list.append(cal_fb)

        # Display the updated LiDAR frame with overlaid transformed centers
        cv2.imshow("LiDAR with Calibrated Sensor Centers", lidar_display)
        cv2.imshow("thermal.jpg", flir_frame)
        
        # For debugging: print calibration results
        print("Calibration results:")
        print("Depth (dummy):", cal_dd)
        print("Green:", cal_gb)
        print("Red:", cal_rb)
        print("Blue:", cal_db)
        print("FLIR:", cal_fb)
        
        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# After exiting the loop, save the calibration constants to a JSON file.
    calibration_data = {
        "depth": cal_depth_list,
        "green": cal_green_list,
        "red": cal_red_list,
        "blue": cal_blue_list,
        "flir": cal_flir_list
    }
    
    with open("calibration_data.json", "w") as json_file:
        json.dump(calibration_data, json_file, indent=4, 
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)

    
    print("Calibration constants saved to calibration_data.json")
