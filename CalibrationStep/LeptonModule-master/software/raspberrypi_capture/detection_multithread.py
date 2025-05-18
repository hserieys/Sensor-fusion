# --- Import necessary libraries ---
import cv2                    # OpenCV for image/video processing
import os                     # For filesystem operations
import numpy as np            # Numerical operations on arrays
import json                   # Read/write JSON files for calibration data and output
import ctypes                 # Interface to C libraries (for FLIR camera)
import time                   # Timestamps and delays
import threading              # Concurrent sensor-reading threads
from primesense import openni2           # OpenNI2 interface for depth (LiDAR) sensor
from primesense import _openni2 as c_api # Underlying C API constants/types

# ============ Global Variables & Constants ============
DILATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))      # Kernel for dilating RGB channel edges
DILATION_KERNEL_FLIR = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)) # Smaller kernel for thermal edges
zeros_template = None               # Placeholder for an empty image template

# Preallocated buffers (filled later in main)
dpt_norm_buffer = None              # Buffer for normalized depth frames
flir_norm_buffer = None             # Buffer for normalized FLIR frames

# Shared sensor data dictionary and a lock for thread-safe access
sensor_data = { 'rgb': None, 'lidar': None, 'flir': None }
data_lock = threading.Lock()
running = True  # Control flag to stop threads gracefully

# -------------------------------------------------------------------
# Sensor Initialization Functions
# -------------------------------------------------------------------
def init_rgb():
    """Initializes the webcam and returns the cv2.VideoCapture object."""
    cap = cv2.VideoCapture(0, cv2.CAP_V4L)  # Open default camera via V4L backend
    if not cap.isOpened():
        print("Error: Could not access the RGB camera.")
        exit()
    print("RGB camera opened successfully.")
    return cap

def init_lidar():
    """
    Initializes OpenNI2 for the depth (LiDAR) stream.
    
    Returns:
        dict: A dictionary containing the depth_stream and device handle.
    """
    # Point to the OpenNI2 module directory
    openni2.initialize(
        "/home/pi/Desktop/AstraSDK-v2.1.3-Linux-arm/"
        "AstraSDK-v2.1.3-94bca0f52e-20210611T022735Z-Linux-arm/"
        "lib/Plugins/openni2"
    )
    dev = openni2.Device.open_any()             # Open any connected device
    depth_stream = dev.create_depth_stream()     # Create depth stream from device
    # Configure the depth stream: 640×480 @ 30fps, 1mm precision
    depth_stream.set_video_mode(c_api.OniVideoMode(
        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
        resolutionX=640, resolutionY=480, fps=30
    ))
    depth_stream.start()                         # Begin streaming
    return {"depth_stream": depth_stream, "device": dev}

def init_flir():
    """
    Initializes the FLIR thermal camera via a C shared library.
    
    Returns:
        dict: Contains the library, buffer pointer, numpy buffer, and frame dims.
    """
    # Load the custom C library for FLIR capture
    lib = ctypes.CDLL(
        '/home/pi/Sensor-fusion/CalibrationStep/LeptonModule-master/'
        'software/raspberrypi_capture/libraspberrypi_capture.so'
    )
    # Define the argument/return types for the library entrypoint
    lib.main.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
    lib.main.restype = ctypes.c_int

    # Define the thermal frame dimensions and bytes per pixel
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

# -------------------------------------------------------------------
# Sensor “get” Functions
# -------------------------------------------------------------------
def get_rgb(cap):
    """
    Grabs a frame from the RGB camera, splits into channels,
    performs contrast check, edge detection, contour filtering,
    merges near contours, draws bounding boxes, and timestamps it.
    """
    ret, frame = cap.read()             # Capture a frame
    rgb_timestamp = time.time()         # Record timestamp
    if not ret:
        print("Error: Failed to grab frame.")
        cap.release()
        return

    # Split the BGR frame into individual channels
    b, g, r = cv2.split(frame)

    # Threshold for rejecting low-contrast frames
    INTENSITY_THRESHOLD = 80

    # Create display images for each channel by merging with zeros_template
    blue_display  = create_channel_display(b, 0)  # Blue channel in BGR
    green_display = create_channel_display(g, 1)  # Green channel in BGR
    red_display   = create_channel_display(r, 2)  # Red channel in BGR
    kernel = DILATION_KERNEL                      

    # --- Process Blue channel ---
    b_blur = cv2.GaussianBlur(b, (5, 5), 0)        # Smooth noise
    diff_b  = np.max(b_blur) - np.min(b_blur)      # Contrast range
    flag_b  = (diff_b >= INTENSITY_THRESHOLD)      # Enough contrast?
    if flag_b:
        # Otsu threshold, dilate to close gaps
        _, edges_blue = cv2.threshold(b_blur, 0, 255, cv2.THRESH_OTSU)
        edges_blue = cv2.dilate(edges_blue, kernel, iterations=2)
        # Find contours
        contours_blue, _ = cv2.findContours(
            edges_blue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        b_contours = []
        for c in contours_blue:
            area = cv2.contourArea(c)
            # Filter out too small or too large shapes
            if area < 1e2 or area > 5e4:
                continue
            b_contours.append(c)
        # Merge near contours, compute ellipses and bounding rects
        b_contours, ell_b, rect_b = reduce_near_contours(b_contours, b)
        # Draw each bounding box on the display
        for box in rect_b:
            cv2.rectangle(blue_display, box[0], box[1], (255, 255, 255), 4)
    else:
        b_contours, ell_b, rect_b = [], [], []  # No valid contours

    # --- Process Green channel (identical steps) ---
    g_blur = cv2.GaussianBlur(g, (5, 5), 0)
    diff_g  = np.max(g_blur) - np.min(g_blur)
    flag_g  = (diff_g >= INTENSITY_THRESHOLD)
    if flag_g:
        _, edges_green = cv2.threshold(g_blur, 0, 255, cv2.THRESH_OTSU)
        edges_green = cv2.dilate(edges_green, kernel, iterations=2)
        contours_green, _ = cv2.findContours(
            edges_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        g_contours = []
        for c in contours_green:
            area = cv2.contourArea(c)
            if area < 1e2 or area > 5e4:
                continue
            g_contours.append(c)
        g_contours, ell_g, rect_g = reduce_near_contours(g_contours, g)
        for box in rect_g:
            cv2.rectangle(green_display, box[0], box[1], (255, 255, 255), 4)
    else:
        g_contours, ell_g, rect_g = [], [], []

    # --- Process Red channel (identical steps) ---
    r_blur = cv2.GaussianBlur(r, (5, 5), 0)
    diff_r  = np.max(r_blur) - np.min(r_blur)
    flag_r  = (diff_r >= INTENSITY_THRESHOLD)
    if flag_r:
        _, edges_red = cv2.threshold(r_blur, 0, 255, cv2.THRESH_OTSU)
        edges_red = cv2.dilate(edges_red, kernel, iterations=2)
        contours_red, _ = cv2.findContours(
            edges_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        r_contours = []
        for c in contours_red:
            area = cv2.contourArea(c)
            if area < 1e2 or area > 5e4:
                continue
            r_contours.append(c)
        r_contours, ell_r, rect_r = reduce_near_contours(r_contours, r)
        for box in rect_r:
            cv2.rectangle(red_display, box[0], box[1], (255, 255, 255), 4)
    else:
        r_contours, ell_r, rect_r = [], [], []

    # Pack results: flags per channel and detailed data tuples
    channel_data = (
        (ell_r, r_contours, red_display, rect_r),
        (ell_g, g_contours, green_display, rect_g),
        (ell_b, b_contours, blue_display, rect_b)
    )
    flags = (flag_r, flag_g, flag_b)
    return rgb_timestamp, (flags, channel_data)

def get_lidar(lidar_params):
    """
    Reads one frame from the LiDAR depth stream, normalizes it,
    detects contours, merges near contours, draws bounding boxes,
    and returns timestamped results.
    """
    depth_stream = lidar_params["depth_stream"]
    frame = depth_stream.read_frame()    # Grab a depth frame
    lidar_timestamp = time.time()        # Timestamp

    # Convert raw buffer to 16-bit depth image
    frame_data = frame.get_buffer_as_uint16()
    depth_img_16 = np.frombuffer(frame_data, dtype=np.uint16).reshape((480, 640))

    # Replace zero (no-data) pixels with max depth, flip horizontally
    depth_img_16[depth_img_16 == 0] = np.max(depth_img_16)
    depth_img_16 = np.fliplr(depth_img_16)

    # Normalize to 8-bit for display in dpt_norm_buffer
    cv2.normalize(
        depth_img_16, dpt_norm_buffer,
        0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    display_frame = dpt_norm_buffer

    # Threshold and find contours on the normalized frame
    _, thresh = cv2.threshold(display_frame, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # Convert single-channel to BGR for colored drawing
    depth_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
    dpt_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1e2 or area > 1e5:
            continue
        dpt_contours.append(c)

    # Merge near contours, compute ellipses and bounding rects
    dpt_contours, ell_dpt, rect_dpt = reduce_near_contours(dpt_contours, depth_frame)
    for box in rect_dpt:
        cv2.rectangle(depth_frame, box[0], box[1], (0, 0, 0), 4)

    return lidar_timestamp, (dpt_contours, ell_dpt, depth_frame, rect_dpt)

def get_flir(flir_params):
    """
    Captures a thermal frame via the C library, normalizes/denoises it,
    performs edge detection, finds and filters contours, merges near contours,
    draws bounding boxes, and returns timestamped results.
    """
    lib          = flir_params["lib"]
    buffer_ptr   = flir_params["buffer_ptr"]
    frame_buffer = flir_params["frame_buffer"]
    FRAME_WIDTH  = flir_params["FRAME_WIDTH"]
    FRAME_HEIGHT = flir_params["FRAME_HEIGHT"]

    # Call into C library; non-zero result indicates error
    result = lib.main(buffer_ptr)
    flir_timestamp = time.time()
    if result != 0:
        print("Error capturing frame:", result)
        return None

    # Interpret buffer as 16-bit image and normalize to 8-bit
    frame_data = frame_buffer.view(dtype=np.uint16).reshape((FRAME_HEIGHT, FRAME_WIDTH))
    flr_as_pic = cv2.normalize(
        frame_data, None,
        0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    # Denoise with fast Non-local Means
    flr_as_pic = cv2.fastNlMeansDenoising(flr_as_pic, None, 90, 3, 9)

    # Compute histogram to find robust min/max
    histr_flr = cv2.calcHist([flr_as_pic], [0], None, [256], [0,256])
    min_flr, max_flr = 0, 255
    # Increase min until at least 5 pixels or reach end
    while histr_flr[min_flr] < 5 and min_flr < 255:
        min_flr += 1
    # Decrease max until at least 5 pixels or reach 0
    while histr_flr[max_flr] < 5 and max_flr > 0:
        max_flr -= 1

    # Linear contrast stretch between min_flr and max_flr
    flr_m = 255.0 / (max_flr - min_flr)
    flr_n = -flr_m * min_flr
    flr_ir = (
        (np.clip(flr_as_pic, min_flr, max_flr) * flr_m) + flr_n
    ).astype('uint8')

    # Rotate 180° for correct orientation
    denoised_frame = cv2.rotate(flr_ir, cv2.ROTATE_180)

    # Edge detection via Canny then dilate
    _, thresh = cv2.threshold(
        denoised_frame, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    canny = cv2.Canny(denoised_frame, 75, 200)
    canny = cv2.dilate(canny, DILATION_KERNEL_FLIR, iterations=1)
    contours, _ = cv2.findContours(
        canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # Convert to BGR so bounding boxes can be colored
    flir_frame   = cv2.cvtColor(denoised_frame, cv2.COLOR_GRAY2BGR)
    flir_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        # Filter out too-small or too-large regions
        if area < 1e1 or area > 1e3:
            continue
        flir_contours.append(c)

    # Merge near contours, compute ellipses and bounding rects
    flir_contours, ell_flir, rect_flir = reduce_near_contours(flir_contours, flir_frame)
    for box in rect_flir:
        cv2.rectangle(flir_frame, box[0], box[1], (0, 0, 255), 1)

    return flir_timestamp, (flir_contours, ell_flir, flir_frame, rect_flir)

# -------------------------------------------------------------------
# Utility Functions for Contour and Rectangle Operations
# -------------------------------------------------------------------
def create_channel_display(channel, channel_index):
    """
    Build a 3-channel image where only one channel is populated,
    using a preallocated zeros_template as the base.
    
    Parameters:
        channel (ndarray): Single-channel image data.
        channel_index (int): Which channel to place data into (0=B,1=G,2=R).
    Returns:
        ndarray: BGR image with only the specified channel data.
    """
    channels = [zeros_template.copy() for _ in range(3)]
    channels[channel_index] = channel
    return cv2.merge(channels)

def rectangle2(ellipse, max_x, max_y):
    """
    Given an ellipse (as returned by cv2.fitEllipse), compute the 
    circumscribing axis-aligned rectangle, clipped to image bounds.
    
    Parameters:
        ellipse: ((x_center,y_center),(major_axis,minor_axis),angle_degrees)
        max_x, max_y: image width/height limits.
    Returns:
        ((x1,y1),(x2,y2)): top-left and bottom-right corners.
    """
    x, y    = ellipse[0]
    mn, mj  = ellipse[1]
    ang     = ellipse[2] * np.pi / 180
    # Half-axes projections
    smnx = (mn/2) * np.cos(ang)
    smny = (mn/2) * np.sin(ang)
    smjx = (mj/2) * np.sin(ang)
    smjy = (mj/2) * np.cos(ang)
    dx   = abs(smnx) + abs(smjx)
    dy   = abs(smny) + abs(smjy)
    # Clip to [0, max-1] and convert to uint32
    x1 = np.clip(x - dx, 0, max_x-1).astype('uint32')
    y1 = np.clip(y - dy, 0, max_y-1).astype('uint32')
    x2 = np.clip(x + dx, 0, max_x-1).astype('uint32')
    y2 = np.clip(y + dy, 0, max_y-1).astype('uint32')
    return ((x1, y1), (x2, y2))

def rectangle(contour):
    """
    Compute the axis-aligned bounding box of a contour.
    
    Parameters:
        contour: Nx1x2 array of points.
    Returns:
        ((x1,y1),(x2,y2)): top-left and bottom-right corners.
    """
    x = contour[:,0,0]
    y = contour[:,0,1]
    x1, y1 = np.min(x).astype('uint32'), np.min(y).astype('uint32')
    x2, y2 = np.max(x).astype('uint32'), np.max(y).astype('uint32')
    return ((x1, y1), (x2, y2))

def unite_rectangles(rect):
    """
    Merge overlapping or near-overlapping rectangles into larger ones.
    
    Parameters:
        rect: list of ((x1,y1),(x2,y2)) rectangles.
    Returns:
        List of merged rectangles.
    """
    uni_rect = []
    remove = []
    bias = 0.05  # Fractional overlap tolerance
    for i in range(len(rect)):
        if i in remove:
            continue
        x01, y01 = rect[i][0]
        x02, y02 = rect[i][1]
        new_rect = ((x01, y01), (x02, y02))
        for j in range(i+1, len(rect)):
            if j in remove:
                continue
            x1, y1 = rect[j][0]
            x2, y2 = rect[j][1]
            dx, dy = abs(x2-x1), abs(y2-y1)
            check = False
            # Check if any corner of rect[i] lies within rect[j] ± bias
            if ((x1 - bias*dx) <= x01 <= (x2 + bias*dx) and
                (y1 - bias*dy) <= y01 <= (y2 + bias*dy)) or \
               ((x1 - bias*dx) <= x02 <= (x2 + bias*dx) and
                (y1 - bias*dy) <= y02 <= (y2 + bias*dy)):
                check = True
            if check:
                # Expand new_rect to include rect[j]
                x01 = min(x01, x1); y01 = min(y01, y1)
                x02 = max(x02, x2); y02 = max(y02, y2)
                new_rect = ((x01, y01), (x02, y02))
                remove.append(j)
        uni_rect.append(new_rect)
    return uni_rect

def normalize_rect(rec, cal, shp, amp=1, nor=True):
    """
    Normalize or denormalize rectangles between sensor coordinate frames
    using calibration parameters.
    
    Parameters:
        rec: list of ((x1,y1),(x2,y2)) rectangles.
        cal: [scale, dx, dy, rotation] calibration constants.
        shp: (height, width) of target frame.
        amp: amplification factor for low-res sources (e.g. FLIR).
        nor: True to normalize, False to denormalize.
    Returns:
        List of transformed rectangles.
    """
    if nor:
        scl = cal[0]; dx = cal[1]; dy = cal[2]; rot = cal[3]; pscl = amp
    else:
        scl = 1/cal[0]; dx = -cal[1]; dy = -cal[2]; rot = -cal[3]; pscl = 1/amp

    n_rec = []
    for (x1,y1), (x2,y2) in rec:
        # Compute center coordinates and half-sides
        cx, cy = pscl*(x1+x2)/2, pscl*(y1+y2)/2
        hx, hy = pscl*abs(x2-x1)/2, pscl*abs(y2-y1)/2
        # Rotate and translate center
        ncx = scl*((cx*np.cos(rot)-cy*np.sin(rot))-dx)
        ncy = scl*((cx*np.sin(rot)+cy*np.cos(rot))-dy)
        # Compute new corners
        nx1, ny1 = ncx-hx, ncy-hy
        nx2, ny2 = ncx+hx, ncy+hy
        if not nor:
            # Clip to bounds and round to ints
            nx1 = int(round(np.clip(nx1, 0, shp[1])))
            ny1 = int(round(np.clip(ny1, 0, shp[0])))
            nx2 = int(round(np.clip(nx2, 0, shp[1])))
            ny2 = int(round(np.clip(ny2, 0, shp[0])))
        if amp != 1:
            # Small offset for low-res sensors
            nx1 += 2; ny1 += 2; nx2 += 2; ny2 += 2
        n_rec.append(((nx1, ny1), (nx2, ny2)))
    return n_rec

def shared_area(rect1, rect2):
    """
    Compute the overlap ratio between two normalized rectangles,
    measured relative to the smaller rectangle’s area.
    
    Returns:
        float: overlap ratio [0,1].
    """
    x11, y11 = rect1[0]; x12, y12 = rect1[1]
    x21, y21 = rect2[0]; x22, y22 = rect2[1]
    area1 = abs(x12-x11)*abs(y12-y11)
    area2 = abs(x22-x21)*abs(y22-y21)
    dx = min(x12, x22) - max(x11, x21)
    dy = min(y12, y22) - max(y11, y21)
    if dx >= 0 and dy >= 0:
        overlap = dx*dy
        return min(overlap/area1, overlap/area2)
    return 0

def merge_rec(rect1, rect2):
    """
    Return the smallest axis-aligned rectangle that contains both inputs.
    """
    x11, y11 = rect1[0]; x12, y12 = rect1[1]
    x21, y21 = rect2[0]; x22, y22 = rect2[1]
    return (
        (min(x11, x21), min(y11, y21)),
        (max(x12, x22), max(y12, y22))
    )

def reduce_near_contours(contours, img):
    """
    Fit ellipses to contours with ≥5 points, merge those whose centers
    are within 1/10 of image width, recompute bounding boxes, and return:
      (merged_contours, ellipses, united_rectangles)
    """
    # Keep only contours large enough for ellipse fitting
    valid = [c for c in contours if len(c) >= 5]
    if not valid:
        return [], [], []

    ell = [cv2.fitEllipse(c) for c in valid]
    dist_border = img.shape[1] / 10  # merge threshold
    merged_contours = []
    removed = []

    # Merge by proximity
    for i in range(len(valid)):
        if i in removed:
            continue
        cnt = valid[i]
        for j in range(i+1, len(valid)):
            if j in removed:
                continue
            xi, yi = ell[i][0]
            xj, yj = ell[j][0]
            if np.hypot(xj-xi, yj-yi) < dist_border:
                cnt = np.vstack((cnt, valid[j]))
                ell[i] = cv2.fitEllipse(cnt)
                removed.append(j)
        merged_contours.append(cnt)

    # Compute final ellipses and bounding rectangles
    ell2 = []
    rects = []
    for c in merged_contours:
        el = cv2.fitEllipse(c)
        ell2.append(el)
        rects.append(rectangle(c))

    # Unite overlapping bounding rectangles
    uni_rects = unite_rectangles(rects)
    return merged_contours, ell2, uni_rects

def near_point(ellip, ellip_ref):
    """
    For each ellipse in `ellip`, find nearest ellipse center in `ellip_ref`.
    Returns an array of [x_ref, x_i, y_ref, y_i] pairs.
    """
    pairs = []
    for e in ellip:
        best = None
        min_d = float('inf')
        for r in ellip_ref:
            d = np.hypot(e[0][0]-r[0][0], e[0][1]-r[0][1])
            if d < min_d:
                min_d = d
                best = [r[0][0], e[0][0], r[0][1], e[0][1]]
        pairs.append(best or [0,0,0,0])
    return np.array(pairs)

def cluster_and_merge_rectangles(rectangles, overlap_threshold=0):
    """
    Build a graph of rectangles with shared_area > threshold,
    find connected components via DFS, and merge each cluster.
    
    Returns:
        List of merged rectangles.
    """
    n = len(rectangles)
    if n == 0:
        return []
    # Build adjacency list
    graph = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if shared_area(rectangles[i], rectangles[j]) > overlap_threshold:
                graph[i].add(j)
                graph[j].add(i)
    # DFS for clusters
    visited = set()
    clusters = []
    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        cluster = []
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            cluster.append(rectangles[curr])
            stack.extend(graph[curr] - visited)
        clusters.append(cluster)
    # Merge rectangles in each cluster
    merged = []
    for cl in clusters:
        m = cl[0]
        for r in cl[1:]:
            m = merge_rec(m, r)
        merged.append(m)
    return merged

def rects_to_list(rects):
    """
    Convert [((x1,y1),(x2,y2)), …] into [[ [x1,y1],[x2,y2] ], …]
    for JSON serialization.
    """
    return [[list(r[0]), list(r[1])] for r in rects]

# ============ Thread Classes for Sensor Acquisition ============
class RGBThread(threading.Thread):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
    def run(self):
        """
        Continuously grab RGB frames in a separate thread,
        storing the latest result in sensor_data['rgb'].
        """
        global sensor_data, running
        while running:
            result = get_rgb(self.cap)
            if result is not None:
                with data_lock:
                    sensor_data['rgb'] = result
            time.sleep(0.01)  # Prevent CPU overuse

class LiDARThread(threading.Thread):
    def __init__(self, lidar_params):
        super().__init__()
        self.lidar_params = lidar_params
    def run(self):
        """
        Continuously grab LiDAR frames in a separate thread,
        storing the latest result in sensor_data['lidar'].
        """
        global sensor_data, running
        while running:
            result = get_lidar(self.lidar_params)
            if result is not None:
                with data_lock:
                    sensor_data['lidar'] = result
            time.sleep(0.01)

class FLIRThread(threading.Thread):
    def __init__(self, flir_params):
        super().__init__()
        self.flir_params = flir_params
    def run(self):
        """
        Continuously grab FLIR frames in a separate thread,
        storing the latest result in sensor_data['flir'].
        """
        global sensor_data, running
        while running:
            result = get_flir(self.flir_params)
            if result is not None:
                with data_lock:
                    sensor_data['flir'] = result
            time.sleep(0.01)

# -------------------------------------------------------------------
# Main Workflow: Real-Time Fusion, Display, and Saving
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Load calibration JSON files from current working directory
    path = os.getcwd()
    # Read depth-depth calibration
    with open(os.path.join(path, 'dd.json'), 'r') as file:
        json_dd = json.load(file)
    # Read green-depth calibration
    with open(os.path.join(path, 'gd.json'), 'r') as file:
        json_gd = json.load(file)
    # Read red-depth calibration
    with open(os.path.join(path, 'rd.json'), 'r') as file:
        json_rd = json.load(file)
    # Read blue-depth calibration
    with open(os.path.join(path, 'bd.json'), 'r') as file:
        json_bd = json.load(file)
    # Read flir-depth calibration
    with open(os.path.join(path, 'fd.json'), 'r') as file:
        json_fd = json.load(file)

    # Build calibration arrays [scale, dx, dy, rotation]
    cal_dd = [json_dd['scale'], json_dd['dx'], json_dd['dy'], json_dd['rotation']]
    cal_gb = [json_gd['scale'], json_gd['dx'], json_gd['dy'], json_gd['rotation']]
    cal_rb = [json_rd['scale'], json_rd['dx'], json_rd['dy'], json_rd['rotation']]
    cal_db = [json_bd['scale'], json_bd['dx'], json_bd['dy'], json_bd['rotation']]
    cal_fb = [json_fd['scale'], json_fd['dx'], json_fd['dy'], json_fd['rotation']]

    # Initialize sensors once
    cap = init_rgb()                 # For RGB sensor
    ret, init_frame = cap.read()     # Grab initial frame
    if not ret:
        print("Error: Couldn't read an initial frame.")
        exit()
    # Create a single-channel zero template matching frame size
    zeros_template = np.zeros_like(init_frame[:, :, 0])

    lidar_params = init_lidar()      # For LiDAR sensor
    dpt_norm_buffer = np.empty((480, 640), dtype=np.uint8)  # Buffer for normalized depth

    flir_params = init_flir()        # For FLIR sensor
    flir_norm_buffer = np.empty(
        (flir_params["FRAME_HEIGHT"], flir_params["FRAME_WIDTH"]),
        dtype=np.uint8
    )

    # Assume shapes for normalizing/denormalizing
    rgb_shp = (480, 640)
    dpt_shp = (480, 640)
    flr_shp = (60, 80)

    # Start sensor threads
    rgb_thread   = RGBThread(cap);    rgb_thread.start()
    lidar_thread = LiDARThread(lidar_params); lidar_thread.start()
    flir_thread  = FLIRThread(flir_params);    flir_thread.start()

    # Timing thresholds
    ACCEPTABLE_THRESHOLD = 0.07  # 70 ms max skew between sensors
    FRAME_TIMEOUT        = 0.15  # 150 ms max frame age
    test = []  # To record JSON output per fused frame

    # Prepare video writer for fused mosaic output
    fourcc    = cv2.VideoWriter_fourcc(*'XVID')  # Codec: XVID
    fps       = 20.0                             # Output fps
    width, height = 600, 600                     # Output frame size
    out_mosaic = cv2.VideoWriter('fusion_output.avi', fourcc, fps, (width, height))

    # Main acquisition & fusion loop
    while True:
        # Safely fetch latest sensor data
        with data_lock:
            rgb_data   = sensor_data['rgb']
            lidar_data = sensor_data['lidar']
            flir_data  = sensor_data['flir']

        # Wait until all sensors have produced at least one frame
        if rgb_data is None or lidar_data is None or flir_data is None:
            time.sleep(0.01)
            continue

        # Unpack data and timestamps
        rgb_timestamp, rgb_result = rgb_data
        lidar_timestamp, lidar_result = lidar_data
        flir_timestamp, flir_result   = flir_data

        # Unpack RGB channel data
        (flag_r, flag_g, flag_b), channel_data = rgb_result
        (red_data, green_data, blue_data)      = channel_data
        (ell_r, r_contours, red_display, rect_r)   = red_data
        (ell_g, g_contours, green_display, rect_g) = green_data
        (ell_b, b_contours, blue_display, rect_b)  = blue_data

        # Check time synchronization between frames
        delta_lidar = abs(flir_timestamp - lidar_timestamp)
        delta_rgb   = abs(flir_timestamp - rgb_timestamp)
        print(f"Flir timestamp: {flir_timestamp:.3f} | "
              f"LiDAR diff: {delta_lidar:.3f} s | "
              f"RGB diff: {delta_rgb:.3f} s")
        if delta_lidar > ACCEPTABLE_THRESHOLD and delta_rgb > ACCEPTABLE_THRESHOLD:
            print("Frames not synchronized well enough. Waiting for a better sync...")
            time.sleep(0.005)
            continue

        # Unpack LiDAR and FLIR results
        dpt_contours, ell_dpt, depth_frame, rect_dpt = lidar_result
        if flir_result is None:
            print("Failed to acquire FLIR frame.")
            continue
        flir_contours, ell_flir, flir_frame, rect_flir = flir_result

        # Check for stale frames older than FRAME_TIMEOUT
        now = time.time()
        age_rgb   = now - rgb_timestamp
        age_lidar = now - lidar_timestamp
        age_flir  = now - flir_timestamp

        use_rgb   = (age_rgb   <= FRAME_TIMEOUT)
        use_lidar = (age_lidar <= FRAME_TIMEOUT)
        use_flir  = (age_flir  <= FRAME_TIMEOUT)

        if not use_rgb:
            rect_r = rect_g = rect_b = []
            print(f"Ignoring stale RGB   frame ({age_rgb:.3f}s old)")
        if not use_lidar:
            rect_dpt = []
            print(f"Ignoring stale LiDAR frame ({age_lidar:.3f}s old)")
        if not use_flir:
            rect_flir = []
            print(f"Ignoring stale FLIR  frame ({age_flir:.3f}s old)")

        # Normalize rectangles into a common reference frame
        nrect_r    = normalize_rect(rect_r,  cal_rb, rgb_shp)
        nrect_g    = normalize_rect(rect_g,  cal_gb, rgb_shp)
        nrect_b    = normalize_rect(rect_b,  cal_db, rgb_shp)
        nrect_dpt  = normalize_rect(rect_dpt,cal_dd, dpt_shp)
        nrect_flir = normalize_rect(rect_flir,cal_fb, flr_shp, amp=8)

        # Cluster and merge overlapping detections per sensor
        merged_rect_r    = cluster_and_merge_rectangles(nrect_r,    overlap_threshold=0)
        merged_rect_g    = cluster_and_merge_rectangles(nrect_g,    overlap_threshold=0)
        merged_rect_b    = cluster_and_merge_rectangles(nrect_b,    overlap_threshold=0)
        merged_rect_dpt  = cluster_and_merge_rectangles(nrect_dpt,  overlap_threshold=0)
        merged_rect_flir = cluster_and_merge_rectangles(nrect_flir, overlap_threshold=0)
        # Fuse all detections together
        merged_all       = cluster_and_merge_rectangles(
            nrect_r + nrect_g + nrect_b + nrect_dpt + nrect_flir,
            overlap_threshold=0
        )

        # Record data for JSON output
        record = {
            "timestamps": {
                "rgb":   rgb_timestamp,
                "lidar": lidar_timestamp,
                "flir":  flir_timestamp
            },
            "rectangles": {
                "rgb": {
                    "red":   rects_to_list(merged_rect_r),
                    "green": rects_to_list(merged_rect_g),
                    "blue":  rects_to_list(merged_rect_b)
                },
                "lidar": rects_to_list(merged_rect_dpt),
                "flir":  rects_to_list(merged_rect_flir)
            },
            "fused_rectangles": rects_to_list(merged_all)
        }
        test.append(record)

        # Denormalize fused rectangles back into each view for display
        dn_dpt = normalize_rect(merged_all, cal_dd, dpt_shp, nor=False)
        for box in dn_dpt:
            cv2.rectangle(depth_frame, box[0], box[1], (0, 255, 0), 2)

        dn_r = normalize_rect(merged_all, cal_rb, rgb_shp, nor=False)
        for box in dn_r:
            cv2.rectangle(red_display, box[0], box[1], (0, 255, 0), 2)

        dn_g = normalize_rect(merged_all, cal_gb, rgb_shp, nor=False)
        for box in dn_g:
            cv2.rectangle(green_display, box[0], box[1], (0, 255, 0), 2)

        dn_b = normalize_rect(merged_all, cal_db, rgb_shp, nor=False)
        for box in dn_b:
            cv2.rectangle(blue_display, box[0], box[1], (0, 255, 0), 2)

        dn_flir = normalize_rect(merged_all, cal_fb, flr_shp, amp=8, nor=False)
        for box in dn_flir:
            # Subtract small offset for FLIR view
            top_left  = np.array(box[0]) - 2
            bottom_right = np.array(box[1]) - 2
            cv2.rectangle(flir_frame, tuple(top_left), tuple(bottom_right), (0, 255, 0), 1)

        # Resize FLIR view for visibility
        flir_frame = cv2.resize(flir_frame, (80*10, 80*10), interpolation=cv2.INTER_NEAREST)

        # --- Build and display the mosaic ---
        # Resize each view to 300×300 for uniform mosaic
        thermal_resized = cv2.resize(flir_frame, (300, 300))   # Thermal
        b_resized       = cv2.resize(blue_display, (300, 300)) # Blue channel
        g_resized       = cv2.resize(green_display, (300, 300))# Green channel
        r_resized       = cv2.resize(red_display, (300, 300))  # Red channel

        # Arrange into a 2×2 grid:
        # Top row: Thermal | Blue
        # Bot row: Green   | Red
        top_row    = np.hstack((thermal_resized, b_resized))
        bottom_row = np.hstack((g_resized, r_resized))
        mosaic     = np.vstack((top_row, bottom_row))

        # Display the fusion view and LiDAR depth view, and write mosaic to video
        cv2.imshow("Fusion View (Thermal + RGB)", mosaic)
        cv2.imshow("LiDAR Depth", depth_frame)
        out_mosaic.write(mosaic)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    # After exiting the main loop: save results, release resources, and join threads
    with open("normalized_rects.json", "w") as f:
        json.dump(test, f, indent=2)
    print(f"Saved {len(test)} frames of data to normalized_rects.json")

    out_mosaic.release()         # Close video writer
    cv2.destroyAllWindows()      # Close all OpenCV windows
    rgb_thread.join()            # Ensure all threads exit cleanly
    lidar_thread.join()
    flir_thread.join()
