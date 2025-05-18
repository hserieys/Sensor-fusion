import cv2                             # OpenCV for image capture and processing
import os                              # Filesystem operations (paths, directory listing)
import numpy as np                     # Numerical operations on arrays
import json                            # JSON serialization/deserialization
import ctypes                          # Interface to C libraries (for FLIR capture)
import time                            # Time stamps and delays
import threading                       # Thread support for concurrent sensor reads
from primesense import openni2         # OpenNI2 for depth (LiDAR) sensor
from primesense import _openni2 as c_api  # OpenNI2 C API types/constants

# ============ Global Variables & Constants ============
DILATION_KERNEL      = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Kernel for dilating RGB edges
DILATION_KERNEL_FLIR = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # Kernel for dilating FLIR edges
zeros_template       = None  # Placeholder image for channel displays

# Preallocate global buffers for normalized depth & FLIR (set later in main)
dpt_norm_buffer  = None
flir_norm_buffer = None

# Shared sensor data and lock (used by sensor threads)
sensor_data = {'rgb': None, 'lidar': None, 'flir': None}
data_lock   = threading.Lock()  # Protects concurrent access to sensor_data
running     = True              # Control flag for threads

# -------------------------------------------------------------------
# Sensor Initialization Functions
# -------------------------------------------------------------------
def init_rgb():
    """Initializes the webcam and returns the capture object."""
    cap = cv2.VideoCapture(0, cv2.CAP_V4L)  # Open device 0 with V4L backend
    if not cap.isOpened():
        print("Error: Could not access the RGB camera.")
        exit()  # Exit if camera cannot be opened
    print("RGB camera opened successfully.")
    return cap

def init_lidar():
    """
    Initializes OpenNI2 for the depth stream.

    Returns:
        dict: A dictionary containing the depth_stream and device.
    """
    # Initialize OpenNI2 with the SDK plugin path
    openni2.initialize("/home/pi/Desktop/AstraSDK-v2.1.3-Linux-arm/"
                       "AstraSDK-v2.1.3-94bca0f52e-20210611T022735Z-Linux-arm/"
                       "lib/Plugins/openni2")
    dev = openni2.Device.open_any()  # Open any connected depth device
    depth_stream = dev.create_depth_stream()
    # Configure video mode: 1 mm depth resolution, 640x480, 30 FPS
    depth_stream.set_video_mode(c_api.OniVideoMode(
        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
        resolutionX=640, resolutionY=480, fps=30
    ))
    depth_stream.start()
    return {"depth_stream": depth_stream, "device": dev}

def init_flir():
    """
    Initializes the thermal camera.

    Returns:
        dict: A dictionary containing the library, buffer pointer, frame buffer,
              and frame dimensions.
    """
    # Load the FLIR capture shared library
    lib = ctypes.CDLL('/home/pi/Sensor-fusion/CalibrationStep/'
                      'LeptonModule-master/software/raspberrypi_capture/'
                      'libraspberrypi_capture.so')
    lib.main.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
    lib.main.restype  = ctypes.c_int

    FRAME_WIDTH      = 80
    FRAME_HEIGHT     = 60
    BYTES_PER_PIXEL  = 2
    FRAME_SIZE       = FRAME_WIDTH * FRAME_HEIGHT * BYTES_PER_PIXEL

    # Allocate buffer for raw frame bytes
    frame_buffer = np.zeros(FRAME_SIZE, dtype=np.uint8)
    buffer_ptr   = frame_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    return {
        "lib": lib,
        "buffer_ptr": buffer_ptr,
        "frame_buffer": frame_buffer,
        "FRAME_WIDTH": FRAME_WIDTH,
        "FRAME_HEIGHT": FRAME_HEIGHT
    }

# -------------------------------------------------------------------
# Sensor Get Functions (Using the Initialized Objects)
# -------------------------------------------------------------------
def get_rgb(cap):
    """
    Grabs a frame from the RGB camera, processes each channel separately
    for contour detection, and returns timestamp plus channel results.
    """
    ret, frame = cap.read()
    rgb_timestamp = time.time()
    if not ret:
        print("Error: Failed to grab frame.")
        cap.release()
        return

    # Split B, G, R channels
    b, g, r = cv2.split(frame)

    # New robustness check: minimal contrast threshold
    INTENSITY_THRESHOLD = 80

    # Create per-channel display images using zeros_template
    blue_display  = create_channel_display(b, 0)  # Blue → B plane
    green_display = create_channel_display(g, 1)  # Green → G plane
    red_display   = create_channel_display(r, 2)  # Red → R plane
    kernel        = DILATION_KERNEL

    # --- Process Blue channel ---
    b_blur = cv2.GaussianBlur(b, (5, 5), 0)
    diff_b  = np.max(b_blur) - np.min(b_blur)
    flag_b  = (diff_b >= INTENSITY_THRESHOLD)
    if flag_b:
        _, edges_blue    = cv2.threshold(b_blur, 0, 255, cv2.THRESH_OTSU)
        edges_blue       = cv2.dilate(edges_blue, kernel, iterations=2)
        b_contours       = []
        contours_blue, _ = cv2.findContours(edges_blue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours_blue):
            area = cv2.contourArea(c)
            if area < 1e2 or area > 5e4:
                continue
            b_contours.append(c)
        b_contours, ell_b, rect_b = reduce_near_contours(b_contours, b)
    else:
        b_contours, ell_b, rect_b = [], [], []

    # --- Process Green channel ---
    g_blur = cv2.GaussianBlur(g, (5, 5), 0)
    diff_g  = np.max(g_blur) - np.min(g_blur)
    flag_g  = (diff_g >= INTENSITY_THRESHOLD)
    if flag_g:
        _, edges_green    = cv2.threshold(g_blur, 0, 255, cv2.THRESH_OTSU)
        edges_green       = cv2.dilate(edges_green, kernel, iterations=2)
        g_contours        = []
        contours_green, _ = cv2.findContours(edges_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours_green):
            area = cv2.contourArea(c)
            if area < 1e2 or area > 5e4:
                continue
            g_contours.append(c)
        g_contours, ell_g, rect_g = reduce_near_contours(g_contours, g)
    else:
        g_contours, ell_g, rect_g = [], [], []

    # --- Process Red channel ---
    r_blur = cv2.GaussianBlur(r, (5, 5), 0)
    diff_r  = np.max(r_blur) - np.min(r_blur)
    flag_r  = (diff_r >= INTENSITY_THRESHOLD)
    if flag_r:
        _, edges_red    = cv2.threshold(r_blur, 0, 255, cv2.THRESH_OTSU)
        edges_red       = cv2.dilate(edges_red, kernel, iterations=2)
        r_contours      = []
        contours_red, _ = cv2.findContours(edges_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours_red):
            area = cv2.contourArea(c)
            if area < 1e2 or area > 5e4:
                continue
            r_contours.append(c)
        r_contours, ell_r, rect_r = reduce_near_contours(r_contours, red_display)
    else:
        r_contours, ell_r, rect_r = [], [], []

    # Structure and return results
    channel_data = (
        (ell_r, r_contours, red_display, rect_r),
        (ell_g, g_contours, green_display, rect_g),
        (ell_b, b_contours, blue_display, rect_b)
    )
    return (rgb_timestamp, ((flag_r, flag_g, flag_b), channel_data))

def get_lidar(lidar_params):
    """
    Grabs a frame from the LiDAR depth sensor, normalizes it,
    detects contours, merges them, and returns timestamp + results.
    """
    depth_stream = lidar_params["depth_stream"]
    frame        = depth_stream.read_frame()
    lidar_timestamp = time.time()

    # Convert raw buffer to 16-bit image and flip/normalize
    frame_data    = frame.get_buffer_as_uint16()
    depth_img_16  = np.frombuffer(frame_data, dtype=np.uint16).reshape((480, 640))
    depth_img_16[depth_img_16 == 0] = np.max(depth_img_16)
    depth_img_16  = np.fliplr(depth_img_16)
    cv2.normalize(depth_img_16, dpt_norm_buffer, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    display_frame = dpt_norm_buffer

    # Threshold + find contours
    ret, thresh     = cv2.threshold(display_frame, 0, 255, cv2.THRESH_OTSU)
    contours, _     = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    depth_frame     = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

    dpt_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1e2 or area > 1e4:
            continue
        dpt_contours.append(c)
    dpt_contours, ell_dpt, rect_dpt = reduce_near_contours(dpt_contours, depth_frame)

    return (lidar_timestamp, (dpt_contours, ell_dpt, depth_frame, rect_dpt))

def get_flir(flir_params):
    """
    Grabs a frame from the FLIR thermal camera via C library call,
    denoises, thresholds, detects & merges contours, returns timestamp + results.
    """
    lib           = flir_params["lib"]
    buffer_ptr    = flir_params["buffer_ptr"]
    frame_buffer  = flir_params["frame_buffer"]
    FRAME_WIDTH   = flir_params["FRAME_WIDTH"]
    FRAME_HEIGHT  = flir_params["FRAME_HEIGHT"]

    result        = lib.main(buffer_ptr)  # Capture via C function
    flir_timestamp = time.time()
    if result != 0:
        print("Error capturing frame:", result)
        return None

    # Interpret as 16-bit → normalize → denoise
    frame_data = frame_buffer.view(dtype=np.uint16).reshape((FRAME_HEIGHT, FRAME_WIDTH))
    flr_as_pic = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    flr_as_pic = cv2.fastNlMeansDenoising(flr_as_pic, None, 90, 3, 9)

    # Histogram stretch
    histr_flr = cv2.calcHist([flr_as_pic],[0],None,[256],[0,256]).flatten()
    min_flr, max_flr = 0, 255
    while histr_flr[min_flr] < 5 and min_flr < 255:
        min_flr += 1
    while histr_flr[max_flr] < 5 and max_flr > 0:
        max_flr -= 1
    flr_m = 255/(max_flr-min_flr); flr_n = -flr_m*min_flr
    flr_ir = ((np.clip(flr_as_pic,min_flr,max_flr)*flr_m)+flr_n).astype('uint8')

    denoised_frame = cv2.rotate(flr_ir, cv2.ROTATE_180)
    ret, thresh    = cv2.threshold(denoised_frame, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    canny          = cv2.Canny(denoised_frame,75,200)
    canny          = cv2.dilate(canny, DILATION_KERNEL_FLIR, iterations=1)
    contours, _    = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    flir_frame    = cv2.cvtColor(denoised_frame, cv2.COLOR_GRAY2BGR)
    flir_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1e1 or area > 1e3:
            continue
        flir_contours.append(c)

    flir_contours, ell_flir, rect_flir = reduce_near_contours(flir_contours, flir_frame)
    return (flir_timestamp, (flir_contours, ell_flir, flir_frame, rect_flir))

# -------------------------------------------------------------------
# Utility Functions for Detection (reduce_near_contours, near_point, etc.)
# -------------------------------------------------------------------
def create_channel_display(channel, channel_index):
    """
    Creates a display image for one channel by merging it into a BGR
    image at the specified index, using zeros_template for other channels.
    """
    channels = [zeros_template.copy() for _ in range(3)]
    channels[channel_index] = channel
    return cv2.merge(channels)

def rectangle2(ellipse, max_x, max_y):
    """
    Generates an axis-aligned bounding box around an ellipse,
    respecting image boundaries max_x, max_y.
    """
    x, y    = ellipse[0]          # ellipse center
    mn, mj  = ellipse[1]          # ellipse axes lengths
    ang     = ellipse[2] * np.pi/180
    smnx = mn/2*np.cos(ang); smny = mn/2*np.sin(ang)
    smjx = mj/2*np.sin(ang); smjy = mj/2*np.cos(ang)
    dx = np.abs(smnx) + np.abs(smjx)
    dy = np.abs(smny) + np.abs(smjy)
    x1 = int(np.clip(x-dx,0, max_x-1)); y1 = int(np.clip(y-dy,0, max_y-1))
    x2 = int(np.clip(x+dx,0, max_x-1)); y2 = int(np.clip(y+dy,0, max_y-1))
    return ((x1,y1),(x2,y2))

def rectangle(contour):
    """
    Axis-aligned bounding box for a contour array of shape (N,1,2).
    """
    x = contour[:,0,0]; y = contour[:,0,1]
    return ((int(x.min()), int(y.min())), (int(x.max()), int(y.max())))

def unite_rectangles(rect):
    """
    Merge overlapping axis-aligned rectangles from a list into unified boxes.
    """
    uni_rect = []
    remove   = []
    bias     = 0.05
    for i in range(len(rect)):
        if i in remove: continue
        rect0 = rect[i]
        x01,y01 = rect0[0]; x02,y02 = rect0[1]
        new_rect = ((x01,y01),(x02,y02))
        for j in range(len(rect)):
            if j < i: continue
            x1,y1 = rect[j][0]; x2,y2 = rect[j][1]
            dx, dy = abs(x2-x1), abs(y2-y1)
            check = ((x1-bias*dx)<=x01<=x2+bias*dx and (y1-bias*dy)<=y01<=y2+bias*dy) or \
                    ((x1-bias*dx)<=x01<=x2+bias*dx and (y1-bias*dy)<=y02<=y2+bias*dy) or \
                    ((x1-bias*dx)<=x02<=x2+bias*dx and (y1-bias*dy)<=y01<=y2+bias*dy) or \
                    ((x1-bias*dx)<=x02<=x2+bias*dx and (y1-bias*dy)<=y02<=y2+bias*dy)
            if check and i!=j:
                x01,y01 = min(x01,x1), min(y01,y1)
                x02,y02 = max(x02,x2), max(y02,y2)
                new_rect = ((x01,y01),(x02,y02))
                remove.append(j)
        uni_rect.append(new_rect)
    return uni_rect

def normalize_rect(rec, cal, shp, amp=1, nor=True):
    """
    Normalize or denormalize rectangles given calibration [scale, dx, dy, rot].
      - nor=True: map from sensor to reference frame
      - nor=False: inverse mapping
    """
    if nor:
        scl, dx, dy, rot = cal
        pscl = amp
    else:
        scl, dx, dy, rot = 1/cal[0], -cal[1], -cal[2], -cal[3]
        pscl = 1/amp
    n_rec = []
    for i in rec:
        x1,y1 = i[0]; x2,y2 = i[1]
        center_x = pscl*(x1+x2)/2; center_y = pscl*(y1+y2)/2
        hside_x  = pscl*abs(x2-x1)/2; hside_y = pscl*abs(y2-y1)/2
        if nor:
            ncenter_x = scl*((center_x*np.cos(rot)-center_y*np.sin(rot))-dx)
            ncenter_y = scl*((center_x*np.sin(rot)+center_y*np.cos(rot))-dy)
        else:
            ncenter_x = scl*((center_x*np.cos(rot)-center_y*np.sin(rot))-pscl*dx)
            ncenter_y = scl*((center_x*np.sin(rot)+center_y*np.cos(rot))-pscl*dy)
        nhside_x = scl*hside_x; nhside_y = scl*hside_y
        nx1 = ncenter_x - nhside_x; ny1 = ncenter_y - nhside_y
        nx2 = ncenter_x + nhside_x; ny2 = ncenter_y + nhside_y
        if not nor:
            nx1 = int(round(np.clip(nx1,0,shp[1]))); ny1 = int(round(np.clip(ny1,0,shp[0])))
            nx2 = int(round(np.clip(nx2,0,shp[1]))); ny2 = int(round(np.clip(ny2,0,shp[0])))
        if amp != 1:
            nx1+=2; ny1+=2; nx2+=2; ny2+=2
        n_rec.append(((nx1,ny1),(nx2,ny2)))
    return n_rec

def shared_area(rect1,rect2):
    """
    Calculates the normalized intersection area between two rectangles.
    Returns min(inter_area/area1, inter_area/area2).
    """
    x11,y11 = rect1[0]; x12,y12 = rect1[1]
    x21,y21 = rect2[0]; x22,y22 = rect2[1]
    area1 = abs(x12-x11)*abs(y12-y11); area2 = abs(x22-x21)*abs(y22-y21)
    dx = min(x12,x22) - max(x11,x21); dy = min(y12,y22) - max(y11,y21)
    if dx>=0 and dy>=0:
        area = dx*dy
        return min(area/area1, area/area2)
    else:
        return 0

def merge_rec(rect1,rect2):
    """
    Merges two axis-aligned rectangles into one that covers both.
    """
    x11,y11 = rect1[0]; x12,y12 = rect1[1]
    x21,y21 = rect2[0]; x22,y22 = rect2[1]
    minx = min(x11,x21); miny = min(y11,y21)
    maxx = max(x12,x22); maxy = max(y12,y22)
    return ((minx,miny),(maxx,maxy))

def reduce_near_contours(contours, img):
    """
    Filters, fits ellipses, merges nearby contours, and returns merged results.
    """
    valid_contours = [c for c in contours if len(c) >= 5]
    if not valid_contours:
        return [], [], []
    ell = [cv2.fitEllipse(c) for c in valid_contours]
    dist_border = img.shape[1] / 10
    remove = []; contour2 = []
    for i in range(len(valid_contours)):
        if i in remove: continue
        contour = valid_contours[i]
        for j in range(i+1, len(valid_contours)):
            if j in remove: continue
            d = np.hypot(ell[j][0][0]-ell[i][0][0], ell[j][0][1]-ell[i][0][1])
            if d < dist_border:
                contour = np.concatenate((contour, valid_contours[j]))
                ell[i] = cv2.fitEllipse(contour)
                valid_contours[i] = contour
                remove.append(j)
        contour2.append(contour)
    ell2 = []; rect0 = []
    for c in contour2:
        ellipse = cv2.fitEllipse(c)
        rectang = rectangle(c)
        ell2.append(ellipse); rect0.append(rectang)
    rect = unite_rectangles(rect0)
    return contour2, ell2, rect

def near_point(ellip, ellip_ref):
    """
    For each ellipse in ellip, finds nearest ellipse center in ellip_ref.
    Returns array of [x1,x2,y1,y2].
    """
    near_pt = []
    for i in ellip:
        min_d = float('inf'); best_point = None
        for j in ellip_ref:
            x1,y1 = j[0]; x2,y2 = i[0]
            d = np.hypot(x2-x1, y2-y1)
            if d < min_d:
                min_d = d; best_point = [x1, x2, y1, y2]
        if best_point is None:
            best_point = [0,0,0,0]
        near_pt.append(best_point)
    return np.array(near_pt)

def cluster_and_merge_rectangles(rectangles, overlap_threshold=0):
    """
    Groups rectangles by overlap into clusters, then merges each cluster.
    """
    n = len(rectangles)
    if n == 0:
        return []
    graph = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if shared_area(rectangles[i], rectangles[j]) > overlap_threshold:
                graph[i].add(j); graph[j].add(i)
    visited = set(); clusters = []
    for i in range(n):
        if i not in visited:
            stack = [i]; cluster = []
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    cluster.append(rectangles[curr])
                    stack.extend(graph[curr] - visited)
            clusters.append(cluster)
    merged_rectangles = []
    for cluster in clusters:
        merged = cluster[0]
        for rect in cluster[1:]:
            merged = merge_rec(merged, rect)
        merged_rectangles.append(merged)
    return merged_rectangles

def to_int_rect(rect):
    """
    Converts float rectangle coords to integer tuples.
    """
    (x1,y1),(x2,y2) = rect
    return (int(x1), int(y1)), (int(x2), int(y2))

# ============ Thread Classes for Sensor Acquisition ============
class RGBThread(threading.Thread):
    def __init__(self, cap):
        super(RGBThread, self).__init__()
        self.cap = cap

    def run(self):
        global sensor_data, running
        while running:
            result = get_rgb(self.cap)
            if result is not None:
                with data_lock:
                    sensor_data['rgb'] = result
            time.sleep(0.01)  # slight delay to prevent high CPU usage

class LiDARThread(threading.Thread):
    def __init__(self, lidar_params):
        super(LiDARThread, self).__init__()
        self.lidar_params = lidar_params

    def run(self):
        global sensor_data, running
        while running:
            result = get_lidar(self.lidar_params)
            if result is not None:
                with data_lock:
                    sensor_data['lidar'] = result
            time.sleep(0.01)

class FLIRThread(threading.Thread):
    def __init__(self, flir_params):
        super(FLIRThread, self).__init__()
        self.flir_params = flir_params

    def run(self):
        global sensor_data, running
        while running:
            result = get_flir(self.flir_params)
            if result is not None:
                with data_lock:
                    sensor_data['flir'] = result
            time.sleep(0.01)

# -------------------------------------------------------------------
# Main Workflow: Real-Time Detection, Stats Accumulation, and Saving
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Get working directory and list of files (for calibration JSONs)
    path = os.getcwd()

    # Load calibration JSON files
    with open(path + '/dd.json', 'r') as f: json_dd = json.load(f)
    with open(path + '/gd.json', 'r') as f: json_gd = json.load(f)
    with open(path + '/rd.json', 'r') as f: json_rd = json.load(f)
    with open(path + '/bd.json', 'r') as f: json_bd = json.load(f)
    with open(path + '/fd.json', 'r') as f: json_fd = json.load(f)

    # Build calibration arrays [scale, dx, dy, rotation]
    cal_dd = [json_dd['scale'], json_dd['dx'], json_dd['dy'], json_dd['rotation']]
    cal_gb = [json_gd['scale'], json_gd['dx'], json_gd['dy'], json_gd['rotation']]
    cal_rb = [json_rd['scale'], json_rd['dx'], json_rd['dy'], json_rd['rotation']]
    cal_db = [json_bd['scale'], json_bd['dx'], json_bd['dy'], json_bd['rotation']]
    cal_fb = [json_fd['scale'], json_fd['dx'], json_fd['dy'], json_fd['rotation']]

    # Initialize sensors once
    cap = init_rgb()                 # Start RGB camera
    ret, init_frame = cap.read()
    if not ret:
        print("Error: Couldn't read an initial frame.")
        exit()
    zeros_template = np.zeros_like(init_frame[:, :, 0])  # For channel displays

    lidar_params = init_lidar()      # Start LiDAR sensor
    dpt_norm_buffer = np.empty((480, 640), dtype=np.uint8)  # Buffer for depth normalization

    flir_params = init_flir()        # Start FLIR sensor
    flir_norm_buffer = np.empty((flir_params["FRAME_HEIGHT"], flir_params["FRAME_WIDTH"]), dtype=np.uint8)

    # Define image shapes
    rgb_shp = (480, 640)
    dpt_shp = (480, 640)
    flr_shp = (60, 80)

    # Create and start sensor threads
    rgb_thread   = RGBThread(cap)
    lidar_thread = LiDARThread(lidar_params)
    flir_thread  = FLIRThread(flir_params)
    rgb_thread.start()
    lidar_thread.start()
    flir_thread.start()

    # Timing thresholds for synchronization and staleness
    ACCEPTABLE_THRESHOLD = 0.07
    FRAME_TIMEOUT        = 0.15

    frame_stats = []  # Accumulate per-frame statistics

    while True:
        # Safely grab latest sensor data
        with data_lock:
            rgb_data   = sensor_data['rgb']
            lidar_data = sensor_data['lidar']
            flir_data  = sensor_data['flir']

        # Wait until all sensors have provided a frame
        if rgb_data is None or lidar_data is None or flir_data is None:
            time.sleep(0.01)
            continue

        # Unpack RGB data
        rgb_timestamp, rgb_result = rgb_data
        (flag_r, flag_g, flag_b), channel_data = rgb_result
        red_data, green_data, blue_data = channel_data
        ell_r, r_contours, red_display, rect_r   = red_data
        ell_g, g_contours, green_display, rect_g = green_data
        ell_b, b_contours, blue_display, rect_b  = blue_data

        # Unpack LiDAR and FLIR data
        lidar_timestamp, lidar_result = lidar_data
        flir_timestamp, flir_result   = flir_data

        # Synchronization checks
        delta_lidar = abs(flir_timestamp - lidar_timestamp)
        delta_rgb   = abs(flir_timestamp - rgb_timestamp)
        print(f"Flir timestamp: {flir_timestamp:.3f} | LiDAR diff: {delta_lidar:.3f} s | Rgb diff: {delta_rgb:.3f} s")
        if delta_lidar > ACCEPTABLE_THRESHOLD and delta_rgb > ACCEPTABLE_THRESHOLD:
            print("Frames not synchronized well enough. Waiting for a better sync...")
            time.sleep(0.005)
            continue

        # Unpack processed LiDAR and FLIR
        dpt_contours, ell_dpt, depth_frame, rect_dpt = lidar_result
        if flir_result is None:
            print("Failed to acquire FLIR frame.")
            continue
        flir_contours, ell_flir, flir_frame, rect_flir = flir_result

        # Age-based staleness checks
        now        = time.time()
        age_rgb    = now - rgb_timestamp
        age_lidar  = now - lidar_timestamp
        age_flir   = now - flir_timestamp
        use_rgb    = (age_rgb   <= FRAME_TIMEOUT)
        use_lidar  = (age_lidar <= FRAME_TIMEOUT)
        use_flir   = (age_flir  <= FRAME_TIMEOUT)

        # Normalize rectangles into common reference frame
        nrect_r    = normalize_rect(rect_r,    cal_rb, rgb_shp)
        nrect_g    = normalize_rect(rect_g,    cal_gb, rgb_shp)
        nrect_b    = normalize_rect(rect_b,    cal_db, rgb_shp)
        nrect_flir = normalize_rect(rect_flir, cal_fb, flr_shp, amp=8)
        nrect_dpt  = normalize_rect(rect_dpt,  cal_dd, dpt_shp)

        if not use_rgb:
            nrect_r, nrect_g, nrect_b = [], [], []
            print(f"Ignoring stale RGB   frame ({age_rgb:.3f}s old)")
        if not use_lidar:
            nrect_dpt = []
            print(f"Ignoring stale LiDAR frame ({age_lidar:.3f}s old)")
        if not use_flir:
            nrect_flir = []
            print(f"Ignoring stale FLIR  frame ({age_flir:.3f}s old)")

        # Concatenate all normalized rects from sensors
        conca_nor = nrect_r + nrect_g + nrect_b + nrect_dpt + nrect_flir

        # Cluster & merge overlapping rects
        merged_rectangles = cluster_and_merge_rectangles(conca_nor, overlap_threshold=0)

        # Denormalize merged rectangles to each sensor's image coords
        dn_merged_dpt = normalize_rect(merged_rectangles, cal_dd, dpt_shp, nor=False)
        dnrect_r      = normalize_rect(merged_rectangles, cal_rb, rgb_shp, nor=False)
        dnrect_g      = normalize_rect(merged_rectangles, cal_gb, rgb_shp, nor=False)
        dnrect_b      = normalize_rect(merged_rectangles, cal_db, rgb_shp, nor=False)
        dnrect_flir   = normalize_rect(merged_rectangles, cal_fb, flr_shp, 8, nor=False)
        dnrect_r      = [to_int_rect(r) for r in dnrect_r]
        dnrect_g      = [to_int_rect(r) for r in dnrect_g]
        dnrect_b      = [to_int_rect(r) for r in dnrect_b]
        dn_merged_dpt = [to_int_rect(r) for r in dn_merged_dpt]
        dnrect_flir   = [to_int_rect(r) for r in dnrect_flir]

        # Prepare blank masks for each channel
        mask_full_r = np.zeros(rgb_shp, dtype=np.uint8)
        mask_full_g = np.zeros(rgb_shp, dtype=np.uint8)
        mask_full_b = np.zeros(rgb_shp, dtype=np.uint8)
        mask_full_d = np.zeros_like(dpt_norm_buffer, dtype=np.uint8)
        mask_full_f = np.zeros_like(flir_norm_buffer, dtype=np.uint8)

        # For each fused detection region, threshold & build per-channel masks & stats
        for idx in range(len(merged_rectangles)):
            rects = {
                'r': dnrect_r[idx],
                'g': dnrect_g[idx],
                'b': dnrect_b[idx],
                'd': dn_merged_dpt[idx],
                'f': dnrect_flir[idx]
            }
            buffers = {
                'r': (red_display,   True,  cv2.THRESH_BINARY   | cv2.THRESH_OTSU),
                'g': (green_display, True,  cv2.THRESH_BINARY   | cv2.THRESH_OTSU),
                'b': (blue_display,  True,  cv2.THRESH_BINARY   | cv2.THRESH_OTSU),
                'd': (dpt_norm_buffer, False, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU),
                'f': (flir_frame,    True,  cv2.THRESH_BINARY   | cv2.THRESH_OTSU)
            }

            # Build masks by thresholding each ROI
            for key, ((x1,y1),(x2,y2)) in rects.items():
                img, to_gray, thresh = buffers[key]
                full_mask = {'r': mask_full_r, 'g': mask_full_g,
                             'b': mask_full_b, 'd': mask_full_d,
                             'f': mask_full_f}[key]
                h, w = img.shape[:2]
                x1c, y1c = max(0,int(x1)), max(0,int(y1))
                x2c = min(w,int(x2)); y2c = min(h,int(y2))
                if x2c > x1c and y2c > y1c:
                    roi = img[y1c:y2c, x1c:x2c]
                    if to_gray:
                        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, small_mask = cv2.threshold(roi, 0, 255, thresh)
                    full_mask[y1c:y2c, x1c:x2c] = small_mask

            # Resize FLIR mask up to RGB resolution
            mask_full_f_resized = cv2.resize(mask_full_f, (rgb_shp[1], rgb_shp[0]), interpolation=cv2.INTER_NEAREST)

            # Count nonzero pixels per sensor
            count_r = cv2.countNonZero(mask_full_r)
            count_g = cv2.countNonZero(mask_full_g)
            count_b = cv2.countNonZero(mask_full_b)
            count_d = cv2.countNonZero(mask_full_d)
            count_f = cv2.countNonZero(mask_full_f_resized)

            # Fuse masks and compute vote-based Jaccard metrics
            stack = np.stack([mask_full_r, mask_full_g, mask_full_b, mask_full_d, mask_full_f_resized], axis=-1)
            votes = np.count_nonzero(stack, axis=-1)
            at_least_1 = (votes >= 1).astype(np.uint8) * 255
            at_least_2 = (votes >= 2).astype(np.uint8) * 255
            at_least_3 = (votes >= 3).astype(np.uint8) * 255
            at_least_4 = (votes >= 4).astype(np.uint8) * 255
            at_least_5 = (votes >= 5).astype(np.uint8) * 255

            possibility       = cv2.countNonZero(at_least_1)
            count_at_least_2  = cv2.countNonZero(at_least_2)
            count_at_least_3  = cv2.countNonZero(at_least_3)
            count_at_least_4  = cv2.countNonZero(at_least_4)
            certainty         = cv2.countNonZero(at_least_5)

            # Morphological cleanup of the chosen Jaccard mask (≥3 sensors)
            ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            fused_mask = cv2.morphologyEx(at_least_3, cv2.MORPH_CLOSE, ker, iterations=1)
            fused_mask = cv2.morphologyEx(fused_mask, cv2.MORPH_OPEN, ker, iterations=2)
            fused_mask = cv2.morphologyEx(fused_mask, cv2.MORPH_CLOSE, ker, iterations=2)

            # Compute contributions per sensor (fraction of fused pixels)
            total_fused = cv2.countNonZero(fused_mask)
            depth_contribution = count_d / total_fused if total_fused > 0 else 0.0
            r_contribution     = count_r / total_fused if total_fused > 0 else 0.0
            g_contribution     = count_g / total_fused if total_fused > 0 else 0.0
            b_contribution     = count_b / total_fused if total_fused > 0 else 0.0
            flir_contribution  = count_f / total_fused if total_fused > 0 else 0.0

            # Compute Jaccard indices
            jaccard1 = certainty / float(possibility)       if possibility      > 0 else 0.0
            jaccard2 = certainty / float(count_at_least_2)  if count_at_least_2 > 0 else 0.0
            jaccard3 = certainty / float(count_at_least_3)  if count_at_least_3 > 0 else 0.0
            jaccard4 = certainty / float(count_at_least_4)  if count_at_least_4 > 0 else 0.0
            jaccard5 = 1.0                                  if certainty        > 0 else 0.0

            # Assemble per-frame stats
            stats = {
                "pixels": {
                    "red":   count_r,
                    "green": count_g,
                    "blue":  count_b,
                    "depth": count_d,
                    "flir":  count_f
                },
                "contribution": {
                    "red":   r_contribution,
                    "green": g_contribution,
                    "blue":  b_contribution,
                    "depth": depth_contribution,
                    "flir":  flir_contribution
                },
                "jaccard": {
                    ">=1_sensor": jaccard1,
                    ">=2_sensors": jaccard2,
                    ">=3_sensors": jaccard3,
                    ">=4_sensors": jaccard4,
                    ">=5_sensors": jaccard5
                }
            }
            frame_stats.append(stats)

            # Display each mask
            cv2.imshow("Mask - Red channel",     mask_full_r)
            cv2.imshow("Mask - Green channel",   mask_full_g)
            cv2.imshow("Mask - Blue channel",    mask_full_b)
            cv2.imshow("Mask - LiDAR depth",     mask_full_d)
            cv2.imshow("Mask - FLIR thermal",    mask_full_f_resized)
            cv2.imshow("Fused Mask",             fused_mask)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    # Once the loop exits, save collected stats to JSON
    with open("sensor_contributions.json", "w") as f:
        json.dump(frame_stats, f, indent=2)

    # Cleanup threads and windows
    cv2.destroyAllWindows()
    rgb_thread.join()
    lidar_thread.join()
    flir_thread.join()
