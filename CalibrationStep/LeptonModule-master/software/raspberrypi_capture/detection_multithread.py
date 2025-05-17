import cv2
import os
import numpy as np
import json
import ctypes
import time
import threading
from primesense import openni2
from primesense import _openni2 as c_api

# ============ Global Variables & Constants ============
DILATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
DILATION_KERNEL_FLIR = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
zeros_template = None

# Preallocate global buffers (they will be set in main)
dpt_norm_buffer = None
flir_norm_buffer = None

# Shared sensor data and lock (used by sensor threads)
sensor_data = { 'rgb': None, 'lidar': None, 'flir': None }
data_lock = threading.Lock()
running = True  # Control flag for threads

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

# -------------------------------------------------------------------
# Sensor Get Functions (Using the Initialized Objects)
# -------------------------------------------------------------------

def get_rgb(cap):   
    ret, frame = cap.read()
    rgb_timestamp = time.time()
    if not ret:
        print("Error: Failed to grab frame.")
        cap.release()
        return

    b, g, r = cv2.split(frame)
    
    # --- New Robustness Check for Contrast ---
    # Calculate the difference between the maximum and minimum pixel intensity for each channel

    # Define a threshold for minimal acceptable intensity range
    INTENSITY_THRESHOLD = 80
    

    # Using the helper function that reuses the zeros_template
    blue_display = create_channel_display(b, 0)   # Channel index 0 for blue
    green_display = create_channel_display(g, 1)    # Channel index 1 for green
    red_display = create_channel_display(r, 2)      # Channel index 2 for red
    kernel = DILATION_KERNEL
            
    # Process Blue channel
    b_blur = cv2.GaussianBlur(b, (5, 5), 0)
    diff_b = np.max(b_blur) - np.min(b_blur)
    flag_b = (diff_b >= INTENSITY_THRESHOLD)
    
    if flag_b:
        _, edges_blue = cv2.threshold(b_blur, 0, 255, cv2.THRESH_OTSU)
        edges_blue = cv2.dilate(edges_blue, kernel, iterations=2)
        b_contours = []
        contours_blue, _ = cv2.findContours(edges_blue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours_blue):
            area = cv2.contourArea(c)
            if area < 1e2 or 5e4 < area:
                continue
            b_contours.append(c)
            #cv2.drawContours(blue_display, contours_blue, i, (255, 0, 0), 2)
        b_contours, ell_b, rect_b = reduce_near_contours(b_contours, b)
        
        for i in rect_b:
            cv2.rectangle(blue_display, i[0],i[1], (255, 255, 255),4)
    else:
        b_contours = []
        ell_b = []
        rect_b = []
    # Process Green channel
    g_blur = cv2.GaussianBlur(g, (5, 5), 0)
    diff_g = np.max(g_blur) - np.min(g_blur)
    flag_g = (diff_g >= INTENSITY_THRESHOLD)
    
    if flag_g:
        _, edges_green = cv2.threshold(g_blur, 0, 255, cv2.THRESH_OTSU)
        edges_green = cv2.dilate(edges_green, kernel, iterations=2)
        g_contours = []
        contours_green, _ = cv2.findContours(edges_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours_green):
            area = cv2.contourArea(c)
            if area < 1e2 or 5e4 < area:
                continue
            g_contours.append(c)
            #cv2.drawContours(green_display, contours_green, i, (0, 255, 0), 2)
        g_contours, ell_g, rect_g = reduce_near_contours(g_contours, g)
        
        for i in rect_g:
            cv2.rectangle(green_display, i[0],i[1], (255, 255, 255),4)
    else:
        g_contours = []
        ell_g = []
        rect_g = []
            
    # Process Red channel
    r_blur = cv2.GaussianBlur(r, (5, 5), 0)
    diff_r = np.max(r_blur) - np.min(r_blur)
    flag_r = (diff_r >= INTENSITY_THRESHOLD)
    
    if flag_r:
        _, edges_red = cv2.threshold(r_blur, 0, 255, cv2.THRESH_OTSU)
        edges_red = cv2.dilate(edges_red, kernel, iterations=2)
        r_contours = []
        contours_red, _ = cv2.findContours(edges_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours_red):
            area = cv2.contourArea(c)
            if area < 1e2 or 5e4 < area:
                continue
            r_contours.append(c)
            #cv2.drawContours(red_display, contours_red, i, (0, 0, 255), 2)
        r_contours, ell_r, rect_r = reduce_near_contours(r_contours, red_display)
        
        for i in rect_r:
            cv2.rectangle(red_display, i[0],i[1], (255, 255, 255),4)
    else:
        r_contours = []
        ell_r = []
        rect_r = []


    # Return timestamp and a tuple with flags and channel data
    channel_data = ( (ell_r, r_contours, red_display, rect_r),
                     (ell_g, g_contours, green_display, rect_g),
                     (ell_b, b_contours, blue_display, rect_b) )
    return (rgb_timestamp, ((flag_r, flag_g, flag_b), channel_data))

def get_lidar(lidar_params):
    
    depth_stream = lidar_params["depth_stream"]
    
    frame = depth_stream.read_frame()
    lidar_timestamp = time.time()
    frame_data = frame.get_buffer_as_uint16()
    depth_img_16 = np.frombuffer(frame_data, dtype=np.uint16).reshape((480, 640))
    dpt_contours = []
    depth_img_16[depth_img_16 == 0] = np.max(depth_img_16)
    depth_img_16 = np.fliplr(depth_img_16)
    cv2.normalize(depth_img_16, dpt_norm_buffer, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    display_frame = dpt_norm_buffer

    ret, thresh = cv2.threshold(display_frame, 0, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    depth_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 1e2 or 1e5 < area:
            continue
        dpt_contours.append(c)
        #cv2.drawContours(depth_frame, contours, i, (0, 0, 255), 2)
    
    dpt_contours, ell_dpt, rect_dpt = reduce_near_contours(dpt_contours, depth_frame)
    
    for i in rect_dpt:
        cv2.rectangle(depth_frame, i[0],i[1], (0, 0, 0),4)    
    
    return (lidar_timestamp, (dpt_contours, ell_dpt, depth_frame, rect_dpt))

def get_flir(flir_params):
    lib = flir_params["lib"]
    buffer_ptr = flir_params["buffer_ptr"]
    frame_buffer = flir_params["frame_buffer"]
    FRAME_WIDTH = flir_params["FRAME_WIDTH"]
    FRAME_HEIGHT = flir_params["FRAME_HEIGHT"]

#     result = call_lib_main_with_timeout(lib, buffer_ptr, timeout_sec=1.0)
    # Capture a new frame using the C function
    result = lib.main(buffer_ptr)
    flir_timestamp = time.time()
    if result != 0:
        print("Error capturing frame:", result)
        return None  
    # Interpret raw data as a 16-bit image and reshape it
    frame_data = frame_buffer.view(dtype=np.uint16).reshape((FRAME_HEIGHT, FRAME_WIDTH))

    flr_as_pic = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    flr_as_pic = cv2.fastNlMeansDenoising(flr_as_pic, None, 90, 3, 9)
    
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
    #denoised_frame = cv2.bilateralFilter(flr_ir, 7,100, 100)
    #denoised_frame = cv2.rotate(denoised_frame, cv2.ROTATE_180)
    denoised_frame = cv2.rotate(flr_ir, cv2.ROTATE_180)
    
    # Threshold the image to create a binary image for contour detection
    ret, thresh = cv2.threshold(denoised_frame, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    canny = cv2.Canny(denoised_frame,75,200)
    canny = cv2.dilate(canny, DILATION_KERNEL_FLIR, iterations=1)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
        #cv2.drawContours(flir_frame, contours, i, (0, 255, 0), 1)
   
    flir_contours, ell_flir, rect_flir = reduce_near_contours(flir_contours, flir_frame)

    for i in rect_flir:
        cv2.rectangle(flir_frame, i[0],i[1], (0, 0, 255),1)
        
    return (flir_timestamp, (flir_contours, ell_flir, flir_frame, rect_flir))
# -------------------------------------------------------------------
# Utility Functions for Detection (reduce_near_contours, near_point, etc.)
# -------------------------------------------------------------------

def create_channel_display(channel, channel_index):
    """
    Creates a display image for the given channel using a preallocated zeros template.
    
    Parameters:
        channel (ndarray): The image data for a single channel.
        channel_index (int): The index for placing the channel. (0 for blue, 1 for green, 2 for red)
    
    Returns:
        ndarray: The merged color image with the channel data in its proper slot.
    """
    # Create a list of three channels by copying the preallocated zeros_template.
    # Use .copy() to ensure each channel gets its own array.
    channels = [zeros_template.copy() for _ in range(3)]
    channels[channel_index] = channel  # Replace the appropriate channel with the actual data.
    return cv2.merge(channels)


def rectangle2(ellipse, max_x, max_y):
    """
    Generates a rectangle circuscribing rectangle for an ellipse
    Parameters
    ----------
    ellipse : Tuple: OpenCV ellipse type
    max_x : int max x axis index for the picture
    max_y : int max y index index for the picture

    Returns
    -------
    Tuple: rectangle OpenCV type
    """
    x = ellipse[0][0]
    y = ellipse[0][1]
    
    mn = ellipse[1][0]
    mj = ellipse[1][1]
    ang = ellipse[2]*np.pi/180
    
    smnx = mn/2*np.cos(ang)
    smny = mn/2*np.sin(ang)
    
    smjx = mj/2*np.sin(ang)
    smjy = mj/2*np.cos(ang)
    
    dx = np.abs(smnx) + np.abs(smjx)
    dy = np.abs(smny) + np.abs(smjy)
    
    x1 = (np.clip(x-dx,0, max_x-1)).astype('uint32')
    y1 = (np.clip(y-dy,0, max_y-1)).astype('uint32')
    x2 = (np.clip(x+dx,0, max_x-1)).astype('uint32')
    y2 = (np.clip(y+dy,0, max_y-1)).astype('uint32')  

    return ((x1,y1),(x2,y2))

def rectangle(contour):
    """
    Generates a rectangle circuscribing rectangle for an cloud point
    Parameters
    ----------
    Countor : array, points to include
    max_x : int max x axis index for the picture
    max_y : int max y index index for the picture

    Returns
    -------
    Tuple: rectangle OpenCV type
    """
    x = contour[:,0,0]
    y = contour[:,0,1]
    
    x1 = (np.min(x)).astype('uint32')
    y1 = (np.min(y)).astype('uint32')
    x2 = (np.max(x)).astype('uint32')
    y2 = (np.max(y)).astype('uint32')    

    return ((x1,y1),(x2,y2))

def unite_rectangles(rect):
    """
    This funtion unite all the overlaping rectangles in a big rectangle type
    Parameters
    ----------
    rect : Tuple: rectangle OpenCV type

    Returns
    -------
    uni_rect :  Tuple: rectangle OpenCV type
    
    """
    uni_rect = []
    remove = []
    bias = 0.05
    for i in range(len(rect)):
        
        if i in remove:
            continue
        
        rect0 = rect[i]
        x01 = rect0[0][0]
        y01 = rect0[0][1]
        x02 = rect0[1][0]
        y02 = rect0[1][1] 
        new_rect = ((x01,y01),(x02,y02))
        
        for j in range(len(rect)):
            if j < i:
                continue
            x1 = rect[j][0][0]
            y1 = rect[j][0][1]
            x2 = rect[j][1][0]
            y2 = rect[j][1][1] 
            dx = np.abs(x2-x1)
            dy = np.abs(y2-y1)
            
            check = False
            
            if (x1-bias*dx) <= x01 and (x2+bias*dx) >= x01:
                if (y1-bias*dy) <= y01 and (y2+bias*dy) >= y01:
                    check = True
                elif (y1-bias*dy) <= y02 and (y2+bias*dy) >= y02:
                    check = True
            elif (x1-bias*dx) <= x02 and (x2+bias*dx) >= x01:
                if (y1-bias*dy) <= y01 and (y2+bias*dy) >= y01:
                    check = True
                elif (y1-bias*dy) <= y02 and (y2+bias*dy) >= y02:
                    check = True
            
        
            if check and i!=j:
                x01 = np.min([x01,x1])
                y01 = np.min([y01,y1])
                x02 = np.max([x02,x2])
                y02 = np.max([y02,y2])
                new_rect = ((x01,y01),(x02,y02))
                remove.append(j)
            
        uni_rect.append(new_rect)
            
    return uni_rect

def normalize_rect(rec, cal, shp, amp = 1, nor = True):
    """
    Conver a rectagle from a source into the reference frame and vice-versa
    Parameters
    ----------
    rec : Tuple: rectangle OpenCV type
    cal : Tuple: calibration contants
    amp : int, amplification value for low resolution sources (for example, the thermal camera)
          The default is 1.
    nor : Bool, this function can be use to denormalize values, whent nor is True it normalizes,
          when it is False, it denormalizes
          The default is True.

    Returns
    -------
    n_rec : Tuple: rectangle OpenCV type. Return a normalized, or denormalized rectangle
        DESCRIPTION.

    """
    
    if nor:
        scl = cal[0]
        dx  = cal[1]
        dy  = cal[2]
        rot = cal[3]
        pscl = amp
    else:
        scl = 1/cal[0]
        dx  = -cal[1]
        dy  = -cal[2]
        rot = -cal[3]
        pscl = 1/amp
    
    n_rec = [] 
    for i in rec:
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[1][0]
        y2 = i[1][1]
        center_x = pscl*(x1+x2)/2
        center_y = pscl*(y1+y2)/2
        hside_x  = pscl*np.abs(x2-x1)/2
        hside_y  = pscl*np.abs(y2-y1)/2
        
        if amp == 1 or nor:
            ncenter_x = scl*((center_x*np.cos(rot)-center_y*np.sin(rot))-dx)
            ncenter_y = scl*((center_x*np.sin(rot)+center_y*np.cos(rot))-dy)
        else:
            ncenter_x = scl*((center_x*np.cos(rot)-center_y*np.sin(rot))-pscl*dx)
            ncenter_y = scl*((center_x*np.sin(rot)+center_y*np.cos(rot))-pscl*dy)
            
        nhside_x  = scl*hside_x
        nhside_y  = scl*hside_y
        
        nx1 = ncenter_x - nhside_x
        ny1 = ncenter_y - nhside_y
        nx2 = ncenter_x + nhside_x
        ny2 = ncenter_y + nhside_y
        
        if nor == False:
            nx1 = int(np.around(np.clip(nx1,0,shp[1])))
            ny1 = int(np.around(np.clip(ny1,0,shp[0])))
            nx2 = int(np.around(np.clip(nx2,0,shp[1])))
            ny2 = int(np.around(np.clip(ny2,0,shp[0])))
            
        if amp != 1:
            nx1 += 2 
            ny1 += 2 
            nx2 += 2 
            ny2 += 2 
            
        n_rec.append(((nx1,ny1),(nx2,ny2)))
        
    return n_rec
 
def shared_area(rect1,rect2):
    """
    Calculates the shared area between two rectangles (Note: you should use normalized rectangles)

    Parameters
    ----------
    rect1 : Tuple: rectangle OpenCV type
    rect2 : Tuple: rectangle OpenCV type

    Returns
    -------
    float64: The shared area between two rectangles
    """
    
    x11 = rect1[0][0]
    y11 = rect1[0][1]
    x12 = rect1[1][0]
    y12 = rect1[1][1]
    
    x21 = rect2[0][0]
    y21 = rect2[0][1]
    x22 = rect2[1][0]
    y22 = rect2[1][1]

    area1 = np.abs(x12-x11)*np.abs(y12-y11)
    area2 = np.abs(x22-x21)*np.abs(y22-y21)
    
    dx = min(x12, x22) - max(x11, x21)
    dy = min(y12, y22) - max(y11, y21)
    
    if (dx>=0) and (dy>=0):
        area = dx*dy
        return np.min([area/area1,area/area2])
        #return 1
    else:
        return 0

def merge_rec(rect1,rect2):
    """
    Merges two rectangles in one containing both 

    Parameters
    ----------
    rect1 : Tuple: rectangle OpenCV type
    rect2 : Tuple: rectangle OpenCV type

    Returns
    -------
    Tuple: rectangle OpenCV type
    """
    
    x11 = rect1[0][0]
    y11 = rect1[0][1]
    x12 = rect1[1][0]
    y12 = rect1[1][1]
    
    x21 = rect2[0][0]
    y21 = rect2[0][1]
    x22 = rect2[1][0]
    y22 = rect2[1][1]

    #area1 = np.abs(x12-x11)*np.abs(y12-y11)
    #area2 = np.abs(x22-x21)*np.abs(y22-y21)
    
    minx = min(x11, x21)
    miny = min(y11, y21)
    maxx = max(x12, x22)
    maxy = max(y12, y22)
    
    return ((minx,miny),(maxx,maxy))

def reduce_near_contours(contours, img):
    # Filter out contours with fewer than 5 points (which are insufficient for ellipse fitting)
    valid_contours = [c for c in contours if len(c) >= 5]
    if len(valid_contours) == 0:
        return [], [], []
    
    ell = [cv2.fitEllipse(c) for c in valid_contours]
    nr_contour = len(valid_contours)
    dist_border = img.shape[1] / 10
    contour2 = []
    remove = []
    
    for i in range(nr_contour):
        if i in remove:
            continue
        contour = valid_contours[i]
        for j in range(i+1, nr_contour):
            if j in remove:
                continue
            x1 = ell[i][0][0]
            y1 = ell[i][0][1]
            x2 = ell[j][0][0]
            y2 = ell[j][0][1]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist < dist_border:
                contour = np.concatenate((contour, valid_contours[j]))
                valid_contours[i] = contour
                ell[i] = cv2.fitEllipse(contour)
                remove.append(j)
        contour2.append(contour)
    
    ell2 = []
    rect0 = []
    for c in contour2:
        ellipse = cv2.fitEllipse(c)
        rectang = rectangle(c)
        ell2.append(ellipse)
        rect0.append(rectang)
        
    rect = unite_rectangles(rect0)
    return contour2, ell2, rect

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

def cluster_and_merge_rectangles(rectangles, overlap_threshold=0):
    """
    Groups rectangles into clusters based on their overlapping area and then merges
    each cluster into a single rectangle.

    Parameters:
        rectangles (list): List of normalized bounding box tuples [((x1,y1),(x2,y2)), ...].
        overlap_threshold (float): Minimum shared area ratio to consider rectangles as overlapping.
                                   (0 means any overlap qualifies)

    Returns:
        merged_rectangles (list): List of merged bounding boxes, one per cluster.
    """
    n = len(rectangles)
    if n == 0:
        return []
    
    # Build a graph (using an adjacency list) based on the shared area
    graph = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            # Use the provided shared_area function
            if shared_area(rectangles[i], rectangles[j]) > overlap_threshold:
                graph[i].add(j)
                graph[j].add(i)
                
    # Use DFS to find connected components in the graph
    visited = set()
    clusters = []
    
    for i in range(n):
        if i not in visited:
            stack = [i]
            cluster = []
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    cluster.append(rectangles[current])
                    # Add neighbors (but subtract the already visited indices)
                    stack.extend(graph[current] - visited)
            clusters.append(cluster)
    
    # Merge rectangles in each cluster (using merge_rec iteratively)
    merged_rectangles = []
    for cluster in clusters:
        merged = cluster[0]
        for rect in cluster[1:]:
            merged = merge_rec(merged, rect)
        merged_rectangles.append(merged)
    
    return merged_rectangles

def rects_to_list(rects):
    # Convert [((x1,y1),(x2,y2)), …] → [ [[x1,y1],[x2,y2]], … ]
    return [
        [ list(r[0]), list(r[1]) ]
        for r in rects
    ]

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
# Main Workflow: Real-Time Detection, Live Visualization, and Saving
# -------------------------------------------------------------------
if __name__ == "__main__":
    #Get the calibration values from differen measurements in the Calibration folder
    path = os.getcwd()
    folder_list = os.listdir(path)

    #Get the calibration values from differen measurements in the Calibration folder

    with open(path+'/'+ 'dd.json', 'r') as file:
        json_dd = json.load(file)
        file.close()
    with open(path+'/'+ 'gd.json', 'r') as file:
        json_gd = json.load(file)
        file.close()
    with open(path+'/'+ 'rd.json', 'r') as file:
        json_rd = json.load(file)
        file.close()
    with open(path+'/'+ 'bd.json', 'r') as file:
        json_bd = json.load(file)
        file.close()
    with open(path+'/'+ 'fd.json', 'r') as file:
        json_fd = json.load(file)
        file.close()

    cal_dd = [json_dd['scale'],json_dd['dx'],json_dd['dy'],json_dd['rotation']]
    cal_gb = [json_gd['scale'],json_gd['dx'],json_gd['dy'],json_gd['rotation']]
    cal_rb = [json_rd['scale'],json_rd['dx'],json_rd['dy'],json_rd['rotation']]
    cal_db = [json_bd['scale'],json_bd['dx'],json_bd['dy'],json_bd['rotation']]
    cal_fb = [json_fd['scale'],json_fd['dx'],json_fd['dy'],json_fd['rotation']]
    # Initialize sensors once
    cap = init_rgb()                 # For RGB sensor
    ret, init_frame = cap.read()
    if not ret:
        print("Error: Couldn't read an initial frame.")
        exit()
    zeros_template = np.zeros_like(init_frame[:, :, 0])
    
    lidar_params = init_lidar()      # For LiDAR sensor
    # For a LiDAR frame of size 480x640, preallocate a buffer:
    dpt_norm_buffer = np.empty((480, 640), dtype=np.uint8)
    
    flir_params = init_flir()        # For FLIR sensor
    # Preallocate a buffer with the dimensions of your FLIR frame, using np.uint8.
    flir_norm_buffer = np.empty((flir_params["FRAME_HEIGHT"], flir_params["FRAME_WIDTH"]), dtype=np.uint8)

    # Assuming RGB and depth images are 480x640
    rgb_shp = (480, 640)
    dpt_shp = (480, 640)
    flr_shp = (60, 80)
    # Create and start sensor threads
    rgb_thread = RGBThread(cap)
    lidar_thread = LiDARThread(lidar_params)
    flir_thread = FLIRThread(flir_params)
    rgb_thread.start()
    lidar_thread.start()
    flir_thread.start()

    # Define an acceptable threshold in seconds (e.g., 50 ms)
    ACCEPTABLE_THRESHOLD = 0.07
    FRAME_TIMEOUT          = 0.15   # max age (in seconds) of any frame
    test = []
    fourcc = cv2.VideoWriter_fourcc(*'XVID')      # codec: XVID, MJPG, MP4V, etc.
    fps    = 20.0                                # choose your frame rate
    width  = 600                                 # mosaic width
    height = 600                                 # mosaic height
    out_mosaic = cv2.VideoWriter('fusion_output.avi', fourcc, fps, (width, height))
    while True:
        with data_lock:
            rgb_data = sensor_data.get('rgb')
            lidar_data = sensor_data.get('lidar')
            flir_data = sensor_data.get('flir')

        # Ensure that all sensor data is available before processing
        if rgb_data is None or lidar_data is None or flir_data is None:
            time.sleep(0.01)
            continue

        # Unpack the RGB data from get_rgb()
        rgb_timestamp, rgb_result = rgb_data
        (flag_r, flag_g, flag_b), channel_data = rgb_result
        (red_data, green_data, blue_data) = channel_data
        (ell_r, r_contours, red_display, rect_r) = red_data
        (ell_g, g_contours, green_display, rect_g) = green_data
        (ell_b, b_contours, blue_display, rect_b) = blue_data

        
        lidar_timestamp, lidar_result = lidar_data
        flir_timestamp, flir_result = flir_data
        
        # Calculate differences; for example, relative to the RGB timestamp:
        delta_lidar = abs(flir_timestamp - lidar_timestamp)
        delta_rgb  = abs(flir_timestamp - rgb_timestamp)
        print(f"Flir timestamp: {flir_timestamp:.3f} | LiDAR diff: {delta_lidar:.3f} s | Rgb diff: {delta_rgb:.3f} s")
        
        # Check if the differences are within the acceptable threshold:
        if delta_lidar > ACCEPTABLE_THRESHOLD and delta_rgb > ACCEPTABLE_THRESHOLD:
            # Frames are not sufficiently synchronized
            # You can choose to skip this set or try to interpolate later
            print("Frames not synchronized well enough. Waiting for a better sync...")
            time.sleep(0.005)  # Optionally add a short delay before continuing
            continue
        

        dpt_contours, ell_dpt, depth_frame, rect_dpt = lidar_result

        # Check for FLIR result correctness
        if flir_result is None:
            print("Failed to acquire FLIR frame.")
            continue
        flir_contours, ell_flir, flir_frame, rect_flir = flir_result
        # 2) Drop any sensor whose frame is too old
        now       = time.time()
        age_rgb   = now - rgb_timestamp
        age_lidar = now - lidar_timestamp
        age_flir  = now - flir_timestamp

        use_rgb   = (age_rgb   <= FRAME_TIMEOUT)
        use_lidar = (age_lidar <= FRAME_TIMEOUT)
        use_flir  = (age_flir  <= FRAME_TIMEOUT)
        
        nrect_r = normalize_rect(rect_r, cal_rb, rgb_shp)
        nrect_g = normalize_rect(rect_g, cal_gb, rgb_shp)
        nrect_b = normalize_rect(rect_b, cal_db, rgb_shp)
        nrect_dpt = normalize_rect(rect_dpt, cal_dd, dpt_shp)
        nrect_flir = normalize_rect(rect_flir, cal_fb, flr_shp, amp=8)
        
        if not use_rgb:
            nrect_r, nrect_g, nrect_b = [], [], []
            print(f"Ignoring stale RGB   frame ({age_rgb:.3f}s old)")
        if not use_lidar:
            nrect_dpt = []
            print(f"Ignoring stale LiDAR frame ({age_lidar:.3f}s old)")
        if not use_flir:
            nrect_flir = []
            print(f"Ignoring stale FLIR  frame ({age_flir:.3f}s old)")

        conca_nor = nrect_r + nrect_g + nrect_b + nrect_dpt + nrect_flir
        
        # --- Cluster and Merge overlapping rectangles ---
        # conca_nor contains all normalized detection rectangles from RGB and depth sensors
        merged_rect_r = cluster_and_merge_rectangles(nrect_r, overlap_threshold=0)
        merged_rect_g = cluster_and_merge_rectangles(nrect_g, overlap_threshold=0)
        merged_rect_b = cluster_and_merge_rectangles(nrect_b, overlap_threshold=0)
        merged_rect_dpt = cluster_and_merge_rectangles(nrect_dpt, overlap_threshold=0)
        merged_rect_flir = cluster_and_merge_rectangles(nrect_flir, overlap_threshold=0)
        merged_rectangles = cluster_and_merge_rectangles(conca_nor, overlap_threshold=0)


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
            "fused_rectangles": rects_to_list(merged_rectangles)
        }
        test.append(record)


        # Denormalize the merged rectangles for the Lidar
        dn_merged_dpt = normalize_rect(merged_rectangles, cal_dd, dpt_shp, nor=False)
        for rect in dn_merged_dpt:
            cv2.rectangle(depth_frame, rect[0], rect[1], (0, 255, 0), 2)
        
        # Denormalize the merged rectangles for the Red channel (RGB)
        dnrect_r = normalize_rect(merged_rectangles, cal_rb, rgb_shp, nor=False)
        for rect in dnrect_r:
            cv2.rectangle(red_display, rect[0], rect[1], (0, 255, 0), 2)

        # Denormalize the merged rectangles for the Green channel (RGB)
        dnrect_g = normalize_rect(merged_rectangles, cal_gb, rgb_shp, nor=False)
        for rect in dnrect_g:
            cv2.rectangle(green_display, rect[0], rect[1], (0, 255, 0), 2)

        # Denormalize the merged rectangles for the Blue channel (RGB)
        dnrect_b = normalize_rect(merged_rectangles, cal_db, rgb_shp, nor=False)
        for rect in dnrect_b:
            cv2.rectangle(blue_display, rect[0], rect[1], (0, 255, 0), 2)
        
        # Denormalize the merged rectangles for the FLIR image.
        # Note: The amplification factor (8) is used to adapt to the low native resolution of FLIR.
        dnrect_flir = normalize_rect(merged_rectangles, cal_fb, flr_shp, 8, nor=False)
        for rect in dnrect_flir:
            # For the FLIR view, we subtract a small offset (e.g., 2 pixels) as shown in your original code.
            cv2.rectangle(flir_frame, np.array(rect[0]) - 2, np.array(rect[1]) - 2, (0, 255, 0), 1)

        flir_frame = cv2.resize(flir_frame, (80 * 10, 80 * 10), interpolation=cv2.INTER_NEAREST)
        # --- Resize images to a common size (300x300 for example) ---
        # Here, we assume `display_frame` is the resized thermal image from your FLIR function.
        thermal_resized = cv2.resize(flir_frame, (300, 300))
        b_resized = cv2.resize(blue_display, (300, 300))
        g_resized = cv2.resize(green_display, (300, 300))
        r_resized = cv2.resize(red_display, (300, 300))

        # --- Build the mosaic for Thermal + RGB ---
        # Arrange as a 2x2 grid:
        # Top Row: Thermal | Blue
        # Bottom Row: Green | Red
        top_row = np.hstack((thermal_resized, b_resized))
        bottom_row = np.hstack((g_resized, r_resized))
        mosaic = np.vstack((top_row, bottom_row))

        # --- Display composite views ---
        # Fusion view: Thermal + RGB mosaic in one window.
        cv2.imshow("Fusion View (Thermal + RGB)", mosaic)
        out_mosaic.write(mosaic)

        # LiDAR depth frame remains in its own window.
        cv2.imshow("LiDAR Depth", depth_frame)
        


        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
        # 4) After exiting the loop, write everything out:
    with open("normalized_rects.json", "w") as f:
        json.dump(test, f, indent=2)
    print(f"Saved {len(test)} frames of data to normalized_rects.json")
    
    out_mosaic.release()
    cv2.destroyAllWindows()
    rgb_thread.join()
    lidar_thread.join()
    flir_thread.join()
