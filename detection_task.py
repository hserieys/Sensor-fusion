import cv2
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
import ctypes
import threading
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
    b_contours, ell_b, rect_b = reduce_near_contours(b_contours, b)

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
    g_contours, ell_g, rect_g = reduce_near_contours(g_contours, g)
            
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
    r_contours, ell_r, rect_r = reduce_near_contours(r_contours, red_display)
    
    for i in rect_r:
        cv2.rectangle(red_display, i[0],i[1], (255, 0, 0),2)

    for i in rect_g:
        cv2.rectangle(green_display, i[0],i[1], (0, 255, 0),2) 
        
    for i in rect_b:
        cv2.rectangle(blue_display, i[0],i[1], (0, 0, 255),2)

    return ell_r, ell_g, ell_b, r_contours, g_contours, b_contours, red_display, green_display, blue_display, rect_r, rect_g, rect_b

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
    
    dpt_contours, ell_dpt, rect_dpt = reduce_near_contours(dpt_contours, depth_frame)
    
    for i in rect_dpt:
        cv2.rectangle(depth_frame, i[0],i[1], (0, 0, 0),4)    
    
    return dpt_contours, ell_dpt, depth_frame, rect_dpt

def get_flir(flir_params):
    lib = flir_params["lib"]
    buffer_ptr = flir_params["buffer_ptr"]
    frame_buffer = flir_params["frame_buffer"]
    FRAME_WIDTH = flir_params["FRAME_WIDTH"]
    FRAME_HEIGHT = flir_params["FRAME_HEIGHT"]
    
    result = lib.main(buffer_ptr)
    if result != 0:
        print("Error capturing FLIR frame:", result)
        return
    # Interpret raw data as a 16-bit image and reshape it
    frame_data = frame_buffer.view(dtype=np.uint16).reshape((FRAME_HEIGHT, FRAME_WIDTH))
    
    # Normalize the 16-bit image into 8-bit and convert its scale
    # Assuming 'frame_data' is your original 16-bit frame (after reshaping)
    frame_norm = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX)
    flr_as_pic = (frame_norm/np.max(frame_norm)*255).astype('uint8')
    
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
    # Threshold the image to create a binary image for contour detection
    ret, thresh = cv2.threshold(denoised_frame, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Convert the denoised image to BGR for drawing colored bounding boxes
    flir_frame = cv2.cvtColor(denoised_frame, cv2.COLOR_GRAY2BGR)
    flir_contours = []
    
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 5e1 or 1e3 < area:
            continue
        # Draw each contour only for visualisation purposes
        flir_contours.append(c)
        cv2.drawContours(flir_frame, contours, i, (0, 255, 0), 1)
    
    flir_contours, ell_flir, rect_flir = reduce_near_contours(flir_contours, flir_frame)

    for i in rect_flir:
        cv2.rectangle(flir_frame, i[0],i[1], (0, 0, 0),1)
        
    return flir_contours, ell_flir, flir_frame, rect_flir
# -------------------------------------------------------------------
# Utility Functions for Detection (reduce_near_contours, near_point, etc.)
# -------------------------------------------------------------------

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
    
def dilatation(val, dilatation_size):
    """
    Dilates de ir image to allow the calibration

    Parameters
    ----------
    val : binary picture array
        binarized Ir picture.

    Returns
    -------
    dilatation_dst : binary picture array
        binarized Ir picture after dilatation.

    """
    #dilatation_size = 7
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(val, None, element)
    return dilatation_dst

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


# -------------------------------------------------------------------
# Main Workflow: Real-Time Detection, Live Visualization, and Saving
# -------------------------------------------------------------------
if __name__ == "__main__":
    use_flir = True  # Flag to indicate whether to use FLIR for calibration
    # Initialize sensors once
    cap = init_rgb()                 # For RGB sensor
    lidar_params = init_lidar()      # For LiDAR sensor
    flir_params = init_flir()        # For FLIR sensor
    cal_rb = (1.0, 0, 0, 0)
    cal_gb = (1.0, 0, 0, 0)
    cal_db = (1.0, 0, 0, 0)
    cal_dd = (1.0, 0, 0, 0)
    cal_fb = (1.0, 0, 0, 0)
    # Assuming RGB and depth images are 480x640
    rgb_shp = (480, 640)
    dpt_shp = (480, 640)
    flr_shp = (480, 640)
    
while True:
    # Acquire FLIR frame only if use_flir is still True
    if use_flir:
        print("Acquiring FLIR frame...")
        flir_result = get_flir(flir_params)
        if flir_result is None:
            print("Failed to acquire FLIR frame, disabling FLIR for subsequent calibrations.")
            use_flir = False
            flir_data = ([], [], None)
        else:
            flir_contours, ell_flir, flir_frame, rect_flir = flir_result
            if flir_frame is None or flir_frame.size == 0:
                print("FLIR frame invalid, disabling FLIR for subsequent calibrations.")
                use_flir = False
                flir_data = ([], [], None)
            elif len(ell_flir) < 6:
                print(f"Not enough FLIR ellipses detected ({len(ell_flir)} found). Disabling FLIR for subsequent calibrations.")
                use_flir = False
                flir_data = ([], [], flir_frame)
            else:
                flir_data = (flir_contours, ell_flir, flir_frame, rect_flir)
    else:
        flir_data = ([], [], None)    
    # Acquire LiDAR frame
    print("Acquiring LiDAR frame...")
    dpt_contours, ell_dpt, depth_frame, rect_dpt = get_lidar(lidar_params)
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
    ell_r, ell_g, ell_b, r_contours, g_contours, b_contours, red_display, green_display, blue_display, rect_r, rect_g, rect_b = rgb_result
    print("RGB frames acquired successfully.")
    
    # --- Existing normalization ---
    nrect_r = normalize_rect(rect_r, cal_rb, rgb_shp)
    nrect_g = normalize_rect(rect_g, cal_gb, rgb_shp)
    nrect_b = normalize_rect(rect_b, cal_db, rgb_shp)
    nrect_flir = normalize_rect(rect_flir, cal_fb, flr_shp, 8)
    nrect_dpt = normalize_rect(rect_dpt, cal_dd, dpt_shp)
    conca_nor = nrect_r + nrect_g + nrect_b + nrect_dpt + nrect_flir

    # --- Compute shared overlapping areas ---
    shared_mtx = []
    for i in conca_nor:
        shared_row = []
        for j in conca_nor:
            shared_row.append(shared_area(i, j))
        shared_mtx.append(shared_row)

    shr_mtx = np.array(shared_mtx)
    shr_0 = np.sum(shr_mtx, axis=0)
    argpivot = np.argmax(shr_0)

    # --- Merge rectangles overlapping with the pivot ---
    merged_indices = []       # To keep track of which rectangles are merged
    pivot = conca_nor[argpivot]
    for i in range(len(conca_nor)):
        # Using a threshold > 0 to decide if there's overlap.
        if shared_mtx[argpivot][i] > 0:
            pivot = merge_rec(pivot, conca_nor[i])
            merged_indices.append(i)

    # --- Identify unmerged rectangles ---
    unmerged_rects = [conca_nor[i] for i in range(len(conca_nor)) if i not in merged_indices]

    # --- Denormalize for sensors display ---
    # For LiDAR
    dnrect_dpt = normalize_rect([pivot], cal_dd, dpt_shp, nor=False)[0]
    dn_unmerged_dpt = normalize_rect(unmerged_rects, cal_dd, dpt_shp, nor=False)

    # For Red channel (RGB)
    dnrect_r = normalize_rect([pivot], cal_rb, rgb_shp, nor=False)[0]
    dn_unmerged_r = normalize_rect(unmerged_rects, cal_rb, rgb_shp, nor=False)

    # For Green channel (RGB)
    dnrect_g = normalize_rect([pivot], cal_gb, rgb_shp, nor=False)[0]
    dn_unmerged_g = normalize_rect(unmerged_rects, cal_gb, rgb_shp, nor=False)

    # For Blue channel (RGB)
    dnrect_b = normalize_rect([pivot], cal_db, rgb_shp, nor=False)[0]
    dn_unmerged_b = normalize_rect(unmerged_rects, cal_db, rgb_shp, nor=False)

    # For FLIR
    dnrect_flir = normalize_rect([pivot], cal_fb, flr_shp, 8, nor=False)[0]
    dn_unmerged_flir = normalize_rect(unmerged_rects, cal_fb, flr_shp, 8, nor=False)


    # Draw on LiDAR frame
    cv2.rectangle(depth_frame, dnrect_dpt[0], dnrect_dpt[1], (0, 255, 0), 3)
    for rect in dn_unmerged_dpt:
        cv2.rectangle(depth_frame, rect[0], rect[1], (255, 255, 255), 1)
        
    # Draw on Red channel display
    cv2.rectangle(red_display, dnrect_r[0], dnrect_r[1], (0, 255, 0), 3)
    for rect in dn_unmerged_r:
        cv2.rectangle(red_display, rect[0], rect[1], (255, 255, 255), 1)
        
    # Draw on Green channel display
    cv2.rectangle(green_display, dnrect_g[0], dnrect_g[1], (0, 255, 0), 3)
    for rect in dn_unmerged_g:
        cv2.rectangle(green_display, rect[0], rect[1], (255, 255, 255), 1)
        
    # Draw on Blue channel display
    cv2.rectangle(blue_display, dnrect_b[0], dnrect_b[1], (0, 255, 0), 3)
    for rect in dn_unmerged_b:
        cv2.rectangle(blue_display, rect[0], rect[1], (255, 255, 255), 1)
        
    # Draw on FLIR frame (if FLIR is enabled)
    if flir_frame is not None:
        cv2.rectangle(flir_frame, dnrect_flir[0], dnrect_flir[1], (0, 255, 0), 3)
        for rect in dn_unmerged_flir:
            cv2.rectangle(flir_frame, rect[0], rect[1], (255, 255, 255), 1)


    
    cv2.imshow("Depth Frame", depth_frame)
    cv2.imshow("Red Channel", red_display)
    cv2.imshow("Green Channel", green_display)
    cv2.imshow("Blue Channel", blue_display)
    if flir_frame is not None:
        cv2.imshow("FLIR Frame", flir_frame)

    
    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()