import numpy as np
import cv2
import glob
#import matplotlib.pyplot as plt

# Define a function that finds calibration points
# given a folder full of calibration images
def FindCalibrationPoints(cal_folder, objpoints, imgpoints):

    # prepare object points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    images = glob.glob(str(cal_folder)+'/calibration*.jpg')

    # search for corners in images
    for i, imgname in enumerate(images):
        img = cv2.imread(imgname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #write_name = 'output_images/corners_found'+str(i)+'.jpg'
            #cv2.imwrite(write_name, img)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

    cv2.destroyAllWindows()

    return (objpoints, imgpoints)



    
# Define a function that converts an RGB image into a binary
# combination of color and gradients thresholds
# NOTE that threshold values are hard coded already!
def ImgToBinary(img):

    #smooth filter
    img = cv2.bilateralFilter(img, 15, 100, 100)

    #extract HLSV channels and combine their thresholded binaries
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    H = img_hls[:,:,0]
    L = img_hls[:,:,1]
    S = img_hls[:,:,2]
    V = img_hsv[:,:,2]
    
    binary_colors =  np.zeros_like(L)    
    binary_colors[((S > 50) & (S < 200))|((L > 200) & (L < 240))|((H > 200) & (H < 220))] = 1
    binary_colors[V < 220] = 0

    #calculate gradients on gray image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #calculate gradients along x and y axes
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    #normalize
    abs_sobel =  np.absolute(sobel_y)
    scaled_sobel_y = np.uint8(255*abs_sobel/np.max(abs_sobel))
    abs_sobel =  np.absolute(sobel_x)
    scaled_sobel_x = np.uint8(255*abs_sobel/np.max(abs_sobel))

    #calculate magnitude of gradients
    gradmag = np.sqrt(sobel_x**2 + sobel_y**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8) 

    #calculate direction of gradients
    absgraddir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))

    #keep only pixels within thresholds
    binary_gradients = np.zeros_like(gradmag)
    binary_gradients[(gradmag >= 20) & (gradmag <= 255)
                  & (absgraddir >= 0.7) & (absgraddir <= 2)
                  & (scaled_sobel_x >= 20) & (scaled_sobel_x <= 255)
                  & (scaled_sobel_y >= 20) & (scaled_sobel_y <= 255)] = 1

    combined_binary = np.zeros_like(binary_gradients)
    combined_binary[(binary_colors == 1) | (binary_gradients == 1)] = 1

    return combined_binary


    

# Define a function that unwarps an image
# given source and destination points
def ImgUnwarp(img, src, dst):

    # calculate perspective matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # warp image
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]))

    """
    #draw source points (trapezoid)
    cv2.line(img, (src[0][0],src[0][1]),(src[1][0],src[1][1]), (1,0,0), 5)
    cv2.line(img, (src[1][0],src[1][1]),(src[2][0],src[2][1]), (1,0,0), 5)
    cv2.line(img, (src[2][0],src[2][1]),(src[3][0],src[3][1]), (1,0,0), 5)
    cv2.line(img, (src[3][0],src[3][1]),(src[0][0],src[0][1]), (1,0,0), 5)

    #draw destination points (rectangle)
    cv2.line(warped, (dst[0][0],dst[0][1]),(dst[1][0],dst[1][1]), (1,0,0), 5)
    cv2.line(warped, (dst[1][0],dst[1][1]),(dst[2][0],dst[2][1]), (1,0,0), 5)
    cv2.line(warped, (dst[2][0],dst[2][1]),(dst[3][0],dst[3][1]), (1,0,0), 5)
    cv2.line(warped, (dst[3][0],dst[3][1]),(dst[0][0],dst[0][1]), (1,0,0), 5)
    """
    
    return warped, M, Minv



# Define a function to identify the pixels belonging to the lane lines
# Use histogram and sliding windows
def FindLines(img, prev_left_fit, prev_right_fit):

    X = img.shape[1]
    Y = img.shape[0]

    # we have 9 regions, two points per region (one left, one right)
    nb_regions = 9
    reg_height = int(Y/nb_regions)
    reg_offset = int(Y/nb_regions)
    PointsLeftX = []
    PointsLeftY = []
    PointsRightX = []
    PointsRightY = []

    # start searching around previous lines ends
    prev_left_pos = int(prev_left_fit[0]*Y*Y+prev_left_fit[1]*Y+prev_left_fit[2])
    prev_right_pos = int(prev_right_fit[0]*Y*Y+prev_right_fit[1]*Y+prev_right_fit[2])
    
    for i in range(nb_regions):
        # get histogram of region
        highY = Y-reg_offset*i
        lowY = highY - reg_height
        histogram = np.sum(img[lowY:highY,:], axis=0)
        #plt.plot(histogram)

        # look for left and right peaks of histogram
        # use neighborhood of previously detected lines
        delta = 30
        if prev_left_pos > 0 and prev_right_pos > 0: #search around previous position
            if len(histogram[prev_left_pos-delta:prev_left_pos+delta]) == 0:
                left_peak_intensity = 0
                left_peak = 0
            else:
                left_peak_intensity = np.max(histogram[prev_left_pos-delta:prev_left_pos+delta])
                left_peak = np.argmax(histogram[prev_left_pos-delta:prev_left_pos+delta]) + (prev_left_pos-delta)

            if len(histogram[prev_right_pos-delta:prev_right_pos+delta]) == 0:
                right_peak_intensity = 0
                right_peak = 0
            else:
                right_peak_intensity = np.max(histogram[prev_right_pos-delta:prev_right_pos+delta])
                right_peak = np.argmax(histogram[prev_right_pos-delta:prev_right_pos+delta]) + (prev_right_pos-delta)            
                

        else: #search whole frame width
            middle = np.int(histogram.shape[0]/2)
            left_peak_intensity = np.max(histogram[:middle])
            right_peak_intensity = np.max(histogram[middle:])
            left_peak = np.argmax(histogram[:middle])
            right_peak = np.argmax(histogram[middle:]) + middle

        #only append point if peak intensity is larger than noise
        if (left_peak_intensity > 1):
            PointsLeftX.append(left_peak)
            PointsLeftY.append(highY-int(reg_height/2))
            prev_left_pos = left_peak #update center of window search
        else: #no point found -> reset window center
            prev_left_pos = 0

        if (right_peak_intensity > 1):
            PointsRightX.append(right_peak)
            PointsRightY.append(highY-int(reg_height/2))
            prev_right_pos = right_peak #update center of window search
        else: #no point found -> reset window center
            prev_right_pos = 0

    return PointsLeftX,PointsLeftY,PointsRightX,PointsRightY



#Define a function that fits polynomials in set of points
# then calculates center offset and radius of curvature
def GeometricalCalculations(PointsLeftX,PointsLeftY,PointsRightX,PointsRightY):

    X = 1280
    Y = 720

    #fit quadratic curve between calculated points (in pixels)
    left_fit = np.polyfit(PointsLeftY, PointsLeftX, 2)
    right_fit = np.polyfit(PointsRightY, PointsRightX, 2)

    #scale all values
    y_scaling = 30/720 # meters per pixel in y dimension
    x_scaling = 3.7/700 # meters per pixel in x dimension

    PointsLeftY = [i*y_scaling for i in PointsLeftY]
    PointsLeftX = [i*x_scaling for i in PointsLeftX]
    PointsRightY = [i*y_scaling for i in PointsRightY]
    PointsRightX = [i*x_scaling for i in PointsRightX]
    
    #calculate offset from center
    offset = X*x_scaling/2 - (PointsRightX[0]+PointsLeftX[0])/2

    #fit points (in meters)
    left_fit_m = np.polyfit(PointsLeftY, PointsLeftX, 2)
    right_fit_m = np.polyfit(PointsRightY, PointsRightX, 2)

    #calculate radius of curvature
    y_eval = Y*y_scaling
    left_radius = ((1 + (2*left_fit_m[0]*y_eval + left_fit_m[1])**2)**1.5) / np.absolute(2*left_fit_m[0])
    right_radius = ((1 + (2*right_fit_m[0]*y_eval + right_fit_m[1])**2)**1.5) / np.absolute(2*right_fit_m[0])
    #output average value 
    radius = (left_radius+right_radius)/2
    
    return left_fit, right_fit, radius, offset



# Define function that unwarps lane
# and plots it back onto original image
def WarpBack(img, left_fit, right_fit, Minv):

    X = img.shape[1]
    Y = img.shape[0]
    
    # Create an image to draw the lines on
    warped_lane = np.zeros_like(img).astype(np.uint8)

    # Generate x and y values for plotting
    ploty = np.linspace(0, Y-1, Y)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    """
    plt.plot(left_fitx, ploty, color='yellow', linewidth=10)
    plt.plot(right_fitx, ploty, color='yellow', linewidth=10)
    plt.plot(PointsRightX, PointsRightY, color='red')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    """

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warped_lane, np.int_([pts]), (0,255,0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarped_lane = cv2.warpPerspective(warped_lane, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, unwarped_lane, 0.3, 0)
    return result



# Define the final frame processing Pipeline
# by combining all the previously defined functions
def FramePipeline(frame, mtx, dist, prev_left_fit, prev_right_fit, prev_radius, prev_offset):
    
    #apply undistortion transformation matrix
    img_undist = cv2.undistort(frame, mtx, dist, None, mtx)

    #apply thresholds and binarize
    img_binary = ImgToBinary(img_undist)

    #warp image (view from above)
    X = frame.shape[1]
    Y = frame.shape[0]
    Offset_Low = 160 #hard coded after fine tuning
    Offset_High = 573  #hard coded after fine tuning
    src = np.float32([[Offset_Low+30, Y], [X-Offset_Low, Y],[X-Offset_High-2, Y*0.64],[Offset_High+5, Y*0.64]])        
    dst = np.float32([[Offset_Low+30, Y], [X-Offset_Low, Y],[X-Offset_Low, 0],[Offset_Low+30, 0]])
    img_unwarp, M, Minv = ImgUnwarp(img_binary, src, dst)

    #find points belonging to lines
    PointsLeftX,PointsLeftY,PointsRightX,PointsRightY = FindLines(img_unwarp, prev_left_fit, prev_right_fit)

    #fit points with polynomial functions
    #use previous fit if not enough points were found
    if len(PointsLeftX)>2 and len(PointsRightX)>2:
        left_fit, right_fit, radius, offset = GeometricalCalculations(PointsLeftX,PointsLeftY,PointsRightX,PointsRightY)
    else:
        left_fit = prev_left_fit
        right_fit = prev_right_fit
        radius = prev_radius
        offset = prev_offset

    return left_fit, right_fit, radius, offset, Minv, img_undist


# Define a function to calculate the moving average of an array given a new input value
def MovingAverageFilter(old_array, new_value, threshold):

    # ignore values that are too far away from current average
    # do not consider zero points when averaging (possibly because buffer is not entirely full yet)
    old_array_nonzero = old_array[np.nonzero(old_array)]
    if len(old_array_nonzero) == 0:
        average = 0
    else:
        average = np.average(old_array_nonzero)

    diff = np.abs(new_value-average)
    if diff > np.abs(average*threshold) and threshold > 0:
        return average, old_array

    new_array = np.roll(old_array,1)
    new_array[0] = new_value
    average = np.average(new_array[np.nonzero(new_array)])

    return average, new_array
 
