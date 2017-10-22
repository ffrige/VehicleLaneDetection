#import camera library
import LaneLib
import cv2
import numpy as np


### STEP1: camera calibration ###

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# folder where the calibration images are stored
cal_folder = 'camera_cal'
img_size = (720,1280)

#find calibration points
objpoints, imgpoints = mylib.FindCalibrationPoints(cal_folder, objpoints, imgpoints)

#calibrate camera -> get undistortion matrix (mtx) and coefficients (dist)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)



### STEP2: video processing ###

#define empty buffers to smooth measurements
buf_size = 5
radius_buf = np.zeros(buf_size)         #buffer for curvature radius
offset_buf = np.zeros(buf_size)         #buffer for center offset
left_fit_0_buf = np.zeros(buf_size)     #buffer for A coeff of left fit
left_fit_1_buf = np.zeros(buf_size)     #buffer for B coeff of left fit
left_fit_2_buf = np.zeros(buf_size)     #buffer for C coeff of left fit
right_fit_0_buf = np.zeros(buf_size)    #buffer for A coeff of right fit
right_fit_1_buf = np.zeros(buf_size)    #buffer for B coeff of right fit
right_fit_2_buf = np.zeros(buf_size)    #buffer for C coeff of right fit

#initial values
radius = 0
offset = 0
left_fit = np.array([0,0,0])
right_fit = np.array([0,0,0])

#read frames from video file
cap = cv2.VideoCapture('project_video.mp4')
frame_cnt = 0

while(cap.isOpened()):
        ret, frame = cap.read()
        
        #stop at end of file
        if frame is None:
                break

        #ignore filter threshold until array is full
        frame_cnt += 1
        if frame_cnt > buf_size:
                threshold = 300
        else:
                threshold = 0.0

        #apply pipeline, feed values from previous frame
        left_fit, right_fit, radius, offset, Minv, frame_undist = mylib.FramePipeline(frame, mtx, dist, left_fit, right_fit, radius, offset)

        #weight average new values
        radius, radius_buf = mylib.MovingAverageFilter(radius_buf,radius,threshold)
        offset, offset_buf = mylib.MovingAverageFilter(offset_buf,offset,threshold)
        left_fit[0], left_fit_0_buf = mylib.MovingAverageFilter(left_fit_0_buf,left_fit[0],threshold)
        left_fit[1], left_fit_1_buf = mylib.MovingAverageFilter(left_fit_1_buf,left_fit[1],threshold)
        left_fit[2], left_fit_2_buf = mylib.MovingAverageFilter(left_fit_2_buf,left_fit[2],threshold)
        right_fit[0], right_fit_0_buf = mylib.MovingAverageFilter(right_fit_0_buf,right_fit[0],threshold)
        right_fit[1], right_fit_1_buf = mylib.MovingAverageFilter(right_fit_1_buf,right_fit[1],threshold)
        right_fit[2], right_fit_2_buf = mylib.MovingAverageFilter(right_fit_2_buf,right_fit[2],threshold)
                        
        #plot image with superimposed information
        frame_dst = mylib.WarpBack(frame_undist, left_fit, right_fit, Minv)
        cv2.putText(frame_dst,'Offset from center = '+str('%.3f'%(offset))+' m',(50,50),2,1.5,(51,153,255),2)
        cv2.putText(frame_dst,'Road curvature = '+str('%.3f'%(radius))+' m',(50,100),2,1.5,(51,153,255),2)

        cv2.imshow('Video',frame_dst)
        cv2.imwrite('video_images/frame'+(str(frame_cnt)).zfill(4)+'.jpg',frame_dst)

        #abort with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()



