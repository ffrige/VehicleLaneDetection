import cv2
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from skimage.feature import hog
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import glob
from VehicleLib import FindCars

model = LinearSVC()

#TRAIN new MODEL
#X_train, X_test, y_train, y_test, scaler = LoadData('training_images')
#TrainModel(model, X_train, y_train)

#LOAD old MODEL
X_train = joblib.load('X_train.pkl') 
X_test = joblib.load('X_test.pkl') 
y_train = joblib.load('y_train.pkl') 
y_test = joblib.load('y_test.pkl') 
scaler = joblib.load('scaler.pkl')
model = joblib.load('model_lin.pkl')

#print('Test Accuracy of SVM = ', round(model.score(X_test, y_test), 3))

#run classifier on each frame of the video
cap = cv2.VideoCapture('project_video.mp4')
frame_cnt = 0
oldcars = []

while(cap.isOpened()):
        ret, frame = cap.read()

        frame_cnt += 1
        
        #stop at end of file
        if frame is None:
                break

        #apply pipeline
        oldcars = FindCars(frame, model, scaler, oldcars)
        
        cv2.imshow('Video',frame)
        cv2.imwrite('video_images/frame'+(str(frame_cnt)).zfill(4)+'.jpg',frame)

        #abort with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()


