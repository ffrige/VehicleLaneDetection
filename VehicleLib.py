import cv2
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from skimage.feature import hog
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import glob


#Extract HOG features from image
def HOGfeatures(img, start_corner, end_corner, draw=False):
    
    orient = 9   #number of possible orientations of the gradients
    cell_per_block = 2   #number of cells per block (2x2)
    pix_per_cell = 8   #number of pixels per cell (8x8)

    img_col = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    img_clipped = img_col[start_corner[1]:end_corner[1],start_corner[0]:end_corner[0],:]

    hog_features = []

    if draw: #output image
        for channel in range(3):
            features, hog_image = hog(img_clipped[:,:,channel], orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      feature_vector=False,visualise=True)
            hog_features.append(features)
        return hog_features, hog_image

    else:
        for channel in range(3):
            features = hog(img_clipped[:,:,channel], orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           feature_vector=False,visualise=False)
            hog_features.append(features)
        return hog_features
        


#load dataset and return feature and label vectors
def LoadData(folder):

    img_cars = glob.glob(folder+'/vehicles/**/*.png', recursive=True)
    img_not_cars = glob.glob(folder+'/non-vehicles/**/*.png', recursive=True)

    print('cars: ',len(img_cars))
    print('not cars: ',len(img_not_cars))

    #load features
    X = []
    
    for file in img_cars:
        img = cv2.imread(file)
        X.append(np.array(HOGfeatures(img,[0,0],[img.shape[1],img.shape[0]],draw=False)).astype(np.float64).ravel())
        #if len(X)%100==0:
        #    print(len(X))

    for file in img_not_cars:
        img = cv2.imread(file)
        X.append(np.array(HOGfeatures(img,[0,0],[img.shape[1],img.shape[0]],draw=False)).astype(np.float64).ravel())
        #if len(X)%100==0:
        #    print(len(X))

    #define labels
    y = np.hstack((np.ones(len(img_cars)), 
              np.zeros(len(img_not_cars))))

    print('Dataset size: ',len(X),len(y))
    
    #scale features
    scaler = StandardScaler().fit(X)
    scaled_X = scaler.transform(X)

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    #save features, labels and scaler to files
    joblib.dump(X_train, 'X_train.pkl') 
    joblib.dump(X_test, 'X_test.pkl') 
    joblib.dump(y_train, 'y_train.pkl') 
    joblib.dump(y_test, 'y_test.pkl') 
    joblib.dump(scaler, 'scaler.pkl') 

    return X_train, X_test, y_train, y_test, scaler


#Train Classifier
def TrainModel(model, X_train, y_train):

    model.fit(X_train, y_train)

    #save model to file
    joblib.dump(model, 'model_lin.pkl') 


#return sliding windows given an image and windows parameters
def SlidingWindows(img, windows, start_corner, end_corner, size, step, color, draw=False):

    dx = end_corner[0] - start_corner[0]
    dy = end_corner[1] - start_corner[1]

    if dx<0 or dy<0 or step[0]==0 or step[1]==0 or size[0]==0 or size[1]==0:
        return windows

    nx = int((dx-size[0])/step[0])+1
    ny = int((dy-size[1])/step[1])+1

    for idx_y in range(ny):
        for idx_x in range(nx):
            x0 = start_corner[0]+idx_x*step[0]
            y0 = start_corner[1]+idx_y*step[1]
            point0 = (x0,y0)
            x1 = x0 + size[0]
            y1 = y0 + size[1]
            point1 = (x1,y1)
            windows.append((point0,point1))
            if draw:
                cv2.rectangle(img, point0, point1, color, 2)
    
    return None


#run classifier on all windows
def CheckWindows(model, scaler, img, windows, img_features, Xmin, Ymin, color, draw=False):

    windows_with_car = []

    for window in windows:

        #get HOG features for this window and scale them
        subsample = (window[1][1]-window[0][1])//64
        features = np.array(img_features)[:,(window[0][1]-Ymin)//8:(window[1][1]-Ymin)//8-1:subsample,(window[0][0]-Xmin)//8:(window[1][0]-Xmin)//8-1:subsample,:,:,:]
        features = features[:,-7:,-7:,:,:,:]
        features_scaled = scaler.transform(features.ravel().reshape(1,-1))

        #test with classifier
        prediction = model.predict(features_scaled)

        #save window if car found
        if prediction:
            windows_with_car.append(window)
            if draw:
                cv2.rectangle(img, window[0], window[1], color, 2)

    return windows_with_car


#generate heatmap of classified windows and apply threshold
def HeatMap(img, windows_with_car, threshold):

    heat_map = np.empty_like(img)

    for window in windows_with_car:
        heat_map[window[0][1]:window[1][1], window[0][0]:window[1][0],2] += 1

    heat_map[heat_map <= threshold] = 0

    #normalize
    if (np.max(heat_map)>0):
        heat_map = heat_map/np.max(heat_map)

    return heat_map


#draw bounding boxes on image given labels for detected cars
def DrawBBoxes(img, labels, color, draw=False):

    bboxes = []
    
    for car in range(1,labels[1]+1):
        nonzero = (labels[0] == car).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
        if draw:
            cv2.rectangle(img, bbox[0], bbox[1], color, 6)
        
    return bboxes

    
#extract center points from windows
def Centers(windows):

    centers = []
    
    for window in windows:
        x = (window[1][0]+window[0][0])//2
        y = (window[1][1]+window[0][1])//2
        centers.append((x,y))

    return centers



#detect cars on image
def FindCars(img, model, scaler, oldcars):

    #extract HOGfeatures for the whole ROI in the image
    offset = 128
    Ymin = 360
    Ymax = 650
    if (len(oldcars)>0):
        Xmin = min(1024,np.min(np.array(oldcars)[:,0]-offset))
    else:
        Xmin = 1024
    Xmax = img.shape[1]
    start_corner = [Xmin,Ymin]
    end_corner = [Xmax,Ymax]
    img_features = HOGfeatures(img, start_corner, end_corner, draw=False)

    windows = []
    windows_with_car = []
    bboxes = []

    ####################################
    #COARSE pass over right edge of ROI#
    ####################################
    
    size = [128,128]
    step = [32,32]
    start_corner = [Xmin,Ymin]
    end_corner = [img.shape[1],Ymin+size[1]*2]
    
    SlidingWindows(img, windows, start_corner, end_corner, size, step, color=(0,0,255), draw=False)

    #run classifier on each window
    new_windows_with_car = CheckWindows(model, scaler, img, windows, img_features, Xmin, Ymin, color=(0,255,0), draw=False)

    #add new found cars to old ones
    newcars = Centers(new_windows_with_car)
    for newcar in newcars:
        #make sure that this car is not present already
        exists = 0
        too_close = 64
        for oldcar in oldcars:
            if abs(newcar[0]-oldcar[0])<too_close
            and abs(newcar[1]-oldcar[1])<too_close:
                exists = 1
        if not exists:
            oldcars.append(newcar)

    
    
    ###########################################
    #FINE pass around previously detected cars#
    ###########################################
    
    for oldcar in oldcars:
        #SMALL WINDOWS
        size = [64,64]
        step = [8,8]
        start_corner = [max(Xmin,oldcar[0]-offset),max(Ymin,oldcar[1]-offset)]
        end_corner = [min(Xmax,oldcar[0]+offset),min(Ymax,oldcar[1]+offset)]
        SlidingWindows(img, windows, start_corner, end_corner, size, step, color=(0,0,255), draw=False)
        #MEDIUM WINDOWS
        size = [128,128]
        step = [8,8]
        start_corner = [max(Xmin,oldcar[0]-offset),max(Ymin,oldcar[1]-offset)]
        end_corner = [min(Xmax,oldcar[0]+offset),min(Ymax,oldcar[1]+offset)]
        SlidingWindows(img, windows, start_corner, end_corner, size, step, color=(0,0,255), draw=False)
        #LARGE WINDOWS
        size = [192,192]
        step = [8,8]
        start_corner = [max(Xmin,oldcar[0]-offset),max(Ymin,oldcar[1]-offset)]
        end_corner = [min(Xmax,oldcar[0]+offset),min(Ymax,oldcar[1]+offset)]
        SlidingWindows(img, windows, start_corner, end_corner, size, step, color=(0,0,255), draw=False)

        #run classifier on each window
        windows_with_car = CheckWindows(model, scaler, img, windows, img_features, Xmin, Ymin, color=(0,255,0), draw=False)
        #print("windows_with_car ",len(windows_with_car))
    
    #filter duplicates and false positives with heatmap
    heat_map = HeatMap(img, windows_with_car, threshold = 2)
    labels = label(heat_map)
    #print("found {0} cars!".format(labels[1]))
    
    #draw bounding boxes around found cars
    if (labels[1]>0):
        bboxes = DrawBBoxes(img, labels, color=(0,255,0), draw=True)
    
    return Centers(bboxes)
