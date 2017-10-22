import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from scipy.ndimage.measurements import label

from collections import defaultdict
from io import StringIO

#generate heatmap of classified windows and apply threshold
def HeatMap(img, ROI_Ymin, ROI_Ymax, ROI_Xmin, ROI_Xmax, windows_with_car, threshold):

    heat_map = np.empty_like(img)

    for window in windows_with_car:
        Ymin = int(window[0] * (ROI_Ymax-ROI_Ymin) + ROI_Ymin)
        Xmin = int(window[1] * (ROI_Xmax-ROI_Xmin) + ROI_Xmin)
        Ymax = int(window[2] * (ROI_Ymax-ROI_Ymin) + ROI_Ymin)
        Xmax = int(window[3] * (ROI_Xmax-ROI_Xmin) + ROI_Xmin)
        heat_map[Ymin:Ymax, Xmin:Xmax,2] += 1

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

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

cap = cv2.VideoCapture('project_video.mp4')

frame_cnt = 0

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while(True):

            frame_cnt += 1

            ret, img = cap.read()

            #stop at end of file
            if img is None:
                break

            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #define ROI
            Ymin = 360
            Ymax = 650
            Xmin = 0
            Xmax = img.shape[1]
            img_ROI = img_RGB[Ymin:Ymax,Xmin:Xmax,:]
            
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(img_ROI, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
            #detect CARS
            #class should be 3 (car) and score should be higher than threshold
            idx = np.intersect1d(np.where(classes[0]==3),np.where(scores[0]>0.10))
            boxes = boxes[0,idx]

            #merge duplicates with heatmap
            heat_map = HeatMap(img, Ymin, Ymax, Xmin, Xmax, boxes, threshold = 0)
            labels = label(heat_map)
            #draw bounding boxes around found cars
            if (labels[1]>0):
                bboxes = DrawBBoxes(img, labels, color=(0,255,0), draw=True)

            cv2.imshow('Video',img)
            cv2.imwrite('video_images/frame'+(str(frame_cnt)).zfill(4)+'.jpg',img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
cap.release()
cv2.destroyAllWindows()
