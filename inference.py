
import numpy as np
import os
import sys
import tensorflow as tf
# from matplotlib import pyplot as plt
from PIL import Image
import cv2
import Main
import time

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'model'
PATH_TO_CKPT = MODEL_NAME + 'frozen_inference_graph.pb'
PATH_TO_LABELS =  './model/test.pbtxt'
NUM_CLASSES = 1


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
  sess = tf.Session(graph=detection_graph)


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = './tn/'
IMAGE_SIZE = (12, 8)
TEST_DHARUN=os.path.join('numplate')
count = 0
imgpath="./data/"

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

##########Inference on collection of test images#############################

# for image_path in os.listdir(PATH_TO_TEST_IMAGES_DIR):
#   print(image_path)
#   image=cv2.imread(PATH_TO_TEST_IMAGES_DIR+'/'+image_path)
#   image=cv2.resize(image,(640,480))
#   image_np_expanded = np.expand_dims(image, axis=0)
#   (boxes, scores, classes, num) = sess.run(
#       [detection_boxes, detection_scores, detection_classes, num_detections],
#       feed_dict={image_tensor: image_np_expanded})
#   ymin = boxes[0,0,0]
#   xmin = boxes[0,0,1]
#   ymax = boxes[0,0,2]
#   xmax = boxes[0,0,3]
#   height,width,color=image.shape
#   vis_util.draw_bounding_box_on_image_array(image,ymin,xmin,ymax,xmax,color="red")
#   scaled_ymin=ymin*height
#   scaled_ymax=ymax*height
#   scaled_xmin=xmin*width
#   scaled_xmax=xmax*width
#   cropped_image=image[int(scaled_ymin):int(scaled_ymax),int(scaled_xmin):int(scaled_xmax)]
#   pred,_=Main.main(cropped_image)
#   # center_pt=scaled_xmax-scaled_xmin
#   cv2.putText(image,pred,(int(scaled_xmin)+30,int(scaled_ymin)),cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 255), 2)
#   cv2.imshow('detected',image)
#   cv2.waitKey(0)


##################Inference on webcam######################

cap=cv2.VideoCapture("./test/test5.mp4")
j=0
result = {}
predicted=""
scaled_ymin=0
scaled_xmin=0
scaled_ymax=0
scaled_xmax=0
# result_imag = {}
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(640,480))
    j+=1
    startTime = time.time()
    image_np_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = sess.run(
      [detection_boxes, detection_scores, detection_classes, num_detections],
      feed_dict={image_tensor: image_np_expanded})
    # ymin = boxes[0,0,0]
    # xmin = boxes[0,0,1]
    # ymax = boxes[0,0,2]
    # xmax = boxes[0,0,3]
    vis_util.visualize_boxes_and_labels_on_image_array(
          frame,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=2)
    # vis_util.draw_bounding_box_on_image_array(frame,ymin,xmin,ymax,xmax,color="red")
    if j%30==0:
        for i in range(0, int(num[0])):
            scores_num=float(format(scores[0][i],'.3f'))
            if scores_num >= 0.59:
                ymin, xmin, ymax, xmax = boxes[0][i]
                height,width,color=frame.shape
                
                scaled_ymin=ymin*height
                scaled_ymax=ymax*height
                scaled_xmin=xmin*width
                scaled_xmax=xmax*width
                cropped_image=frame[int(scaled_ymin):int(scaled_ymax),int(scaled_xmin):int(scaled_xmax)]
                pred,img=Main.main(cropped_image)
                if pred in result.keys():
                    result[pred] = result[pred] + 1
                    count+=1
                elif pred != ' ':
                    result[pred] = 1
                    count+=1
                    # result_imag[pred] = img
                #endTime = datetime.now()
                if count==5:
                    endTime = time.time()
                    l = {x: y for y, x in result.items()}
                    r = list(sorted(l.keys()))
                    index = r[len(r) - 1]
                    plate = l[index]
                    # img = result_imag[plate]
                    executionTime = "{0:.2f}".format(endTime - startTime)
                    print('The name plate is :', plate, ' frequency is: ', result[plate])
                    # try:
                    #     Image.fromarray(img).show()
                    # except:
                    #     print("Problem in displaying license plate")
                    print('execution time is : ' + executionTime)
                    count=0
                    predicted=plate
                    result = {}
        j=0
    cv2.putText(frame,str(predicted),(int(scaled_xmin)+30,int(scaled_ymin)-20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
    cv2.imshow("Detected",frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break     