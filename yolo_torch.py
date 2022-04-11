import torch
import cv2
from PIL import Image
import numpy as np
#import pandas


#Inference Settings
#model.conf = 0.25  # NMS confidence threshold
#      iou = 0.45  # NMS IoU threshold
#      agnostic = False  # NMS class-agnostic
#      multi_label = False  # NMS multiple labels per box
#      classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
#      max_det = 1000  # maximum number of detections per image
#      amp = False  # Automatic Mixed Precision (AMP) inference
#results = model(imgs, size=320)  # custom inference size


class YoloDetector(object):
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.classes=0 #running only person detection
        self.detection=np.array([0, 0, 0, 0])


    def bbox_format(self):
        #detection format xmin, ymin, xmax,ymax, conf, class, 'person'
        #bbox format xcenter, ycenter, width, height
        if(self.detection.shape[0]==1):
            self.detection=np.squeeze(self.detection)
            xmin=self.detection[0]
            ymin=self.detection[1]
            xmax=self.detection[2]
            ymax=self.detection[3]
            x_center=(xmin+xmax)/2
            y_center=(ymin+ymax)/2
            width=xmax-xmin
            height=ymax-ymin
            arr=[x_center, y_center, width, height]
            bbox=np.array(arr)
            bbox=np.expand_dims(bbox, 0)
            return bbox
        else:
            bbox_list=[]
            for i in range(self.detection.shape[0]):
                xmin=self.detection[i][0]
                ymin=self.detection[i][1]
                xmax=self.detection[i][2]
                ymax=self.detection[i][3]
                x_center=(xmin+xmax)/2
                y_center=(ymin+ymax)/2
                width=xmax-xmin
                height=ymax-ymin
                arr=[x_center, y_center, width, height]
                bbox_unit=np.array(arr)
                bbox_list.append(bbox_unit)
                #bbox=np.append(bbox_unit, axis=0)
            bbox=np.vstack(bbox_list)
            return bbox

            

    def best_detection(self):
        N=self.detection.shape[0]
        if(N != 1):
            print("multiple persons detected")
            #extracting the detection with max confidence
            idx=np.argmax(self.detection[range(N),4])
            self.detection=self.detection[idx]
        else: #1 detection
            self.detection=np.squeeze(self.detection)


    def predict(self, image, thresh=0.01):
        #threshold for confidence detection
        
        # Inference
        results = self.model(image) #might need to specify the size

        #results.xyxy: [xmin, ymin, xmax, ymax, conf, class]
        detect_pandas=results.pandas().xyxy

        self.detection=np.array(detect_pandas)
        #print("shape of the detection: ", self.detection.shape)
        #print("detection: ",self.detection)

        if (self.detection.shape[1]!=0):
            #print("DETECTED SOMETHING !!!")
            #save resuts
            #results.save()
            
            #use np.squeeze to remove 0 dim from the tensor
            self.detection=np.squeeze(self.detection,axis=0) 

            #class function to decide which detection to keep
            self.best_detection()
            if(self.detection[4]>thresh):
                label=True
            #modify the format of detection for bbox
            bbox=self.bbox_format()
            return bbox, label
        return [0.0, 0.0, 0.0, 0.0],False


    def predict_multiple(self, image, thresh=0.01):
      #threshold for confidence detection
      
      # Inference
      results = self.model(image) #might need to specify the size

      #results.xyxy: [xmin, ymin, xmax, ymax, conf, class]
      detect_pandas=results.pandas().xyxy

      self.detection=np.array(detect_pandas)
      #print("shape of the detection: ", self.detection.shape)
      #print("detection: ",self.detection)

      if (self.detection.shape[1]!=0):
          #print("DETECTED SOMETHING !!!")
          #save resuts
          #results.save()
          
          #use np.squeeze to remove 0 dim from the tensor
          self.detection=np.squeeze(self.detection,axis=0) 

          #modify the format of detection for bbox
          bbox=self.bbox_format()
          return bbox, True
      return [0.0, 0.0, 0.0, 0.0],False