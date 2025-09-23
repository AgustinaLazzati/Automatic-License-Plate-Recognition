# import the necessary packages
import numpy as np
import cv2
import glob
import os
from imutils import perspective
from matplotlib import pyplot as plt

from ultralytics import YOLO

#### EXP-SET UP
# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
DataDir=r'D:\Teaching\Grau\GrauIA\V&L\Challenges\Matricules\Dades\real_plates'
Views=['Frontal','Lateral']



# Load YOLO model
model = YOLO("YOLO.pt") # Trained on COCO DataSet
modelclasses=np.array(list(model.names.values()))
model.device # By default model is in GPU device: model=model.to('cpu') for execution in CPU


#### COMPUTE PROPERTIES FOR EACH VIEW
yoloConf={}
yoloObj={}
for View in Views:
    
    ImageFiles=sorted(glob.glob(os.path.join(DataDir,View,'*.jpg')))
    yoloConf[View]=[]
    yoloObj[View]=[]

    # loop over the images
    for imagePath in ImageFiles:
        # load the image
        image = cv2.imread(imagePath)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(img_rgb) 
        obj=results[0].boxes.cls.cpu().numpy().astype(int)
        yoloObj[View].append(obj)
        yoloConf[View].append(results[0].boxes.conf.cpu().numpy())
        # Show results
        results[0].show()

####  EXPLORE OBJECT DISTRIBUTION FOR EACH VIEW USING HISTOGRAMS AND BOXPLOTS
