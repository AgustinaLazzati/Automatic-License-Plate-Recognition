# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:45:11 2025
Universitat Autonoma de Barcelona

__author__ = Xavier Roca
__license__ = "GPL"
__email__ = "xavier.roca@cvc.uab.es"
"""
# import the necessary packages
#from collections import namedtuple
# from skimage.filters import threshold_local
# from skimage import segmentation
# from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt




def detectCharacterCandidates(image, reg, SHOW=0):
    # apply a 4-point transform to extract the license plate
    plate = perspective.four_point_transform(image, reg)
    plate = imutils.resize(plate, width=400)

    if (SHOW):
        cv2.imshow("Perspective Transform", plate)
    
    # extract the Value component from the HSV color space and apply adaptive thresholding
    # to reveal the characters on the license plate
    V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]        
    thresh = cv2.adaptiveThreshold(V, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    if (SHOW):
        cv2.imshow("Adaptative Threshold", thresh)

    # Structuring Element  rectangular shape (width 1 hight 3)  s
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    # clossing follow by an opening to fill holes and join regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel,iterations = 2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rectKernel,iterations = 2)

    # resize the license plate region 
    thresh = imutils.resize(thresh, width=400)

    if (SHOW):
        print("START DIMENSIONAL ANALYSIS")
    
    DimY, DimX = thresh.shape[:2]
    (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    MycharCandidates = np.zeros(thresh.shape, dtype="uint8")
    for c in cnts:
        # grab the bounding box associated with the contour and compute the area and
        # aspect ratio
        (x,y,w, h) = cv2.boundingRect(c)

        # condition of not touching the border of the region
        NotouchBorder = x!=0 and y!=0 and x+w!=DimX and y+h!=DimY
        
        if (NotouchBorder):
            

            # hight ratio of the blob numbers are blobs with a hight near the DimY
            hW=(h / float(DimY))
            area = cv2.contourArea(c)
            if (SHOW):
                print("AREA: ",area," ASPECT: ",hW)
            heightRatio = 0.5 < hW <0.9
            if (area>300) and (heightRatio):
                
                hull = cv2.convexHull(c)
                cv2.drawContours(MycharCandidates, [hull], -1, 255, -1)
    if (SHOW):            
        print("END DIMENSIONAL ANALYSIS")

    

    # return the license plate region object containing the license plate, the thresholded
    # license plate, and the character candidates
    return plate, thresh, MycharCandidates