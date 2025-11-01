### THIS FILE IS CREATED TO SIMPLY UNDERSTAND THE PROCESS OF THE MANUAL PLATE DETECTION. 
### EXERCISE 3 SESSION 2. ---- when finished it and add it to the report it can be removed. 

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:45:11 2025

@author: debora
"""

# import the necessary packages
from collections import namedtuple
import skimage
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import os
import argparse

SHOW = 1

# Plate size thresholds (smaller min size for distant plates)
minPlateW = 60   # previously 100
minPlateH = 20   # previously 30

def detectPlates(image):
    imHeight, imWidth = image.shape[:2]

    # Resize image if too large for faster processing
    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)

    # Slightly larger kernels to capture faint/blurred edges
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))  # was (15,5)
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # was (3,3)

    regions = []

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Step 1: Blackhat
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, squareKernel, iterations=2)  # reduce iterations for distant, faint plates
    if SHOW:
        plt.figure()
        plt.imshow(blackhat, cmap="gray")
        plt.title("Black Top Hat")
        plt.show()

    # Step 2: Gradient X
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    if SHOW:
        plt.figure()
        plt.imshow(gradX, cmap="gray")
        plt.title("Gradient X")
        plt.show()

    # Step 3: Blur + Close
    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)  # slightly smaller blur for distant plates
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel, iterations=1)  # fewer iterations
    if SHOW:
        plt.figure()
        plt.imshow(gradX, cmap="gray")
        plt.title("Blur + Close")
        plt.show()

    # Step 4: Threshold
    ThrValue = 0.30 * np.max(gradX)  # was 0.40
    ThrGradX = cv2.threshold(gradX, ThrValue, 255, cv2.THRESH_BINARY)[1]
    if SHOW:
        plt.figure()
        plt.imshow(ThrGradX, cmap="gray")
        plt.title("Threshold")
        plt.show()

    # Step 5: Morphology
    thresh = cv2.morphologyEx(ThrGradX, cv2.MORPH_OPEN, squareKernel, iterations=2)  # was 4
    thresh = cv2.dilate(thresh, rectKernel, iterations=1)  # was 2
    if SHOW:
        plt.figure()
        plt.imshow(thresh, cmap="gray")
        plt.title("Final Morphology")
        plt.show()

    # Step 6: Contour Detection
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspectRatio = w / float(h)

        NotouchBorder = x != 0 and y != 0 and x + w != imWidth and y + h != imHeight

        # Relax area and aspect ratio for distant plates
        keepArea = area > 2000 and area < 10000  # was 3400-8000
        keepWidth = w > minPlateW and w <= 300   # was <= 250
        keepHeight = h > minPlateH and h <= 70   # was <= 60
        keepAspectRatio = 2.0 < aspectRatio < 8  # was 2.5-7

        if all((NotouchBorder, keepAspectRatio, keepWidth, keepHeight, keepArea)):
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            regions.append(box)
            if SHOW:
                print("REGION BOX ACCEPTED->", box)

    return regions, image



def main():
   # Relative path to the folder with the plate images
    datapath = os.path.join(".", "data", "Frontal")
    datapath = os.path.abspath(datapath)  # normalize to absolute path

    if not os.path.isdir(datapath):
        print(f"Error: Directory '{datapath}' not found.")
        return

    # List all image files
    image_files = [f for f in os.listdir(datapath) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print(f"Error: No images found in '{datapath}'.")
        return

    # Target plate filename for report
    target_plate = "8727JTC.jpg"  # change extension if needed
    if target_plate not in image_files:
        print(f"Error: '{target_plate}' not found in '{datapath}'.")
        return

    # Full path to the image
    image_path = os.path.abspath(os.path.join(datapath, target_plate)).replace("\\", "/")
    print(f"Processing image: {image_path}")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at '{image_path}'.")
        return

    detected_plates, processed_image = detectPlates(image)

    output_image = processed_image.copy()
    for box in detected_plates:
        box = np.intp(box)
        cv2.drawContours(output_image, [box], -1, (0, 255, 0), 2)

    if SHOW:
        plt.figure()
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title("Detected License Plates")
        plt.show()

    print(f"Found {len(detected_plates)} potential license plates.")


if __name__ == "__main__":
    main()

