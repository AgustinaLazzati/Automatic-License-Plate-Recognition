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
import random

SHOW = 1
minPlateW = 100
minPlateH = 30

def detectPlates(image):
    # grab the dimensions of the image
    imHeight, imWidth = image.shape[:2]

    # if the width is greater than 640 pixels, then resize the image
    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)
        
    # initialize the rectangular and square kernels to be applied to the image,
    # then initialize the list of license plate regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    regions = []

    # convert the image to grayscale, and apply the blackhat operation
    # to emphasize narrow regions with dark gray level on a light background
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, squareKernel, iterations=3)

    if SHOW:
        plt.figure()
        plt.imshow(blackhat, cmap="gray")
        plt.title("Black Top Hat")
        plt.show()

    # compute the gradient in the x-direction
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    if SHOW:
        plt.figure()
        plt.imshow(gradX, cmap="gray")
        plt.title("Gradient X")
        plt.show()

    # smooth the gradient image using Gaussian blur
    gradX = cv2.GaussianBlur(gradX, (7, 7), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel, iterations=2)

    if SHOW:
        plt.figure()
        plt.imshow(gradX, cmap="gray")
        plt.title("Gaussian Gx")
        plt.show()

    # threshold the gradient image (40% of max value)
    ThrValue = 0.40 * np.max(gradX)
    ThrGradX = cv2.threshold(gradX, ThrValue, 255, cv2.THRESH_BINARY)[1]

    if SHOW:
        plt.figure()
        plt.imshow(ThrGradX, cmap="gray")
        plt.title("Threshold Gx")
        plt.show()

    # apply morphological operations to clean up the thresholded image
    thresh = cv2.morphologyEx(ThrGradX, cv2.MORPH_OPEN, squareKernel, iterations=4)
    thresh = cv2.dilate(thresh, rectKernel, iterations=2)

    if SHOW:
        plt.figure()
        plt.imshow(thresh, cmap="gray")
        plt.title("Possible license plates")
        plt.show()

    # find contours in the thresholded image
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours to filter based on size and aspect ratio
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspectRatio = w / float(h)

        if SHOW:
            print("BLOB ANALYSIS ->", x, y, w, h, aspectRatio, area)

        # ensure the contour is not touching the image border
        NotouchBorder = x != 0 and y != 0 and x + w != imWidth and y + h != imHeight

        # conditions to keep a region as a potential license plate
        keepArea = area > 3400 and area < 8000
        keepWidth = w > minPlateW and w <= 250
        keepHeight = h > minPlateH and h <= 60
        keepAspectRatio = 2.5 < w / h < 7

        if all((NotouchBorder, keepAspectRatio, keepWidth, keepHeight, keepArea)):
            # compute the rotated bounding box of the region
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            regions.append(box)

            if SHOW:
                print("REGION BOX ACCEPTED ->", box)

    # return both the detected regions and the processed (possibly resized) image
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

