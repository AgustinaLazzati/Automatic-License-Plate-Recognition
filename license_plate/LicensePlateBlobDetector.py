# -*- coding: utf-8 -*-
# This file is dedicated to solve Exercise 2 BLOB DETECTORS. 
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
from skimage.feature import blob_log
from skimage.color import rgb2gray
import os


def detectCharacterCandidates(image, reg, SHOW=0):
    # apply a 4-point transform to extract the license plate
    plate = perspective.four_point_transform(image, reg)
    plate = imutils.resize(plate, width=400)

    if (SHOW):
        cv2.imshow("Perspective Transform (LoG)", plate)
    
    # extract the Value component from the HSV color space and apply adaptive thresholding
    # to reveal the characters on the license plate
    V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]        
    thresh = cv2.adaptiveThreshold(V, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    if (SHOW):
        cv2.imshow("Adaptative Threshold (LoG)", thresh)

    # Structuring Element  rectangular shape (width 1 hight 3)  s
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    # clossing follow by an opening to fill holes and join regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rectKernel, iterations=2)

    # resize the license plate region 
    thresh = imutils.resize(thresh, width=400)

    if (SHOW):
        print("START DIMENSIONAL ANALYSIS (LoG)")

    # --- LoG Blob Detection instead of contour analysis ---
    gray_plate = rgb2gray(plate)
    blobs = blob_log(gray_plate, min_sigma=2, max_sigma=10, num_sigma=10, threshold=0.05)

    # Compute radio from sigma values
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

    # create blank image to store blob candidates
    MycharCandidates = np.zeros(thresh.shape, dtype="uint8")

    for y, x, r in blobs:
        cv2.circle(MycharCandidates, (int(x), int(y)), int(r), 255, -1)

    if (SHOW):
        print("Number of detected blobs:", len(blobs))
        # Show blobs over grayscale plate
        vis_plate = cv2.cvtColor((gray_plate * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
        for y, x, r in blobs:
            cv2.circle(vis_plate, (int(x), int(y)), int(r), (0, 0, 255), 1)
        cv2.imshow("Detected Blobs (LoG)", vis_plate)
        print("END DIMENSIONAL ANALYSIS (LoG)")



    # return the license plate region object containing the license plate, the thresholded
    # license plate, and the character candidates
    return plate, thresh, MycharCandidates


# ---------------------------------------------------------------
# MAIN FUNCTION FOR EXERCISE 2 (LoG)
if __name__ == "__main__":
    # Define path (adjust if necessary)
    base_path = "./data/cropped_real_plates/Frontal"  # or "Lateral"

    # Load region data
    npz_path = os.path.join(base_path, "PlateRegions.npz")
    data = np.load(npz_path, allow_pickle=True)

    print("Variables in PlateRegions.npz:", data.files)
    regionsImCropped = data["regionsImCropped"]
    ImID = data["imID"]

    # Loop over some plates
    for idx, img_name in enumerate(ImID):
        img_filename = f"{img_name}_MLPlate0.png"
        img_path = os.path.join(base_path, img_filename)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_filename}")
            continue

        image = cv2.imread(img_path)
        reg = np.array(regionsImCropped[idx], dtype="float32")

        # Run the LoG-based segmentation
        plate, thresh, candidates = detectCharacterCandidates(image, reg, SHOW=1)

        # Visualize intermediate and final outputs
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 3, 1)
        plt.title("Plate Region")
        plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Thresholded (Adaptive)")
        plt.imshow(thresh, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Character Candidates (LoG)")
        plt.imshow(candidates, cmap="gray")
        plt.axis("off")

        # Adjust layout and title position
        plt.suptitle(f"Character Segmentation using LoG: {img_name}", fontsize=14, y=0.65)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # reduce top margin for less white space
        plt.show()

        # stop after a few images
        if idx >= 2:
            break
