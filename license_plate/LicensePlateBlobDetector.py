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
import os
from skimage.feature import blob_log  # <-- Added for LoG blob detection
from skimage.color import rgb2gray    # <-- For grayscale conversion


def detectCharacterCandidates(image, reg, SHOW=1):
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

    # Structuring Element  rectangular shape (width 1 hight 3)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    # clossing follow by an opening to fill holes and join regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rectKernel, iterations=2)

    # resize the license plate region 
    thresh = imutils.resize(thresh, width=400)

    # -------------------------------------------------------------------------
    #  We add Laplacian of Gaussian (LoG) Blob Detection
    # -------------------------------------------------------------------------
    norm = thresh.astype(np.float32) / 255.0  # normalize to [0,1]
    blobs = blob_log(norm, max_sigma=10, num_sigma=10, threshold=0.1)

    # Convert sigma to radius
    if blobs.size > 0:
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

    # Create blob mask
    blob_mask = np.zeros_like(thresh, dtype="uint8")
    for (y, x, r) in blobs:
        cv2.circle(blob_mask, (int(x), int(y)), int(r), 255, -1)

    # visualization of blobs over the grayscale plate ---
    if SHOW:
        print("Number of detected blobs:", len(blobs))
        gray_plate = rgb2gray(plate)
        vis_plate = cv2.cvtColor((gray_plate * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
        for y, x, r in blobs:
            cv2.circle(vis_plate, (int(x), int(y)), int(r), (0, 0, 255), 1)
        cv2.imshow("Detected Blobs (LoG)", vis_plate)
    # -------------------------------------------------------------------------

    if (SHOW):
        print("START DIMENSIONAL ANALYSIS")
    
    DimY, DimX = thresh.shape[:2]
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    MycharCandidates = np.zeros(thresh.shape, dtype="uint8")
    for c in cnts:
        # grab the bounding box associated with the contour and compute the area and
        # aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)

        # condition of not touching the border of the region
        NotouchBorder = x != 0 and y != 0 and x + w != DimX and y + h != DimY
        
        if (NotouchBorder):
            # hight ratio of the blob numbers are blobs with a hight near the DimY
            hW = (h / float(DimY))
            area = cv2.contourArea(c)
            if (SHOW):
                print("AREA: ", area, " ASPECT: ", hW)
            heightRatio = 0.5 < hW < 0.9
            if (area > 300) and (heightRatio):
                hull = cv2.convexHull(c)
                cv2.drawContours(MycharCandidates, [hull], -1, 255, -1)
    if (SHOW):            
        print("END DIMENSIONAL ANALYSIS")

    # return the license plate region, the thresholded image, the contour-based mask, and the LoG blob mask
    return plate, thresh, MycharCandidates, blob_mask


# MAIN FUNCTION FOR EXERCISE 1
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Define path (adjust if necessary)
    base_path = "./data/cropped_real_plates/Lateral"  # or "Lateral"

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

        # Run the segmentation
        plate, thresh, candidates, blob_mask = detectCharacterCandidates(image, reg, SHOW=1)

        # Visualize intermediate and final outputs
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 4, 1)
        plt.title("Plate Region")
        plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Thresholded (Adaptive)")
        plt.imshow(thresh, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Contour Candidates")
        plt.imshow(candidates, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title("LoG Blob Detection")
        plt.imshow(blob_mask, cmap="gray")
        plt.axis("off")

        # Adjust layout and title position
        plt.suptitle(f"Character Segmentation: {img_name}", fontsize=14, y=0.65)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

        # stop after a few images
        if idx >= 1:
            break
