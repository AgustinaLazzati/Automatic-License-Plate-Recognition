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
from LicensePlateBlobDetector import postprocess_character_candidates   #THIS IS THE FUNCTION TO MAKE THE BOUNDING BOXES. 


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


# MAIN FUNCTION FOR EXERCISE 1
# ---------------------------------------------------------------
if __name__ == "__main__":
    SHOW=0
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

        # Run the segmentation
        plate, thresh, candidates = detectCharacterCandidates(image, reg, SHOW=0)
        
        if SHOW:
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
            plt.title("Character Candidates")
            plt.imshow(candidates, cmap="gray")
            plt.axis("off")

            # Adjust layout and title position
            plt.suptitle(f"Character Segmentation: {img_name}", fontsize=14, y=0.65)
            plt.tight_layout(rect=[0, 0, 1, 0.93])  # reduce top margin for less white space
            plt.show()


        # ADDING THE POSTPROCESSING CREATED IN BLOBDETECTOR PIPELINE
        char_boxes, char_crops, refined_mask = postprocess_character_candidates(candidates, plate, SHOW=0)
        print(f"Detected {len(char_boxes)} character regions.")



        if SHOW:
            plt.figure(figsize=(10, 5))

            # Left: postprocessed mask
            plt.subplot(1, 2, 1)
            plt.imshow(refined_mask, cmap="gray")
            plt.title("Postprocessed Mask")
            plt.axis("off")

            # Right: plate with bounding boxes
            plt.subplot(1, 2, 2)
            vis_plate = plate.copy()
            for (x, y, w, h) in char_boxes:
                cv2.rectangle(vis_plate, (x, y), (x + w, y + h), (0, 255, 0), 1)

            plt.imshow(cv2.cvtColor(vis_plate, cv2.COLOR_BGR2RGB))
            plt.title("Detected Character Boxes")
            plt.axis("off")

            plt.suptitle(f"Postprocessing Results: {img_name}", fontsize=13,  y=0.85)
            plt.tight_layout()
            plt.show()

        # ----------------------------------------------------------------------
        # SAVE CROPPED CHARACTERS
        # ----------------------------------------------------------------------
        # Detect if we are processing frontal or lateral dataset from the path
        view_type = "Frontal" if "Frontal" in base_path else "Lateral"

        # Root folder for character crops
        save_root = os.path.join("data", "char_crops", view_type)
        os.makedirs(save_root, exist_ok=True)

        # Create a subfolder for each plate
        plate_folder = os.path.join(save_root, img_name)
        os.makedirs(plate_folder, exist_ok=True)

        print(f"[{view_type}] Saving {len(char_crops)} cropped characters to {plate_folder}")

        for i, ch in enumerate(char_crops):
            # Convert to grayscale before saving
            gray_ch = cv2.cvtColor(ch, cv2.COLOR_BGR2GRAY)
            
            # Save cropped character
            save_path = os.path.join(plate_folder, f"char_{i:02d}.png")
            cv2.imwrite(save_path, gray_ch)

        # stop after a few images
        if idx >= 30:
            break