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
from scipy.ndimage import gaussian_laplace   # <-- Added for Gaussian Laplace filtering

def preprocess_enhance_plate(thresh, SHOW=0):
    #Morphological post-processing for binary license plate mask.
    close_size = (3, 3)
    open_size  = (2, 2)
    iterations = 1

    # Closing to fill small gaps in characters
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, close_size)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=iterations)

    # Opening to remove small isolated noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, open_size)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=iterations)

    # Slight blur to smooth contours. BUT THIS DIDNT WORKED
    #thresh = cv2.GaussianBlur(thresh, (3, 3), 0)

    if SHOW:
        cv2.imshow("Morphological Postprocessing", imutils.resize(thresh, width=400))

    return thresh



def detectCharacterCandidates(image, reg, SHOW=0, PREPROCESSING=1):
    # apply a 4-point transform to extract the license plate
    plate = perspective.four_point_transform(image, reg)
    plate = imutils.resize(plate, width=400)

    if (SHOW):
        cv2.imshow("Perspective Transform", plate)

    V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]        
    thresh = cv2.adaptiveThreshold(V, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    if (SHOW):
        cv2.imshow("Adaptative Threshold", thresh)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rectKernel, iterations=2)

    #CALLING THE PREPROCESSING
    if PREPROCESSING:
        thresh = preprocess_enhance_plate(thresh, SHOW)

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
    #  Detect blobs with gaussian_laplace function from scipy
    # -------------------------------------------------------------------------
    if blobs.size > 0:
        # Estimate the appropriate isotropic Gaussian sigma for LoG detection
        median_r = np.median(blobs[:, 2])
        sigma = median_r / (2 * np.sqrt(2))
        threshold_scipy = 0.04  # threshold for LoG peaks

        # Prepare normalized grayscale image
        gray = norm.astype(np.float32)

        # Negative sign is used because characters are white on black
        # blobs become positive peaks
        log_response = -gaussian_laplace(gray, sigma=sigma)

        # Threshold the LoG response to find candidates
        candidates_scipy = (log_response > threshold_scipy).astype(np.uint8) * 255
    
    else:
        candidates_scipy = np.zeros_like(thresh, dtype=np.uint8)
        log_response = np.zeros_like(norm, dtype=np.float32)

    if (SHOW):
        print("START DIMENSIONAL ANALYSIS")
    
    DimY, DimX = thresh.shape[:2]
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    MycharCandidates = np.zeros(thresh.shape, dtype="uint8")
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        NotouchBorder = x != 0 and y != 0 and x + w != DimX and y + h != DimY
        
        if (NotouchBorder):
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
     # and the Gaussian Laplacian candidates and response
    return plate, thresh, MycharCandidates, blob_mask, candidates_scipy, log_response


def postprocess_character_candidates(candidate_map, plate, SHOW=0, area_limits=(200, 2500)):
    """
    Refines the raw candidate map (from LoG or contour detection),
    removes noise, extracts bounding boxes and crops each detected character.
    Returns:
    --------
    char_boxes : list of tuples
        Each box as (x, y, w, h)
    char_crops : list of np.ndarray
        Cropped character images from the plate
    clean_mask : np.ndarray
        Binary mask after morphological refinement
    """
    # make sure we have a binary mask
    if candidate_map.dtype != np.uint8:
        candidate_map = ((candidate_map > 0) * 255).astype(np.uint8)

    #Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(candidate_map, cv2.MORPH_OPEN, kernel, iterations=1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    #Find connected components (contours) 
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_plate, w_plate = plate.shape[:2]
    char_boxes, char_crops = [], []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        #simple geometric filtering 
        if area_limits[0] < area < area_limits[1]:
            aspect = w / float(h)
            if 0.2 < aspect < 1.5:  # characters are usually tall
                char_boxes.append((x, y, w, h))
                char_crops.append(plate[y:y+h, x:x+w])

    #sort boxes from left to right 
    char_boxes = sorted(char_boxes, key=lambda b: b[0])
    char_crops = [crop for _, crop in sorted(zip([b[0] for b in char_boxes], char_crops))]

    return char_boxes, char_crops, clean_mask


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
        plate, thresh, candidates, blob_mask, candidates_scipy, log_response = detectCharacterCandidates(image, reg, SHOW=0, PREPROCESSING=1)
        """
        # Visualize intermediate and final outputs
        plt.figure(figsize=(15, 6))

        plt.subplot(2, 3, 1)
        plt.title("Plate Region")
        plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.title("Thresholded (Adaptive)")
        plt.imshow(thresh, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.title("Contour Candidates")
        plt.imshow(candidates, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.title("LoG Blob Detection (skimage)")
        plt.imshow(blob_mask, cmap="gray")
        plt.axis("off")
        
        plt.subplot(2, 3, 5)
        plt.title("Gaussian Laplacian Response (scipy)")
        plt.imshow(log_response, cmap='gray')
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.title("Thresholded LoG Candidates (scipy)")
        plt.imshow(candidates_scipy, cmap='gray')
        plt.axis("off")

        # Adjust layout and title position
        plt.suptitle(f"Character Segmentation: {img_name}", fontsize=14, y=0.85)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
        """

        #POST PROCESSING
        char_boxes, char_crops, refined_mask = postprocess_character_candidates(candidates, plate, SHOW=1)
        print(f"Detected {len(char_boxes)} character regions.")

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

        # We could also save cropped characters
        # for i, ch in enumerate(char_crops):
        #     cv2.imwrite(f"./char_crops/char_{i}.png", ch)
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
        if idx >= 20:
            break
