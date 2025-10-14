# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:52:26 2025

Example of WaterShed Segmentation

Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres"
__license__ = "GPL"
__email__ = "debora,gtorres@cvc.uab.es"

"""
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from imutils import perspective

#img_path = '/home/tomiock/uni2025/vision/license/data/real_plates/Frontal/3340JMF.jpg'
# Define path AS BEFORE
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
    img_fullpath = os.path.join(base_path, img_filename)

    if not os.path.exists(img_fullpath):
        print(f"Image not found: {img_filename}")
        continue

    img = cv2.imread(img_fullpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reg = np.array(regionsImCropped[idx], dtype="float32")
    

    # MORPHOLOGICAL PRE-PROCESSING
    # -------------------------------------------------------------------------
    # apply a 4-point transform to extract the license plate
    img = perspective.four_point_transform(img, reg)
    img = imutils.resize(img, width=400)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img)

    # Black-hat transformation (enhance dark letters on bright background)
    kernel_blackhat = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    img_blackhat = cv2.morphologyEx(img_eq, cv2.MORPH_BLACKHAT, kernel_blackhat)

    # Smoothing
    img_prep = cv2.GaussianBlur(img_blackhat, (3,3), 0)
    #-------------------------------------------------------------------------

    # Supongamos que 'img' es la imagen original en gris
    # 1. Umbral binario ---> THIS IS THE THRESHOLDING 
    _, binary = cv2.threshold(img_prep, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 2. Ruido y separación de letras cercanas 
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Distancia transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(opening, sure_fg)

    # 4. Marcadores
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0

    # 5. Watershed
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    img_color[markers == -1] = [0,0,255]  # Bordes en rojo 

    # 6. Visualización
    cv2.imshow("Watershed", img_color)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    #Ploting all steps in subplots
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    images = [
        (img, "Original Gray"),
        (binary, "Binary (Otsu inv)"),
        (opening, "Morphological Opening"),
        (dist_transform / dist_transform.max(), "Distance Transform"),
        (sure_fg, "Sure Foreground"),
        (unknown, "Unknown Region"),
        (markers, "Markers"),
        (img_color[..., ::-1], "Watershed Result")  # BGR -> RGB for Matplotlib
    ]

    for ax, (img_disp, title) in zip(axes, images):
        if len(img_disp.shape) == 2:  # grayscale
            ax.imshow(img_disp, cmap='gray')
        else:  # color
            ax.imshow(img_disp)
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle(f"WaterShed Segmentation: {img_name}", fontsize=14, y=0.95)
    plt.tight_layout()
    plt.show()

    # stop after a few images
    if idx >= 1:
        break