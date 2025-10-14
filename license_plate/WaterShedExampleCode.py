# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:52:26 2025

Example of WaterShed Segmentation

Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres"
__license__ = "GPL"
__email__ = "debora,gtorres@cvc.uab.es"

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'data/Frontal/3340JMF.jpg'   #'/home/tomiock/uni2025/vision/license/data/real_plates/Frontal/3340JMF.jpg'
img= cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Supongamos que 'img' es la imagen original en gris
# 1. Umbral binario
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

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
#watershed_result = img_color.copy()
#watershed_result[markers == -1] = [0, 0, 255]  # red borders

# 6. Visualización
cv2.imshow("Watershed", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Ploting all steps in subplots
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()

images = [
    (img, "Original Gray"),
    (binary, "Binary (Otsu inv)"),
    (opening, "Morphological Opening"),
    (dist_transform / dist_transform.max(), "Distance Transform"),
    (sure_fg, "Sure Foreground"),
    (unknown, "Unknown Region"),
    (markers, "Markers")
    #(watershed_result[..., ::-1], "Watershed Result")  # BGR -> RGB for Matplotlib
]

for ax, (img_disp, title) in zip(axes, images):
    if len(img_disp.shape) == 2:  # grayscale
        ax.imshow(img_disp, cmap='gray')
    else:  # color
        ax.imshow(img_disp)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()