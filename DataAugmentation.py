import cv2
import os
import random
import numpy as np

# Paths (THIS CHANGES SINCE DATA IS IN GITIGNORE)
input_folder = "./data/Patentes/Frontal"
output_folder = "./data/Patentes/LateralAugmented"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(input_folder, filename)

        #Read image
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read {filename}")
            continue

        name, ext = os.path.splitext(filename)   

        # Random brightness adjustment (dark or bright)
        # 50% chance dark, 50% chance bright
        if random.random() < 0.5:
            # Dark version
            alpha = round(random.uniform(0.2, 0.5), 2)  # low contrast
            beta = random.randint(-10, 20)           # subtract brightness to darken
            img_mod = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            suffix = "O"  # O for dark

        else:
            # Bright / glare version
            alpha = round(random.uniform(1.5, 2.0), 2)  # increase contrast
            beta = random.randint(30, 60)               # add glare/brightness
            img_mod = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            suffix = "B"  # B for bright

        # Save adjusted image
        cv2.imwrite(os.path.join(output_folder, f"{name}{suffix}{ext}"), img_mod)

        # Strong random blur
        k = random.choice([21, 25, 27, 31, 35])
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        cv2.imwrite(os.path.join(output_folder, f"{name}BL{ext}"), blurred)

        #print(f"{filename} -> Random brightness (alpha={alpha}, beta={beta}, suffix={suffix}), Strong blur (kernel={k})")