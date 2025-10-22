import argparse

import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

IMAGE = 'data/real_plates/Frontal/0216KZP.jpg'

def main():
    parser = argparse.ArgumentParser(description="Detect license plates in images.")
    parser.add_argument(
        "image", type=str, nargs="?"
    )
    args = parser.parse_args()
    
    image_path = args.image
    image = cv2.imread(image_path)
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = YOLO('YOLO/best.pt')
    results = model.predict(img_rgb, conf=.5)

    for r in results:
        im_array = r.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(im)
        plt.axis('off')
        plt.show()


# --- 2. CROP AND TRANSFORM LOGIC ---
    for r in results:
        # Check if any detection was made
        if len(r.boxes) == 0:
            continue

        # Get the bounding box coordinates (the first detection)
        # xyxy format: [xmin, ymin, xmax, ymax]
        # We use .cpu().numpy().astype(int)[0] to extract the coordinates as integers
        # [0] targets the first detected license plate
        box = r.boxes.xyxy.cpu().numpy().astype(int)[0] 
        xmin, ymin, xmax, ymax = box

        # 2.1. Crop the Plate
        # Crop the license plate region from the original RGB image
        img_original = img_rgb[ymin-20:ymax+20, xmin-20:xmax+20]
        
        # 2.2. Preprocessing for Warp Perspective
        gray_plate = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
        blurred_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)
        edged_plate = cv2.Canny(blurred_plate, 50, 150)

        # Warp Perspective
        corners = [[xmin-20, ymin-20], [xmax+20, ymin-20], [xmax+20, ymax+20], [xmin-20, ymax+20]]
        corners_original = np.array(corners, dtype="float32")

        corners = corners_original - np.array([xmin-20, ymin-20], dtype="float32")

        M = cv2.getPerspectiveTransform(corners, np.array([[0, 0], [200, -20], [200, 80], [0, 120]], dtype="float32"))
        width, height = 200, 80
        warped_plate = cv2.warpPerspective(img_original, M, (width, height))

        img = cv2.cvtColor(warped_plate, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_eq = clahe.apply(img)

        kernel_blackhat = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        img_blackhat = cv2.morphologyEx(img_eq, cv2.MORPH_BLACKHAT, kernel_blackhat)

        img_prep = cv2.GaussianBlur(img_blackhat, (3,3), 0)

        _, binary = cv2.threshold(img_prep, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

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

        plt.figure(figsize=(6, 6))
        plt.imshow(img_color)
        plt.title('Watershed Segmentation')
        plt.show()

        char_mask = np.zeros_like(img, dtype=np.uint8)
        unique_labels = np.unique(markers)

        # Heuristics: we skip background (label 1) and borders (-1)
        for label in unique_labels:
            if label <= 1:
                continue  # skip background and border
            region_mask = np.uint8(markers == label) * 255
            area = cv2.countNonZero(region_mask)

            # Filter by area and bounding box ratio (since characters have moderate height/width)
            if 50 < area < img.shape[0] * img.shape[1] * 0.3:  # skip too small or too large
                x, y, w, h = cv2.boundingRect(region_mask)
                aspect_ratio = h / float(w)
                if 1.0 < aspect_ratio < 5.0:  # plausible aspect ratio for characters
                    char_mask[markers == label] = 255

        # finally clean the mask
        kernel = np.ones((3,3), np.uint8)
        char_mask = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        char_mask = cv2.morphologyEx(char_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(char_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_boxes = []

        h_plate, w_plate = img.shape[:2]
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h

            # Filter out noise or oversized regions
            if 100 < area < (h_plate * w_plate * 0.2):
                aspect = h / float(w)
                if 1.0 < aspect < 6.0:  # characters are usually tall and narrow
                    char_boxes.append((x, y, w, h))
                    # draw for visualization
                    cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Sort boxes left-to-right
        char_boxes = sorted(char_boxes, key=lambda b: b[0])
        print(f"Detected {len(char_boxes)} characters.")

        # Display the warped plate
        plt.figure(figsize=(6, 6))
        plt.imshow(img_original, cmap='gray')
        plt.title('Warped License Plate')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
