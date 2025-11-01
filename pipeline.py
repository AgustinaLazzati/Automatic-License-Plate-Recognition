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
    image = cv2.imread(IMAGE)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = YOLO('YOLO/best.pt')
    results = model.predict(image, conf=.5)

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

        plt.figure(figsize=(6, 6))
        plt.imshow(edged_plate, cmap='gray')
        plt.title('Canny Edges')
        plt.show()

        roi = img_original[ymin-20:ymax+20, xmin-20:xmax+20]

        # Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
        contours = cv2.findContours(edged_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        print(f"Number of contours found: {len(contours)}")
        cntrRect = []
        for i in contours:
            epsilon = 0.005*cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,epsilon,True)
            if len(approx) == 4:
                cv2.drawContours(img_original,cntrRect,-1,(0,255,0),2)
                cv2.imshow('Roi Rect ONLY', img_original)
                cntrRect.append(approx)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Warp Perspective
        corners = [[xmin-20, ymin-20], [xmax+20, ymin-20], [xmax+20, ymax+20], [xmin-20, ymax+20]]
        corners_original = np.array(corners, dtype="float32")

        corners = corners_original - np.array([xmin-20, ymin-20], dtype="float32")

        M = cv2.getPerspectiveTransform(corners, np.array([[0, 0], [200, -20], [200, 80], [0, 120]], dtype="float32"))
        width, height = 200, 80
        warped_plate = cv2.warpPerspective(img_original, M, (width, height))

        plt.figure(figsize=(6, 6))
        plt.imshow(warped_plate, cmap='gray')
        plt.title('Watershed Segmentation')
        plt.show()


if __name__ == "__main__":
    main()
