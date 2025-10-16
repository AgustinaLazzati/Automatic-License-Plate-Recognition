import time
# import the necessary packages
import numpy as np
import cv2
import glob
import os
import argparse
import random
from imutils import perspective
from matplotlib import pyplot as plt

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Detect license plates in images.")
    parser.add_argument(
        "datapath", type=str, nargs="?", default=os.path.join(".", "data", "real_plates", "Frontal"), help="Path to the directory containing images."
    )
    args = parser.parse_args()
    
    datapath = args.datapath
    datapath = os.path.abspath(datapath)

    if not os.path.isdir(datapath):
        print(f"Error: Directory '{datapath}' not found.")
        return

    image_files = [
        f for f in os.listdir(datapath) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print(f"Error: No images found in '{datapath}'.")
        return

    image_name = random.choice(image_files)
    image_path = os.path.join(datapath, image_name)
    
    print(f"Processing image: {image_path}")

    # Load YOLO model
    model = YOLO("yolov5nu.pt")
    print(model)
    
    # Process the single image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at '{image_path}'.")
        return

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    time1 = time.time()
    print(time1)
    results = model.predict(img_rgb, conf=.8)
    
    
    # Show results
    for r in results:
        im_array = r.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(im)
        plt.title(f"YOLOv8 Detection for {image_name}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
