import cv2
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

def get_v_distribution(folder, size=(256, 256)):
    v_values = []
    for filename in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, size)
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                _, _, v = cv2.split(hsv_img)
                v_values.extend(v.flatten())

            del img
    return v_values

folder1 = 'data/real_plates/Frontal'
folder2 = 'data/real_plates/Lateral'

v_dist_folder1 = get_v_distribution(folder1)
v_dist_folder2 = get_v_distribution(folder2)

plt.figure(figsize=(10, 6))
plt.hist(v_dist_folder1, bins=256, alpha=0.5, label='Frontal', color='blue')
plt.hist(v_dist_folder2, bins=256, alpha=0.5, label='Lateral', color='red')
plt.title('Value (V) Channel Distribution Comparison')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
