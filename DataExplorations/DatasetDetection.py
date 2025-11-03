# -*- coding: utf-8 -*-
"""
Dataset Analysis Script
Creates bar plots comparing number of images and detection accuracy
across Real_Plates, New_plates, and NewAugmented_plates datasets.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from ExploreParameters import detectPlates  


def normalize_view(view):
    """
    Map augmented folder names to normal ones.
    This was needed before to normalize the views do to the names of the folders 
    """
    if "Frontal" in view:
        return "Frontal"
    if "Lateral" in view:
        return "Lateral"
    return view  # fallback

def analyze_datasets(datasets):
    # Counting images per folder
    image_counts = {}
    detected_counts = {}

    for dataset_name, (base_dir, views) in datasets.items():
        image_counts[dataset_name] = {}
        detected_counts[dataset_name] = {}

        for view in views:
            folder = os.path.join(base_dir, view)
            norm_view = normalize_view(view)

            if not os.path.isdir(folder):
                print(f"Warning: {folder} not found, skipping.")
                image_counts[dataset_name][norm_view] = 0
                detected_counts[dataset_name][norm_view] = 0
                continue

            image_files = [
                f for f in os.listdir(folder) if f.endswith((".jpg", ".jpeg", ".png"))
            ]
            total = len(image_files)
            detected = 0

            for fname in image_files:
                img_path = os.path.join(folder, fname)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                plates, _ = detectPlates(img)

                
                #IF NO PLATES DETECTED, WE WILL ZOOM IN OR ZOOM OUT THE IMAGE.
                if len(plates) == 0:
                    # Zoom-in: crop center 90% and resize back 
                    h, w = img.shape[:2]
                    crop_x1, crop_y1 = int(w * 0.05), int(h * 0.05)
                    crop_x2, crop_y2 = int(w * 0.95), int(h * 0.95)
                    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
                    zoomed_in = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
                    plates, _ = detectPlates(zoomed_in)
                """
                if len(plates) == 0:
                    # MORE AGRESSIVE Zoom-in: crop center 70% and resize back
                    h, w = img.shape[:2]
                    crop_x1, crop_y1 = int(w*0.15), int(h*0.15)
                    crop_x2, crop_y2 = int(w*0.85), int(h*0.85)
                    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
                    zoomed_in = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
                    plates, _ = detectPlates(zoomed_in)
                """
                if len(plates) == 0:
                    # Zoom-out: add black padding and resize back
                    zoom_factor = 1.5
                    new_h, new_w = int(h*zoom_factor), int(w*zoom_factor)
                    canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    y_offset = (new_h - h)//2
                    x_offset = (new_w - w)//2
                    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img
                    zoomed_out = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_CUBIC)
                    plates, _ = detectPlates(zoomed_out)
        
                
                if len(plates) >0 and len(plates)<=2:
                    detected += 1

            image_counts[dataset_name][norm_view] = total
            detected_counts[dataset_name][norm_view] = detected

    return image_counts, detected_counts


def plot_bar(labels, values, title, ylabel, colors, percentages=None, y_limit=None):
    #Simple bar plot with optional percentage labels.
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors)
    
    if percentages:
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(values)*0.02 if max(values)>0 else 0.02,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10
            )
    if y_limit:
        plt.ylim(0, y_limit)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def main():
    Real_DataDir = r"data"
    WithProtocol_DataDir = r"data/All/with_Protocol"
    WithoutProtocol_DataDir = r"data/All/without_Protocol"

    Views = ["Frontal", "Lateral"]

    datasets = {
        "Real_Plates": (Real_DataDir, Views),
        "With_Protocol": (WithProtocol_DataDir, Views),
        "Without_Protocol": (WithoutProtocol_DataDir, Views),
    }

    image_counts, detected_counts = analyze_datasets(datasets)

    datasets_list = list(datasets.keys())
    colors = ["#1F4E79", "#4A90E2", "#87CEFA"]

    # ---------- Plot number of images ----------
    for view in Views:
        values = [image_counts[d].get(view, 0) for d in datasets_list]
        plot_bar(
            labels=datasets_list,
            values=values,
            title=f"Number of {view} Images per Dataset",
            ylabel="Number of Images",
            colors=colors
        )

    # ---------- Plot detection accuracy (%) ----------
    for view in Views:
        percentages = []
        for d in datasets_list:
            total = image_counts[d].get(view, 0)
            detected = detected_counts[d].get(view, 0)
            pct = (detected / total * 100) if total > 0 else 0
            percentages.append(pct)
        plot_bar(
            labels=datasets_list,
            values=percentages,
            title=f"Detection Accuracy (%) for {view}",
            ylabel="Accuracy (%)",
            colors=colors,
            y_limit=100
        )

    print("\n===== SUMMARY =====")
    for dataset in datasets_list:
        print(f"\n{dataset}:")
        for view in ["Frontal", "Lateral"]:
            total = image_counts[dataset].get(view, 0)
            detected = detected_counts[dataset].get(view, 0)
            rate = (detected / total * 100) if total > 0 else 0
            print(f"  {view}: {detected}/{total} detected ({rate:.2f}%)")


if __name__ == "__main__":
    main()
