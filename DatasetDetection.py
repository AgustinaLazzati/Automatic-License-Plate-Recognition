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

from LicensePlateDetector import detectPlates  


def normalize_view(view):
    """Map augmented folder names to normal ones."""
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
                if len(plates) > 0:
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
    Own_DataDir = r"data/Patentes"
    Augmented_DataDir = r"data/Patentes"

    Views = ["Frontal", "Lateral"]
    Views_A = ["FrontalAugmented", "LateralAugmented"]

    datasets = {
        "Real_Plates": (Real_DataDir, Views),
        "New_plates": (Own_DataDir, Views),
        "NewAugmented_plates": (Augmented_DataDir, Views_A),
    }

    image_counts, detected_counts = analyze_datasets(datasets)

    datasets_list = list(datasets.keys())
    colors = ["#1F4E79", "#4A90E2", "#87CEFA"]

    # ---------- Plot number of images ----------
    for view in ["Frontal", "Lateral"]:
        values = [image_counts[d].get(view, 0) for d in datasets_list]
        plot_bar(
            labels=datasets_list,
            values=values,
            title=f"Number of {view} Images per Dataset",
            ylabel="Number of Images",
            colors=colors
        )

    # ---------- Plot detection accuracy (%) ----------
    for view in ["Frontal", "Lateral"]:
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
