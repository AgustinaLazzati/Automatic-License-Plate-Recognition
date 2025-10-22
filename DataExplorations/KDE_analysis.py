"""
Protocol Analysis - KDE Comparison Across Datasets

Compares properties (plateAngle, imageColor, imageIlluminance, imageSaturation)
for Frontal and Lateral views across Real, Own, and Augmented datasets.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from DataExploration import computeProperties, setup_plots_folder, PLOTS_DIR

# -------------------------------
# FUNCTION TO PLOT KDE COMPARISON
def plot_kde_comparison(dataset_names, dataset_dicts, Views, property_name, save_dir=PLOTS_DIR):
    """
    Compare distributions of a property across datasets using KDE with filled area,
    and highlight shared areas.
    
    dataset_dicts: list of dicts returned by computeProperties(), e.g. [R_plateAngle, O_plateAngle, A_plateAngle]
    Views: list of views to compare, e.g., ["Frontal", "Lateral"]
    property_name: "plateAngle", "imageColor", "imageIlluminance", "imageSaturation"
    """
    os.makedirs(save_dir, exist_ok=True)
    colors = ['b', 'purple', 'c']  # colors for datasets
    shared_color = 'pink'  # vibrant color for shared area
    
    for view in Views:
        plt.figure()
        kdes = []
        x_min, x_max = float('inf'), float('-inf')
        
        # First pass: compute KDEs and global min/max for x-axis
        for data_dict in dataset_dicts:
            values = data_dict.get(view, [])
            if len(values) < 2:
                kdes.append(None)
                continue
            x_min = min(x_min, min(values))
            x_max = max(x_max, max(values))
            kde = gaussian_kde(values)
            kdes.append(kde)
        
        if x_min >= x_max:
            continue  # skip if not enough data
        
        x_vals = np.linspace(x_min, x_max, 200)
        kde_scaled_list = []
        
        # Compute scaled KDE values
        for i, kde in enumerate(kdes):
            values = dataset_dicts[i].get(view, [])
            if kde is None:
                kde_scaled_list.append(np.zeros_like(x_vals))
            else:
                kde_scaled = kde(x_vals) * len(values) * ((x_max - x_min) / 20)
                kde_scaled_list.append(kde_scaled)
                # Fill individual dataset area
                plt.fill_between(x_vals, kde_scaled, color=colors[i], alpha=0.5, label=dataset_names[i])
        
        # Compute shared area (min of all KDEs)
        shared_area = np.min(kde_scaled_list, axis=0)
        plt.fill_between(x_vals, 0, shared_area, color=shared_color, hatch='//', alpha=0.7, label="Shared Area")
        
        plt.title(f"{property_name} KDE Comparison - {view}")
        plt.xlabel(property_name)
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"KDE_{property_name}_{view}.png"))
        plt.close()

# -------------------------------
# MAIN PROTOCOL ANALYSIS
def main():
    setup_plots_folder()
    
    # Define datasets. NOW WE ADDED ALL THE IMAGES TAKEN BY OUR COLLEAGES. 
    Real_DataDir = r"data"
    WithProtocol_DataDir = r"data/All/with_Protocol"
    WithoutProtocol_DataDir = r"data/All/without_Protocol"
    
    Views = ["Frontal", "Lateral"]
    
    datasets = {
        "Real_Plates": (Real_DataDir, Views),
        "With_Protocol": (WithProtocol_DataDir, Views),
        "Without_Protocol": (WithoutProtocol_DataDir, Views),
    }
    
    # Compute properties for all datasets
    print("Computing properies of the complete dataset...")
    computed = {}
    for name, (dir_path, views) in datasets.items():
        plateArea, plateAngle, imageColor, imageIlluminance, imageSaturation = computeProperties(dir_path, views)
        computed[name] = {
            "plateArea": plateArea,
            "plateAngle": plateAngle,
            "imageColor": imageColor,
            "imageIlluminance": imageIlluminance,
            "imageSaturation": imageSaturation
        }
    
    dataset_names = list(computed.keys())
    
    # Properties to compare
    properties = ["plateAngle", "imageColor", "imageIlluminance", "imageSaturation"]
    
    # Plot KDE comparisons for each property and each view
    print("Creating KDE plots...")
    for prop in properties:
        for view in Views:
            dataset_dicts = [computed[name][prop] for name in dataset_names]
            
            # Build remapped dicts per view
            remapped_dicts = []
            for d in dataset_dicts:
                remapped_dicts.append({view: d.get(view, [])})
            
            plot_kde_comparison(dataset_names, remapped_dicts, [view], property_name=prop)
    
    print("All KDE comparison plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
