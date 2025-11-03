import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def get_image_paths(root_dir):
    """
    Walks through a directory and its subdirectories to find all image files.

    Args:
        root_dir (str): The path to the root directory to search.

    Returns:
        list: A list of full paths to all found image files.
    """
    image_paths = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    if not os.path.isdir(root_dir):
        print(f"❌ Error: Directory not found at '{root_dir}'")
        return image_paths

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

def calculate_average_histograms_from_paths(image_paths, dir_name):
    """
    Calculates the average HSV histograms from a list of image file paths.

    Args:
        image_paths (list): A list of paths to the image files.
        dir_name (str): The name of the root directory for logging purposes.

    Returns:
        tuple: A tuple of (Hue, Saturation, Value) histograms as NumPy arrays.
               Returns (None, None, None) if the list is empty.
    """
    if not image_paths:
        print(f"❌ No valid images found for '{dir_name}'.")
        return None, None, None

    h_hists, s_hists, v_hists = [], [], []
    
    print(f"📂 Processing {len(image_paths)} images from '{dir_name}'...")
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Warning: Could not read image '{os.path.basename(image_path)}'. Skipping.")
            continue

        # Convert to HSV and calculate histograms
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        
        # Normalize to account for different image sizes
        cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        h_hists.append(hist_h)
        s_hists.append(hist_s)
        v_hists.append(hist_v)
    
    if not h_hists:
        print(f"❌ Could not process any images for '{dir_name}'.")
        return None, None, None
    
    print(f"✅ Finished processing for '{dir_name}'.")
    
    # Calculate the mean of all collected histograms
    avg_h_hist = np.mean(h_hists, axis=0)
    avg_s_hist = np.mean(s_hists, axis=0)
    avg_v_hist = np.mean(v_hists, axis=0)
    
    return avg_h_hist, avg_s_hist, avg_v_hist

def main():
    # --- 1. Setup command-line argument parser ---
    parser = argparse.ArgumentParser(
        description="Compare average HSV histograms from two nested directory structures."
    )
    parser.add_argument(
        "-d1", "--dir1", type=str, required=True,
        help="Path to the first root directory (e.g., 'own_plates')."
    )
    parser.add_argument(
        "-d2", "--dir2", type=str, required=True,
        help="Path to the second root directory (e.g., 'real_plates')."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="Path to the directory where the comparison plot will be saved."
    )
    args = parser.parse_args()

    # --- 2. Collect all image paths from the nested directories ---
    paths1 = get_image_paths(args.dir1)
    paths2 = get_image_paths(args.dir2)

    # --- 3. Calculate histograms for both sets of images ---
    h1, s1, v1 = calculate_average_histograms_from_paths(paths1, args.dir1)
    h2, s2, v2 = calculate_average_histograms_from_paths(paths2, args.dir2)

    if h1 is None or h2 is None:
        print("\nComparison failed. Ensure both directories contain valid images.")
        return

    # --- 4. Plot the comparison ---
    os.makedirs(args.output_dir, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Comparison of Average HSV Histograms', fontsize=18, y=0.98)

    label1 = os.path.basename(os.path.normpath(args.dir1))
    label2 = os.path.basename(os.path.normpath(args.dir2))

    # Plot Hue, Saturation, and Value comparisons
    plot_titles = ['Hue', 'Saturation', 'Value (Brightness)']
    data1 = [h1, s1, v1]
    data2 = [h2, s2, v2]
    x_limits = [180, 256, 256]

    for i, (ax, title) in enumerate(zip(axs, plot_titles)):
        ax.plot(data1[i], color='#1f77b4', label=label1)  # Muted Blue
        ax.plot(data2[i], color='#ff7f0e', label=label2)  # Safety Orange
        ax.set_title(f'Average {title} Comparison')
        ax.set_xlim([0, x_limits[i]])
        ax.set_ylabel('Normalized Frequency')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    axs[-1].set_xlabel('Bin Value') # Add x-label only to the last plot
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- 5. Save the final plot ---
    output_path = os.path.join(args.output_dir, 'comparison_nested_hsv_histograms.png')
    plt.savefig(output_path)
    
    print(f"\n📊 Comparison plot saved successfully to '{output_path}'")

if __name__ == "__main__":
    main()
