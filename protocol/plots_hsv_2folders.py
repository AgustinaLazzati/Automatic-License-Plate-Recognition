import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def calculate_average_histograms(image_dir):
    """
    Calculates the average HSV histograms for all images in a single directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        tuple: A tuple of (Hue, Saturation, Value) histograms as NumPy arrays.
               Returns (None, None, None) if no valid images are found.
    """
    if not os.path.isdir(image_dir):
        print(f"❌ Error: Directory not found at '{image_dir}'")
        return None, None, None

    h_hists, s_hists, v_hists = [], [], []
    image_count = 0
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    print(f"📂 Processing images in '{image_dir}'...")
    
    # Iterate through all files in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(supported_extensions):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"⚠️ Warning: Could not read image '{filename}'. Skipping.")
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
            image_count += 1
    
    if image_count == 0:
        print(f"❌ No valid images found in '{image_dir}'.")
        return None, None, None

    print(f"✅ Processed {image_count} images from '{image_dir}'.")

    # Calculate the mean of all histograms in the list
    avg_h_hist = np.mean(h_hists, axis=0)
    avg_s_hist = np.mean(s_hists, axis=0)
    avg_v_hist = np.mean(v_hists, axis=0)
    
    return avg_h_hist, avg_s_hist, avg_v_hist

def main():
    # --- 1. Setup command-line argument parser ---
    parser = argparse.ArgumentParser(
        description="Compare the average HSV histograms of images in two separate directories."
    )
    parser.add_argument(
        "-d1", "--dir1", type=str, required=True,
        help="Path to the first directory of images."
    )
    parser.add_argument(
        "-d2", "--dir2", type=str, required=True,
        help="Path to the second directory of images."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="Path to the directory where the comparison plot will be saved."
    )
    args = parser.parse_args()

    # --- 2. Calculate histograms for both directories ---
    h1, s1, v1 = calculate_average_histograms(args.dir1)
    h2, s2, v2 = calculate_average_histograms(args.dir2)

    # Exit if either directory failed to produce a result
    if h1 is None or h2 is None:
        print("\nComparison cannot be generated as one or both directories lacked valid images.")
        return

    # --- 3. Plot the comparison ---
    os.makedirs(args.output_dir, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('HSV Histograms for Real Data', fontsize=18, y=0.98)

    # Use directory names as labels in the legend
    label1 = os.path.basename(os.path.normpath(args.dir1))
    label2 = os.path.basename(os.path.normpath(args.dir2))

    # Plot Hue Comparison
    axs[0].plot(h1, color='#1f77b4', label=label1)  # Muted Blue
    axs[0].plot(h2, color='#ff7f0e', label=label2)  # Safety Orange
    axs[0].set_title('Average Hue Comparison')
    axs[0].set_xlim([0, 180])
    axs[0].set_ylabel('Normalized Frequency')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Plot Saturation Comparison
    axs[1].plot(s1, color='#1f77b4', label=label1)
    axs[1].plot(s2, color='#ff7f0e', label=label2)
    axs[1].set_title('Average Saturation Comparison')
    axs[1].set_xlim([0, 256])
    axs[1].set_ylabel('Normalized Frequency')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Plot Value Comparison
    axs[2].plot(v1, color='#1f77b4', label=label1)
    axs[2].plot(v2, color='#ff7f0e', label=label2)
    axs[2].set_title('Average Value (Brightness) Comparison')
    axs[2].set_xlim([0, 256])
    axs[2].set_xlabel('Bin Value')
    axs[2].set_ylabel('Normalized Frequency')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- 4. Save the final plot ---
    output_path = os.path.join(args.output_dir, 'comparison_hsv_histograms.png')
    plt.savefig(output_path)
    
    print(f"\n📊 Comparison plot saved successfully to '{output_path}'")

if __name__ == "__main__":
    main()