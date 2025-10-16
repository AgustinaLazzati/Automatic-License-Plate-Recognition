import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def get_image_paths(root_dir):
    """
    Walks through a directory and its subdirectories to find all image files.
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

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        
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
    
    avg_h_hist = np.mean(h_hists, axis=0).flatten()
    avg_s_hist = np.mean(s_hists, axis=0).flatten()
    avg_v_hist = np.mean(v_hists, axis=0).flatten()
    
    return avg_h_hist, avg_s_hist, avg_v_hist

def compute_fft(histogram_data):
    """
    Computes the Fast Fourier Transform of a histogram and returns the magnitude spectrum.
    """
    # Compute the FFT
    fft_raw = np.fft.fft(histogram_data)
    # Calculate the magnitude (absolute value)
    fft_mag = np.abs(fft_raw)
    return fft_mag

def main():
    # --- 1. Setup argument parser ---
    parser = argparse.ArgumentParser(
        description="Compare the Fourier transforms of average HSV histograms from two nested directories."
    )
    parser.add_argument("-d1", "--dir1", type=str, required=True, help="Path to the first root directory.")
    parser.add_argument("-d2", "--dir2", type=str, required=True, help="Path to the second root directory.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path for saving the plot.")
    args = parser.parse_args()

    # --- 2. Get image paths and calculate average histograms ---
    paths1 = get_image_paths(args.dir1)
    paths2 = get_image_paths(args.dir2)
    h1, s1, v1 = calculate_average_histograms_from_paths(paths1, args.dir1)
    h2, s2, v2 = calculate_average_histograms_from_paths(paths2, args.dir2)

    if h1 is None or h2 is None:
        print("\nComparison failed. Ensure both directories contain valid images.")
        return

    # --- 3. Compute the Fourier Transform for each histogram ---
    print("\n🔬 Computing Fourier Transforms...")
    fft_h1, fft_s1, fft_v1 = compute_fft(h1), compute_fft(s1), compute_fft(v1)
    fft_h2, fft_s2, fft_v2 = compute_fft(h2), compute_fft(s2), compute_fft(v2)
    print("✅ FFT computation complete.")

    # --- 4. Plot the comparison of the FFT spectra ---
    os.makedirs(args.output_dir, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Fourier Transform Comparison of HSV Histograms', fontsize=18, y=0.98)

    label1 = os.path.basename(os.path.normpath(args.dir1))
    label2 = os.path.basename(os.path.normpath(args.dir2))

    # Data for plotting loop
    plot_titles = ['Hue', 'Saturation', 'Value (Brightness)']
    data1 = [fft_h1, fft_s1, fft_v1]
    data2 = [fft_h2, fft_s2, fft_v2]

    for i, (ax, title) in enumerate(zip(axs, plot_titles)):
        # We only plot the first half of the FFT result due to symmetry
        N = len(data1[i])
        ax.plot(np.arange(1, N // 2), data1[i][1:N // 2], color='#1f77b4', label=label1)
        ax.plot(np.arange(1, N // 2), data2[i][1:N // 2], color="#0eebff", label=label2)
        
        ax.set_title(f'Frequency Spectrum of Average {title}')
        ax.set_ylabel('Magnitude')
        ax.set_yscale('log')  # Log scale is often better for viewing frequency spectra
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    axs[-1].set_xlabel('Frequency Component')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- 5. Save the plot ---
    output_path = os.path.join(args.output_dir, 'comparison_fft_hsv.png')
    plt.savefig(output_path)
    
    print(f"\n📊 FFT comparison plot saved successfully to '{output_path}'")

if __name__ == "__main__":
    main()