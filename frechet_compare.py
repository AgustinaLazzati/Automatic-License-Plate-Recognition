import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import similaritymeasures

# --- Helper function to prepare data for Fréchet distance calculation ---
def prepare_for_frechet(data):
    """
    Converts a 1D array (like a histogram) into a 2D array of (index, value)
    points, which is the format required by the similaritymeasures library.
    """
    curve = np.column_stack((np.arange(len(data)), data))
    return curve

# --- Functions from previous script (unchanged) ---
def get_image_paths(root_dir):
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
    if not image_paths:
        print(f"❌ No valid images found for '{dir_name}'.")
        return None, None, None
    h_hists, s_hists, v_hists = [], [], []
    print(f"📂 Processing {len(image_paths)} images from '{dir_name}'...")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None: continue
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
    if not h_hists: return None, None, None
    print(f"✅ Finished processing for '{dir_name}'.")
    return (np.mean(h_hists, axis=0).flatten(), 
            np.mean(s_hists, axis=0).flatten(), 
            np.mean(v_hists, axis=0).flatten())

def compute_fft(histogram_data):
    return np.abs(np.fft.fft(histogram_data))

# --- Plotting Functions ---
def plot_comparison(data1, data2, labels, titles, x_limits, output_path, y_scale='linear'):
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    main_title = ' '.join(output_path.split(os.sep)[-1].split('_')[1:-1]).title()
    fig.suptitle(f'{main_title} Comparison', fontsize=18, y=0.98)
    for i, ax in enumerate(axs):
        N = len(data1[i])
        x_range = np.arange(N) if y_scale == 'linear' else np.arange(1, N // 2)
        plot_data1 = data1[i] if y_scale == 'linear' else data1[i][1:N//2]
        plot_data2 = data2[i] if y_scale == 'linear' else data2[i][1:N//2]
        
        ax.plot(x_range, plot_data1, color='#1f77b4', label=labels[0])
        ax.plot(x_range, plot_data2, color='#ff7f0e', label=labels[1])
        ax.set_title(f'Average {titles[i]}')
        ax.set_ylabel('Magnitude' if y_scale != 'linear' else 'Normalized Frequency')
        ax.set_xlim([0, x_limits[i]])
        if y_scale == 'log': ax.set_yscale('log')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    axs[-1].set_xlabel('Bin Value' if y_scale == 'linear' else 'Frequency Component')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"📊 Plot saved successfully to '{output_path}'")

def main():
    parser = argparse.ArgumentParser(description="Quantify differences between two image sets using HSV histograms, FFT, and Fréchet distance.")
    parser.add_argument("-d1", "--dir1", type=str, required=True, help="Path to the first root directory.")
    parser.add_argument("-d2", "--dir2", type=str, required=True, help="Path to the second root directory.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path for saving plots and results.")
    args = parser.parse_args()

    # 1. & 2. Compute Histograms and FFTs
    paths1, paths2 = get_image_paths(args.dir1), get_image_paths(args.dir2)
    h1, s1, v1 = calculate_average_histograms_from_paths(paths1, args.dir1)
    h2, s2, v2 = calculate_average_histograms_from_paths(paths2, args.dir2)
    if h1 is None or h2 is None: return

    fft_h1, fft_s1, fft_v1 = compute_fft(h1), compute_fft(s1), compute_fft(v1)
    fft_h2, fft_s2, fft_v2 = compute_fft(h2), compute_fft(s2), compute_fft(v2)

    # 3. Compute Fréchet Distances
    print("\n📏 Computing Fréchet Distances...")
    dist_h_hist = similaritymeasures.frechet_dist(prepare_for_frechet(h1), prepare_for_frechet(h2))
    dist_s_hist = similaritymeasures.frechet_dist(prepare_for_frechet(s1), prepare_for_frechet(s2))
    dist_v_hist = similaritymeasures.frechet_dist(prepare_for_frechet(v1), prepare_for_frechet(v2))
    
    dist_h_fft = similaritymeasures.frechet_dist(prepare_for_frechet(fft_h1), prepare_for_frechet(fft_h2))
    dist_s_fft = similaritymeasures.frechet_dist(prepare_for_frechet(fft_s1), prepare_for_frechet(fft_s2))
    dist_v_fft = similaritymeasures.frechet_dist(prepare_for_frechet(fft_v1), prepare_for_frechet(fft_v2))

    # 4. Average the measures
    avg_hist_dist = np.mean([dist_h_hist, dist_s_hist, dist_v_hist])
    avg_fft_dist = np.mean([dist_h_fft, dist_s_fft, dist_v_fft])
    
    # --- Print Results Summary ---
    label1 = os.path.basename(os.path.normpath(args.dir1))
    label2 = os.path.basename(os.path.normpath(args.dir2))
    print("\n--- Quantitative Comparison Results ---")
    print(f"Comparing '{label1}' vs '{label2}':\n")
    print("--- Histogram Shape Distance (Fréchet) ---")
    print(f"  - Hue Diff:        {dist_h_hist:.4f}")
    print(f"  - Saturation Diff: {dist_s_hist:.4f}")
    print(f"  - Value Diff:      {dist_v_hist:.4f}")
    print(f"  - AVERAGE HISTOGRAM DISTANCE: {avg_hist_dist:.4f}\n")
    
    print("--- FFT Spectrum Shape Distance (Fréchet) ---")
    print(f"  - Hue FFT Diff:        {dist_h_fft:.4f}")
    print(f"  - Saturation FFT Diff: {dist_s_fft:.4f}")
    print(f"  - Value FFT Diff:      {dist_v_fft:.4f}")
    print(f"  - AVERAGE FFT DISTANCE: {avg_fft_dist:.4f}\n")
    print("---------------------------------------")

    # --- Generate and Save Plots ---
    os.makedirs(args.output_dir, exist_ok=True)
    labels = [label1, label2]
    plot_titles = ['Hue', 'Saturation', 'Value']
    
    plot_comparison([h1, s1, v1], [h2, s2, v2], labels, plot_titles, 
                    [180, 256, 256], os.path.join(args.output_dir, 'comparison_histograms_downloaded.png'))
    
    plot_comparison([fft_h1, fft_s1, fft_v1], [fft_h2, fft_s2, fft_v2], labels, plot_titles, 
                    [180//2, 256//2, 256//2], os.path.join(args.output_dir, 'comparison_fft_spectra_downloaded.png'), y_scale='log')

if __name__ == "__main__":
    main()