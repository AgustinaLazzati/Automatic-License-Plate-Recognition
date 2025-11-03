import numpy as np
import argparse
from hsv_utils import get_image_paths, calculate_average_histograms, compute_fft

def main():
    parser = argparse.ArgumentParser(description="Create a dataset fingerprint from HSV histograms and their FFTs.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to the directory of authentic images.")
    parser.add_argument("-o", "--output_file", type=str, default="fingerprint.npz", help="Path to save the output .npz fingerprint file (default: fingerprint.npz).")
    args = parser.parse_args()

    # 1. Get all image paths from the directory
    image_paths = get_image_paths(args.input_dir)
    if not image_paths:
        print(f"❌ No images found in '{args.input_dir}'. Aborting.")
        return

    # 2. Calculate the average histograms for the dataset
    h_hist, s_hist, v_hist = calculate_average_histograms(image_paths, args.input_dir)
    if h_hist is None:
        print(f"❌ Failed to process histograms for '{args.input_dir}'. Aborting.")
        return

    # 3. Compute the FFT for each average histogram
    h_fft = compute_fft(h_hist)
    s_fft = compute_fft(s_hist)
    v_fft = compute_fft(v_hist)

    # 4. Save all 6 arrays into a single compressed .npz file
    np.savez_compressed(
        args.output_file,
        h_hist=h_hist,
        s_hist=s_hist,
        v_hist=v_hist,
        h_fft=h_fft,
        s_fft=s_fft,
        v_fft=v_fft
    )
    print(f"\n✅ Fingerprint saved successfully to '{args.output_file}'")

if __name__ == "__main__":
    main()
