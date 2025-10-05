import cv2
import numpy as np
import argparse
import similaritymeasures
from hsv_utils import compute_fft, prepare_for_frechet

def process_single_image(image_path):
    """Calculates HSV histograms and their FFTs for a single image."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate and normalize histograms
    h_hist = cv2.normalize(cv2.calcHist([hsv_image], [0], None, [180], [0, 180]), None, 0, 1, cv2.NORM_MINMAX).flatten()
    s_hist = cv2.normalize(cv2.calcHist([hsv_image], [1], None, [256], [0, 256]), None, 0, 1, cv2.NORM_MINMAX).flatten()
    v_hist = cv2.normalize(cv2.calcHist([hsv_image], [2], None, [256], [0, 256]), None, 0, 1, cv2.NORM_MINMAX).flatten()
    
    # Compute FFTs
    h_fft = compute_fft(h_hist)
    s_fft = compute_fft(s_hist)
    v_fft = compute_fft(v_hist)
    
    return h_hist, s_hist, v_hist, h_fft, s_fft, v_fft

def main():
    parser = argparse.ArgumentParser(description="Classify a single image against a pre-computed dataset fingerprint.")
    parser.add_argument("image", type=str, help="Path to the new image to classify.")
    parser.add_argument("-f", "--fingerprint", type=str, default='real_fingerprint.npz', help="Path to the .npz fingerprint file.")
    parser.add_argument("-t", "--threshold", type=float, default=10.0, help="FFT distance threshold for classification (default: 8.0).")
    args = parser.parse_args()

    # 1. Load the pre-computed fingerprint
    try:
        fingerprint = np.load(args.fingerprint)
    except FileNotFoundError:
        print(f"❌ Error: Fingerprint file not found at '{args.fingerprint}'")
        return
        
    base_h_hist, base_s_hist, base_v_hist = fingerprint['h_hist'], fingerprint['s_hist'], fingerprint['v_hist']
    base_h_fft, base_s_fft, base_v_fft = fingerprint['h_fft'], fingerprint['s_fft'], fingerprint['v_fft']

    # 2. Process the new image
    new_data = process_single_image(args.image)
    if new_data is None:
        print(f"❌ Error: Could not read or process the image at '{args.image}'")
        return
    new_h_hist, new_s_hist, new_v_hist, new_h_fft, new_s_fft, new_v_fft = new_data
    
    # 3. Calculate Fréchet distances
    dist_h_hist = similaritymeasures.frechet_dist(prepare_for_frechet(new_h_hist), prepare_for_frechet(base_h_hist))
    dist_s_hist = similaritymeasures.frechet_dist(prepare_for_frechet(new_s_hist), prepare_for_frechet(base_s_hist))
    dist_v_hist = similaritymeasures.frechet_dist(prepare_for_frechet(new_v_hist), prepare_for_frechet(base_v_hist))

    dist_h_fft = similaritymeasures.frechet_dist(prepare_for_frechet(new_h_fft), prepare_for_frechet(base_h_fft))
    dist_s_fft = similaritymeasures.frechet_dist(prepare_for_frechet(new_s_fft), prepare_for_frechet(base_s_fft))
    dist_v_fft = similaritymeasures.frechet_dist(prepare_for_frechet(new_v_fft), prepare_for_frechet(base_v_fft))

    avg_hist_dist = np.mean([dist_h_hist, dist_s_hist, dist_v_hist])
    avg_fft_dist = np.mean([dist_h_fft, dist_s_fft, dist_v_fft])

    # 4. Print results and classify
    print("\n--- Image Analysis Results ---")
    print(f"Image: '{args.image}'")
    print(f"Fingerprint: '{args.fingerprint}'\n")
    print("--- Distance Scores ---")
    print(f"  - Avg. Histogram Distance: {avg_hist_dist:.4f}")
    print(f"  - Avg. FFT Distance:       {avg_fft_dist:.4f}\n")
    print("--- Classification ---")
    print(f"Threshold: {args.threshold:.2f}")
    if avg_fft_dist < args.threshold:
        print(f"✅ Result: Image is CONSISTENT with the fingerprint.")
    else:
        print(f"❌ Result: Image is an OUTLIER.")
    print("------------------------")

if __name__ == "__main__":
    main()
