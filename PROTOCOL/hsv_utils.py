import os
import cv2
import numpy as np

def get_image_paths(root_dir):
    """Walks through a directory to find all image files."""
    image_paths = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    if not os.path.isdir(root_dir):
        return None
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

def calculate_average_histograms(image_paths, dir_name):
    """Calculates the average HSV histograms from a list of image paths."""
    if not image_paths: return None, None, None
    h_hists, s_hists, v_hists = [], [], []
    print(f"📂 Processing {len(image_paths)} images for '{dir_name}' fingerprint...")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None: continue
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Calculate and normalize histograms for each channel
        hist_h = cv2.normalize(cv2.calcHist([hsv_image], [0], None, [180], [0, 180]), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_s = cv2.normalize(cv2.calcHist([hsv_image], [1], None, [256], [0, 256]), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_v = cv2.normalize(cv2.calcHist([hsv_image], [2], None, [256], [0, 256]), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        h_hists.append(hist_h)
        s_hists.append(hist_s)
        v_hists.append(hist_v)
    if not h_hists: return None, None, None
    # Return the flattened average of the histograms
    return (np.mean(h_hists, axis=0).flatten(), 
            np.mean(s_hists, axis=0).flatten(), 
            np.mean(v_hists, axis=0).flatten())

def compute_fft(histogram_data):
    """Computes the magnitude of the FFT for a histogram."""
    return np.abs(np.fft.fft(histogram_data))

def prepare_for_frechet(data):
    """Converts a 1D array into a 2D curve for Fréchet distance calculation."""
    return np.column_stack((np.arange(len(data)), data))
