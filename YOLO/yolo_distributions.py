import argparse
import os
from glob import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

def analyze_yolo_output(image_folder: str, yolo_model_path: str, output_folder: str):
    """
    Runs YOLO inference over a folder of images, analyzes the detections,
    and visualizes the confidence score distribution per class.

    Args:
        image_folder (str): Path to the folder containing images.
        yolo_model_path (str): Path to the YOLO model file (.pt).
    """
    # --- 1. Model Loading and Setup ---
    print(f"Loading YOLO model from: {yolo_model_path}")
    try:
        model = YOLO(yolo_model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 2. Image Collection ---
    # Supported image extensions (can be extended)
    image_extensions = ['jpg', 'jpeg', 'png', 'webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(image_folder, f'*.{ext}')))

    if not image_paths:
        print(f"No images found in folder: {image_folder}. Check path and file types.")
        return

    print(f"Found {len(image_paths)} images to process.")

    # --- 3. Inference and Data Collection ---
    all_confidences = {}
    object_counts = {}

    # Run inference in a batch for efficiency
    print("Running inference...")
    results = model(image_paths, verbose=False, stream=False)

    for result in results:
        # result.boxes is an Boxes object containing detection data
        if result.boxes:
            # Get confidence scores (conf) and class IDs (cls) as numpy arrays
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for conf, cls_id in zip(confidences, class_ids):
                # Class name is needed for clear output and plotting
                class_name = model.names[cls_id]

                # Store confidence scores for plotting
                if class_name not in all_confidences:
                    all_confidences[class_name] = []
                all_confidences[class_name].append(conf)

                # Count the total number of detections for distribution
                object_counts[class_name] = object_counts.get(class_name, 0) + 1

    print("Inference complete.")

    # --- 4. Distribution Output ---
    print("\n" + "="*40)
    print("Detected Object Distribution")
    print("="*40)
    total_detections = sum(object_counts.values())
    if total_detections == 0:
        print("No objects were detected across all images.")
        return

    for class_name, count in sorted(object_counts.items(), key=lambda item: item[1], reverse=True):
        percentage = (count / total_detections) * 100
        print(f"| {class_name:<20} | Count: {count:<6} | Percentage: {percentage:.2f}% |")
    print("="*40)
    print(f"Total Detections: {total_detections}")
    print("="*40)

    # --- 5. Histogram Plotting ---
    print("\nGenerating confidence score histograms...")

    plt.figure(figsize=(12, 8))
    
    # We will plot multiple histograms on the same axes.
    # The 'histtype="stepfilled"' and 'alpha' arguments help distinguish them.
    for class_name, confs in all_confidences.items():
        # Plot the histogram for the current class
        plt.hist(
            confs, 
            bins=np.linspace(0, 1, 51),  # Bins from 0.0 to 1.0 in steps of 0.05
            label=f'{class_name} (n={len(confs)})',
            density=False, # Show raw counts on the Y-axis
            alpha=0.6,
            histtype='stepfilled',
            edgecolor='black'
        )

    yolo_model_name = os.path.basename(yolo_model_path)
    plt.title(f"Confidence Scores {yolo_model_name}")
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Detections')
    plt.xlim(0, 1) # Confidence scores range from 0 to 1
    plt.grid(axis='y', alpha=0.75)
    plt.legend(title='Class')
    plt.tight_layout() # Adjust plot to prevent labels from overlapping
    plt.savefig(os.path.join(output_folder, f"confidence_{yolo_model_name}.png"))

def main():
    """Parses command-line arguments and calls the analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze YOLO model output over a folder of images and plot confidence score distributions."
    )
    
    parser.add_argument(
        '--image_folder', 
        type=str, 
        required=True, 
        help="Path to the folder containing images for inference."
    )
    
    parser.add_argument(
        '--yolo_model', 
        type=str, 
        required=True, 
        help="Path to the YOLO model file (e.g., yolov8n.pt)."
    )

    parser.add_argument(
        '--output_folder', 
        type=str, 
        required=True, 
        help="Path where to save the plots"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.image_folder):
        print(f"Error: Image folder not found at '{args.image_folder}'")
        return

    if not os.path.isfile(args.yolo_model):
        print(f"Error: YOLO model file not found at '{args.yolo_model}'")
        return

    analyze_yolo_output(args.image_folder, args.yolo_model, args.output_folder)

if __name__ == "__main__":
    main()
