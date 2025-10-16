import os
import yaml
import glob
import pandas as pd
import seaborn as sns
from ultralytics import YOLO
import matplotlib.pyplot as plt

from tqdm import tqdm

def analyze_yolo_outputs(model_path, data_yaml_path):
    """
    Loads a YOLO model, runs inference on the test dataset specified
    in the local data.yaml, and generates analysis plots.
    """
    # 1. Load the YOLO model
    print(f"🚀 Loading model from '{model_path}'...")
    model = YOLO(model_path)

    # 2. Find the test dataset path from the local data.yaml file
    print(f"📚 Reading dataset configuration from '{data_yaml_path}'...")
    try:
        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
        
        # The data.yaml path is relative to the YAML file's location
        dataset_root = os.path.dirname(data_yaml_path)
        test_images_dir = os.path.join(dataset_root, data_yaml['test'])
        
        image_files = glob.glob(os.path.join(test_images_dir, '*.*'))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print(f"❌ Error: No images found in the directory specified for 'test' in your YAML: '{test_images_dir}'")
            return
            
    except Exception as e:
        print(f"Error reading YAML file or finding images: {e}")
        return

    # 3. Process all images and collect detection data
    print(f"🔎 Found {len(image_files)} images. Running inference...")
    all_detections = []
    for image_path in tqdm(image_files):
        results = model(image_path, verbose=False)
        for box in results[0].boxes:
            x_center, y_center, width, height = box.xywhn[0].tolist()
            confidence = box.conf[0].item()
            all_detections.append({
                'x_center': x_center, 'y_center': y_center,
                'width': width, 'height': height,
                'confidence': confidence
            })

    if not all_detections:
        print("🤷 No detections were made on the test set. Cannot generate analysis.")
        return

    print(f"✅ Inference complete. Found a total of {len(all_detections)} detections.")
    df = pd.DataFrame(all_detections)

    # 4. Generate and display the plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('YOLO Model Performance Analysis', fontsize=16)

    # Plot 1: Bbox Distribution Heatmap
    sns.kdeplot(x=df['x_center'], y=df['y_center'], fill=True, cmap="viridis", ax=axes[0])
    axes[0].set_title('Distribution of BBox Centers')
    axes[0].set_xlabel('Normalized X-coordinate')
    axes[0].set_ylabel('Normalized Y-coordinate')
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()

    # Plot 2: Correlation Matrix
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1])
    axes[1].set_title('Correlation Matrix of Detection Properties')

    # Plot 3: Confidence Score Distribution
    sns.histplot(df['confidence'], bins=25, kde=True, ax=axes[2])
    axes[2].set_title('Confidence Score Distribution')
    axes[2].set_xlabel('Confidence Score')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    # --- ⚠️ ACTION REQUIRED: Set the paths to your local model and dataset YAML file ---
    MODEL_WEIGHTS_PATH = "best.pt"
    DATASET_YAML_PATH = "roboflow_dataset/data.yaml"

    # --- Sanity checks before running ---
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"❌ Error: Model file not found at '{MODEL_WEIGHTS_PATH}'")
    elif not os.path.exists(DATASET_YAML_PATH):
        print(f"❌ Error: Dataset YAML file not found at '{DATASET_YAML_PATH}'")
    else:
        analyze_yolo_outputs(MODEL_WEIGHTS_PATH, DATASET_YAML_PATH)