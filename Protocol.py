import os
import cv2
import glob
import numpy as np
import random

from LicensePlateDetector import detectPlates
from DataExploration import computeProperties


# =========================
# NORMALIZATION FUNCTIONS
# =========================

def normalize_plate_area(image, plate_box, target_rel_area):
    rect = cv2.minAreaRect(plate_box.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    img_h, img_w = image.shape[:2]
    current_rel_area = (w * h) / float(img_w * img_h)

    if current_rel_area <= 0:
        return image

    scale_factor = np.sqrt(target_rel_area / current_rel_area)
    new_w = int(img_w * scale_factor)
    new_h = int(img_h * scale_factor)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if new_w >= img_w and new_h >= img_h:
        sx = (new_w - img_w) // 2
        sy = (new_h - img_h) // 2
        out = resized[sy:sy + img_h, sx:sx + img_w]
    else:
        out = cv2.copyMakeBorder(
            resized,
            (img_h - new_h) // 2, (img_h - new_h + 1) // 2,
            (img_w - new_w) // 2, (img_w - new_w + 1) // 2,
            cv2.BORDER_REFLECT
        )
    return out


def normalize_plate_angle(image, plate_box, target_angle):
    rect = cv2.minAreaRect(plate_box.astype(np.float32))
    (_, _), (_, _), angle = rect
    delta = target_angle - angle

    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), delta, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated


def normalize_hsv(image, target_means):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    h_m, s_m, v_m = np.mean(h), np.mean(s), np.mean(v)
    t_h, t_s, t_v = target_means

    h = (h - h_m) + t_h
    s = (s / (s_m + 1e-6)) * t_s
    v = (v / (v_m + 1e-6)) * t_v

    hsv_new = cv2.merge([
        np.clip(h % 180, 0, 179),
        np.clip(s, 0, 255),
        np.clip(v, 0, 255)
    ]).astype(np.uint8)

    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)


# =========================
# SAMPLING & TARGETS
# =========================

def build_target_distributions(real_plateArea, real_plateAngle,
                               real_color, real_sat, real_val, Views):
    targets = {}
    for v in Views:
        targets[v] = {
            # keep absolute relative areas
            "rel_area": np.array(real_plateArea[v]),  
            "angle": np.array(real_plateAngle[v]),
            "hsv_means": list(zip(real_color[v], real_sat[v], real_val[v]))
        }
    return targets


# =========================
# DATASET ADJUSTMENT
# =========================

def adjust_dataset(data_dir, views, out_dir, targets):
    os.makedirs(out_dir, exist_ok=True)

    for v in views:
        os.makedirs(os.path.join(out_dir, v), exist_ok=True)
        img_files = sorted(glob.glob(os.path.join(data_dir, v, "*.jpg")))
        print(f"Processing {len(img_files)} images for view {v}")

        for img_path in img_files:
            image = cv2.imread(img_path)
            if image is None:
                continue

            regions, _ = detectPlates(image)
            if not regions:
                continue

            plate_box = np.array(regions[0], dtype=np.float32)

            # ---- sample targets ----
            t_rel_area = np.random.choice(targets[v]["rel_area"])
            t_angle = np.random.choice(targets[v]["angle"])
            t_hsv = random.choice(targets[v]["hsv_means"])

            # ---- apply transforms ----
            image = normalize_plate_area(image, plate_box, t_rel_area)
            image = normalize_plate_angle(image, plate_box, t_angle)
            image = normalize_hsv(image, t_hsv)

            # ---- save ----
            fname = os.path.basename(img_path)
            out_path = os.path.join(out_dir, v, fname)
            cv2.imwrite(out_path, image)


# =========================
# DETECTION EVALUATION
# =========================

def evaluate_detection(dataset_dir, views):
    """Compute detection accuracy for dataset (no plots)."""
    results = {}
    for v in views:
        folder = os.path.join(dataset_dir, v)
        if not os.path.isdir(folder):
            results[v] = (0, 0, 0.0)  # total, detected, accuracy
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        total, detected = len(files), 0
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            plates, _ = detectPlates(img)
            if plates:
                detected += 1
        acc = (detected / total * 100) if total > 0 else 0.0
        results[v] = (total, detected, acc)
    return results


# =========================
# MAIN PIPELINE
# =========================

def main():
    # --- Dataset paths ---
    Real_DataDir = r"data"
    Own_DataDir = r"data/Patentes"
    Augmented_DataDir = r"data/Patentes"

    Views = ["Frontal", "Lateral"]
    Views_A = ["FrontalAugmented", "LateralAugmented"]

    datasets = {
        "Real_Plates": (Real_DataDir, Views),
        "New_Plates": (Own_DataDir, Views),
        "NewAugmented_Plates": (Augmented_DataDir, Views_A),
    }

    # --- Compute distributions from Real dataset ---
    R_area, R_angle, R_color, R_sat, R_val = computeProperties(Real_DataDir, Views)
    targets = build_target_distributions(R_area, R_angle, R_color, R_sat, R_val, Views)

    # --- Adjust and evaluate ---
    for name, (data_dir, views) in datasets.items():
        if name == "Real_Plates":
            continue  # do not adjust Real

        out_dir = os.path.join("data", f"Adjusted_{name}")
        print(f"\n>>> Adjusting {name} -> {out_dir}")
        adjust_dataset(data_dir, views, out_dir, targets)

        # --- Detection evaluation ---
        results = evaluate_detection(out_dir, Views)
        print(f"\nDetection results for {name}:")
        for v, (total, detected, acc) in results.items():
            print(f"  {v}: {detected}/{total} detected ({acc:.2f}%)")


if __name__ == "__main__":
    main()
