# USAGE
# python CharacterClassifier_Modified.py

##### PYTHON PACKAGES
# Generic
import pickle
import numpy as np
import pandas as pd
import os
import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import copy # Used for deep copying image data

# External Libraries for Image Transformation (Crucial for adding noise)
# We will use simple NumPy operations here, assuming the original 'images'
# are in a format that NumPy can handle (e.g., NumPy arrays representing pixels).

# Classifiers
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)
# Assuming these are available in your local environment
from Descriptors.blockbinarypixelsum import FeatureBlockBinaryPixelSum
from Descriptors.lbp import FeatureLBP
from Descriptors.hog import FeatureHOG

#### EXP-SET UP
# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
DataDir='example_fonts'

# Load Font DataSets
fileout=os.path.join(DataDir,'alphabetIms')+'.pkl'
f=open(fileout, 'rb')
data=pickle.load(f)
f.close()
alphabetIms=data['alphabetIms']
alphabetLabels=np.array(data['alphabetLabels'])

fileout=os.path.join(DataDir,'digitsIms')+'.pkl'
f=open(fileout, 'rb')
data=pickle.load(f)
f.close()
digitsIms=data['digitsIms']
digitsLabels=np.array(data['digitsLabels'])

print("Data Loaded:")
print(f"Alphabet Images: {len(alphabetIms)}")
print(f"Digit Images: {len(digitsIms)}")

# --- VOWEL REMOVAL LOGIC ---
VOWELS = np.array(['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u'])

SCALE = 2

# Create a boolean mask to identify and exclude vowel labels
consonant_mask = ~np.isin(alphabetLabels, VOWELS)

# Filter the labels and images to keep only consonants
alphabetLabels_consonants = alphabetLabels[consonant_mask]
# Need to convert the list of images to a NumPy array to apply the mask
alphabetIms_consonants = np.array(alphabetIms, dtype=object)[consonant_mask]

print(f"Original Alphabet size: {len(alphabetLabels)}")
print(f"Consonant-only Alphabet size: {len(alphabetLabels_consonants)}")
# ---------------------------

# --- AUGMENTATION FUNCTIONALITY (Unchanged) ---

def apply_gaussian_noise(image_data, scale=10):
    """
    Applies simple Gaussian (Normal) noise to the image data.
    Assumes image_data is a NumPy array of pixel values.
    """
    if not isinstance(image_data, np.ndarray):
        try:
            image_data = np.array(image_data, dtype=np.float32)
        except:
            return image_data

    # Generate random noise with mean 0 and standard deviation (scale)
    noise = np.random.normal(0, scale, image_data.shape)
    
    # Add noise, clip values to be within a valid range
    noisy_image = image_data + noise
    
    # Clipping for robustness
    # The original images are likely binary (0/1) or 0-255 grayscale.
    # We check the max value to decide the clipping range.
    max_val = np.max(image_data) if image_data.size > 0 else 1.0 # Handle empty array edge case
    clip_max = 255 if max_val > 1.0 else 1.0

    noisy_image = np.clip(noisy_image, 0, clip_max) 
    
    return noisy_image.astype(image_data.dtype) # Return in original data type

def augment_test_data(X_test_ims, y_test, num_duplicates=3, scale=10):
    """
    Duplicates test images and applies Gaussian noise to the duplicates.
    
    Returns: X_test_aug_ims (list of augmented images), y_test_aug (list of corresponding labels)
    """
    X_test_aug_ims = list(X_test_ims) # Start with original test set images
    y_test_aug = list(y_test)
    
    for _ in range(num_duplicates):
        for img, label in zip(X_test_ims, y_test):
            # 1. Apply Transformation (Gaussian Noise)
            transformed_img = apply_gaussian_noise(img, scale=scale)
            
            X_test_aug_ims.append(transformed_img)
            y_test_aug.append(label)
            
    print(f"Test support increased from {len(X_test_ims)} to {len(X_test_aug_ims)} (x{num_duplicates + 1} factor).")
    
    return np.array(X_test_aug_ims, dtype=object), np.array(y_test_aug)

# --- FEATURE EXTRACTION SETUP (Initialized inside the loop) ---

# initialize descriptors
descBlckAvg = FeatureBlockBinaryPixelSum()
descHOG = FeatureHOG()
descLBP = FeatureLBP()

def extract_features_from_images(image_list, desc_blck_avg, desc_hog, desc_lbp):
    """Extracts all feature types from a list of images."""
    features = {
        'BLCK_AVG': [],
        'HOG': [],
        'LBP': []
    }
    for roi in image_list:
        features['BLCK_AVG'].append(desc_blck_avg.extract_image_features(roi))
        features['HOG'].append(desc_hog.extract_image_features(roi))
        features['LBP'].append(desc_lbp.extract_image_features(roi))
    
    # Convert lists of features to NumPy arrays
    return {k: np.array(v) for k, v in features.items()}


### CLASSIFICATION (MODIFIED TO DO SPLIT -> AUGMENT -> FEATURE EXTRACTION)
classifiers = {
    "SVM": LinearSVC(random_state=42, max_iter=5000),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(random_state=42, max_iter=5000)
}

recall_results_digits = {feature_name: {} for feature_name in ['BLCK_AVG', 'HOG', 'LBP']}
recall_results_alpha = {feature_name: {} for feature_name in ['BLCK_AVG', 'HOG', 'LBP']}


# --- Classification for ALPHABET (CONSONANTS ONLY) ---
print("\n" + "="*50)
print("CLASSIFICATION FOR ALPHABET (CONSONANTS ONLY)")
print("="*50)

# 1. Split the raw IMAGE data (alphabetIms_consonants) and labels
X_alpha_ims_train, X_alpha_ims_test_orig, y_alpha_train, y_alpha_test_orig = train_test_split(
    alphabetIms_consonants, alphabetLabels_consonants, 
    test_size=0.25, random_state=42, stratify=alphabetLabels_consonants
)

# 2. Augment the raw test images (using Gaussian noise)
# NOTE: The augmentation parameters are set to match the Digits section for consistency, 
# although the original request suggested *skipping* augmentation for Alphabet due to the former structure.
X_alpha_ims_test_aug, y_alpha_test_aug = augment_test_data(
    X_alpha_ims_test_orig, y_alpha_test_orig, num_duplicates=1, scale=SCALE
)

# 3. Extract features from training and augmented test images
print("\nExtracting Features for Alphabet Consonants...")
train_features_alpha = extract_features_from_images(X_alpha_ims_train, descBlckAvg, descHOG, descLBP)
test_features_alpha = extract_features_from_images(X_alpha_ims_test_aug, descBlckAvg, descHOG, descLBP)
print("Feature Extraction Complete.")


# 4. Train and Evaluate
for feature_name in train_features_alpha.keys():
    print(f"\n--- Classification for ALPHABET CONSONANTS with {feature_name} features (Augmented Test) ---")
    
    X_train = train_features_alpha[feature_name]
    y_train = y_alpha_train
    X_test = test_features_alpha[feature_name]
    y_test = y_alpha_test_aug
    
    for clf_name, clf in classifiers.items():
        print(f"Training {clf_name}...")
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0) 

        macro_recall = report['macro avg']['recall']
        recall_results_alpha[feature_name][clf_name] = macro_recall
        
        print(f"Results for {clf_name}:")
        print(classification_report(y_test, predictions, zero_division=0))
        print("-" * 30)

# --- Classification for DIGITS ---
print("\n" + "="*50)
print("CLASSIFICATION FOR DIGITS")
print("="*50)

# 1. Split the raw IMAGE data (digitsIms) and labels
X_digits_ims_train, X_digits_ims_test_orig, y_digits_train, y_digits_test_orig = train_test_split(
    digitsIms, digitsLabels, 
    test_size=0.25, random_state=42, stratify=digitsLabels
)

# 2. Augment the raw test images (using Gaussian noise)
X_digits_ims_test_aug, y_digits_test_aug = augment_test_data(
    X_digits_ims_test_orig, y_digits_test_orig, num_duplicates=1, scale=SCALE
)

# 3. Extract features from training and augmented test images
print("\nExtracting Features for Digits...")
train_features_digits = extract_features_from_images(X_digits_ims_train, descBlckAvg, descHOG, descLBP)
test_features_digits = extract_features_from_images(X_digits_ims_test_aug, descBlckAvg, descHOG, descLBP)
print("Feature Extraction Complete.")


# 4. Train and Evaluate
for feature_name in train_features_digits.keys():
    print(f"\n--- Classification for DIGITS with {feature_name} features (Augmented Test) ---")
    
    X_train = train_features_digits[feature_name]
    y_train = y_digits_train
    X_test = test_features_digits[feature_name]
    y_test = y_digits_test_aug
    
    for clf_name, clf in classifiers.items():
        print(f"Training {clf_name}...")
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test) # Predict on augmented test set

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0) 

        macro_recall = report['macro avg']['recall']
        recall_results_digits[feature_name][clf_name] = macro_recall

        print(f"Results for {clf_name}:")
        print(classification_report(y_test, predictions, zero_division=0))
        print("-" * 30)


# Convert the results dictionary to a DataFrame for plotting
df_recall_digits = pd.DataFrame(recall_results_digits).T
df_recall_alpha = pd.DataFrame(recall_results_alpha).T 

# Print the gathered data before plotting
print("\n--- Gathered Macro Average Recall Results (Digits - Properly Augmented Test) ---")
print(df_recall_digits.to_markdown(floatfmt=".4f"))

print("\n--- Gathered Macro Average Recall Results (Alphabet Consonants - Properly Augmented Test) ---")
print(df_recall_alpha.to_markdown(floatfmt=".4f"))

# --- PLOTTING CODE (Unchanged) ---

# 1. Determine the global minimum and maximum recall values across both DataFrames
global_vmin = min(df_recall_digits.min().min(), df_recall_alpha.min().min())
global_vmax = max(df_recall_digits.max().max(), df_recall_alpha.max().max())
global_vmax = min(1.0, global_vmax + 0.01) 
global_vmin = max(0.0, global_vmin - 0.01) 

plt.figure(figsize=(14, 6))

# Subplot 1: Digits
ax1 = plt.subplot(1, 2, 1)

sns.heatmap(
    df_recall_digits,
    ax=ax1,
    annot=True,
    fmt=".4f",
    cmap="viridis",
    linewidths=.5,
    cbar_kws={'label': 'Macro Average Recall'},
    vmin=global_vmin,
    vmax=global_vmax
)

ax1.set_title('Macro Average Recall Heatmap (Digits - Properly Augmented Test)')
ax1.set_xlabel('Classifier Type (X)')
ax1.set_ylabel('Feature Type (Y)')

# Subplot 2: Alphabet Consonants
ax2 = plt.subplot(1, 2, 2)

sns.heatmap(
    df_recall_alpha,
    ax=ax2,
    annot=True,
    fmt=".4f",
    cmap="viridis",
    linewidths=.5,
    cbar_kws={'label': 'Macro Average Recall'},
    vmin=global_vmin,
    vmax=global_vmax
)

ax2.set_title('Macro Average Recall Heatmap (Alphabet Consonants - Properly Augmented Test)')
ax2.set_xlabel('Classifier Type (X)')
ax2.set_ylabel('Feature Type (Y)')

plt.tight_layout()
plt.show()