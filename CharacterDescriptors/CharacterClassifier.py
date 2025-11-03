# USAGE
# python CharacterClassifier.py

##### PYTHON PACKAGES
# Generic
import pickle
import numpy as np
import pandas as pd
import os
import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# Classifiers
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)
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

# --- VOWEL REMOVAL LOGIC ---
VOWELS = np.array(['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u'])

# Create a boolean mask to identify and exclude vowel labels
consonant_mask = ~np.isin(alphabetLabels, VOWELS)

# Filter the labels and images to keep only consonants
alphabetLabels_consonants = alphabetLabels[consonant_mask]
# Need to convert the list of images to a NumPy array to apply the mask
alphabetIms_consonants = np.array(alphabetIms, dtype=object)[consonant_mask] 

print(f"Original Alphabet size: {len(alphabetLabels)}")
print(f"Consonant-only Alphabet size: {len(alphabetLabels_consonants)}")
# ---------------------------


digitsFeat={}
digitsFeat['BLCK_AVG']=[]
digitsFeat['HOG'] = []

alphabetFeat={}
alphabetFeat['BLCK_AVG']=[]
alphabetFeat['HOG'] = []

# initialize descriptors
descBlckAvg = FeatureBlockBinaryPixelSum()
descHOG = FeatureHOG()

### EXTRACT FEATURES
# Digits (unchanged)
for roi in digitsIms:
     # extract features
     digitsFeat['BLCK_AVG'].append(descBlckAvg.extract_image_features(roi))
     digitsFeat['HOG'].append(descHOG.extract_image_features(roi))
     
# Alphabet (using filtered consonant data)
for roi in alphabetIms_consonants:
    # extract features
    alphabetFeat['BLCK_AVG'].append(descBlckAvg.extract_image_features(roi))
    alphabetFeat['HOG'].append(descHOG.extract_image_features(roi))


### CLASSIFICATION
classifiers = {
    "SVM": LinearSVC(random_state=42, max_iter=5000),
    "MLP": MLPClassifier(random_state=42, max_iter=5000)
}

recall_results_digits = {feature_name: {} for feature_name in digitsFeat.keys()}
recall_results_alpha = {feature_name: {} for feature_name in alphabetFeat.keys()}


# Classification for ALPHABET (CONSONANTS ONLY)
for feature_name, features in alphabetFeat.items():
    print(f"--- Classification for ALPHABET CONSONANTS with {feature_name} features ---")
    
    X = np.array(features)
    y = alphabetLabels_consonants # <--- USE FILTERED LABELS
    
    # Use stratification to maintain class proportions in train/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    for clf_name, clf in classifiers.items():
        print(f"Training {clf_name}...")
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)

        # Ensure zero_division=0 to handle cases where a class has no samples in test set
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0) 

        macro_recall = report['macro avg']['recall']
        recall_results_alpha[feature_name][clf_name] = macro_recall
        
        print(f"Results for {clf_name}:")
        print(classification_report(y_test, predictions, zero_division=0))
        print("-" * 30)

# Classification for DIGITS (unchanged)
for feature_name, features in digitsFeat.items():
    print(f"--- Classification for DIGITS with {feature_name} features ---")
    
    X = np.array(features)
    y = digitsLabels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    for clf_name, clf in classifiers.items():
        print(f"Training {clf_name}...")
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0) 

        macro_recall = report['macro avg']['recall']
        recall_results_digits[feature_name][clf_name] = macro_recall

        print(f"Results for {clf_name}:")
        print(classification_report(y_test, predictions, zero_division=0))
        print("-" * 30)

# Convert the results dictionary to a DataFrame for plotting
df_recall_digits = pd.DataFrame(recall_results_digits).T
df_recall_alpha = pd.DataFrame(recall_results_alpha).T # This now contains only consonant results

# Print the gathered data before plotting
print("\n--- Gathered Macro Average Recall Results (Digits) ---")
print(df_recall_digits.to_markdown(floatfmt=".4f"))

print("\n--- Gathered Macro Average Recall Results (Alphabet Consonants) ---")
print(df_recall_alpha.to_markdown(floatfmt=".4f"))

# --- PLOTTING CODE (FIXED for subplot assignment and matching color maps) ---

# 1. Determine the global minimum and maximum recall values across both DataFrames
global_vmin = min(df_recall_digits.min().min(), df_recall_alpha.min().min())
global_vmax = max(df_recall_digits.max().max(), df_recall_alpha.max().max())
global_vmax = min(1.0, global_vmax + 0.01) # Cap max at 1.0 and add buffer if needed
global_vmin = max(0.0, global_vmin - 0.01) # Ensure min is not less than 0

plt.figure(figsize=(14, 6))

# Subplot 1: Digits
ax1 = plt.subplot(1, 2, 1)

sns.heatmap(
    df_recall_digits,
    ax=ax1,              # <-- FIX: Explicitly assign to ax1
    annot=True,          
    fmt=".4f",           
    cmap="viridis",      # <-- FIX: Use same cmap as ax2
    linewidths=.5,
    cbar_kws={'label': 'Macro Average Recall'},
    vmin=global_vmin,    # <-- FIX: Set common min
    vmax=global_vmax     # <-- FIX: Set common max
)

ax1.set_title('Macro Average Recall Heatmap (Digits)')
ax1.set_xlabel('Classifier Type (X)')
ax1.set_ylabel('Feature Type (Y)')

# Subplot 2: Alphabet Consonants
ax2 = plt.subplot(1, 2, 2)

sns.heatmap(
    df_recall_alpha,
    ax=ax2,              # <-- FIX: Explicitly assign to ax2
    annot=True,          
    fmt=".4f",           
    cmap="viridis",      # <-- FIX: Use same cmap as ax1
    linewidths=.5,
    cbar_kws={'label': 'Macro Average Recall'}, 
    vmin=global_vmin,    # <-- FIX: Set common min
    vmax=global_vmax     # <-- FIX: Set common max
)

ax2.set_title('Macro Average Recall Heatmap (Alphabet Consonants)') # Updated title
ax2.set_xlabel('Classifier Type (X)')
ax2.set_ylabel('Feature Type (Y)')

plt.tight_layout()
plt.show()
