# USAGE
# python CharacterClassifier_Ensemble_Bootstrap.py

##### PYTHON PACKAGES
# Generic
import pickle
import numpy as np
import pandas as pd
import os
import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # For stats.sem and calculate_ci

# Classifiers
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV # To get probabilities for LinearSVC/AUC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, recall_score

# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)
from Descriptors.blockbinarypixelsum import FeatureBlockBinaryPixelSum
from Descriptors.hog import FeatureHOG

# --- Helper Functions ---

def calculate_ci(data):
    """Calculates the 95% Confidence Interval for the mean of the data."""
    if len(data) < 2:
        return (np.mean(data) if data else 0, np.mean(data) if data else 0)
    return stats.t.interval(
        confidence=0.95, 
        df=len(data) - 1, 
        loc=np.mean(data), 
        scale=stats.sem(data)
    )

def majority_vote(preds_1, preds_2):
    """Combines two arrays of predictions (strings) using mode (Majority Voting)."""
    combined = np.stack((preds_1, preds_2), axis=1)
    
    ensemble_preds = []
    for row in combined:
        unique, counts = np.unique(row, return_counts=True)
        # Find the index of the largest count (mode)
        mode_index = np.argmax(counts)
        ensemble_preds.append(unique[mode_index])
        
    return np.array(ensemble_preds)


#### EXP-SET UP
# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
DataDir='example_fonts'
NTrial = 20 # Requested number of bootstrap trials

# --- DATA LOADING AND PREPROCESSING ---
try:
    fileout=os.path.join(DataDir,'alphabetIms')+'.pkl'
    with open(fileout, 'rb') as f:
        data=pickle.load(f)
        alphabetIms=data['alphabetIms']
        alphabetLabels=np.array(data['alphabetLabels'])

    fileout=os.path.join(DataDir,'digitsIms')+'.pkl'
    with open(fileout, 'rb') as f:
        data=pickle.load(f)
        digitsIms=data['digitsIms']
        digitsLabels=np.array(data['digitsLabels'])
except FileNotFoundError:
    print(f"ERROR: Data files not found in {DataDir}. Check your DataDir path.")
    exit()

# VOWEL REMOVAL LOGIC
VOWELS = np.array(['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u'])
consonant_mask = ~np.isin(alphabetLabels, VOWELS)
alphabetLabels_consonants = alphabetLabels[consonant_mask]
alphabetIms_consonants = np.array(alphabetIms, dtype=object)[consonant_mask]

# initialize descriptors
descBlckAvg = FeatureBlockBinaryPixelSum()
descHOG = FeatureHOG()

# --- FEATURE EXTRACTION ---
digitsFeat={'BLCK_AVG': [], 'HOG': []}
alphabetFeat={'BLCK_AVG': [], 'HOG': []}

for roi in digitsIms:
     digitsFeat['BLCK_AVG'].append(descBlckAvg.extract_image_features(roi))
     digitsFeat['HOG'].append(descHOG.extract_image_features(roi))
for roi in alphabetIms_consonants:
     alphabetFeat['BLCK_AVG'].append(descBlckAvg.extract_image_features(roi))
     alphabetFeat['HOG'].append(descHOG.extract_image_features(roi))

# Convert lists to NumPy arrays
for key in digitsFeat:
    digitsFeat[key] = np.array(digitsFeat[key])
    alphabetFeat[key] = np.array(alphabetFeat[key])

# --- COMBINE DATASETS ---
# Multi-class labels are the original character/digit strings (31 classes total)
X_full = {}
y_full = np.concatenate((digitsLabels, alphabetLabels_consonants))
ALL_CLASSES = np.unique(y_full)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}

for feature_name in ['BLCK_AVG', 'HOG']:
    X_full[feature_name] = np.concatenate((digitsFeat[feature_name], alphabetFeat[feature_name]))
    
print(f"Total Combined Classes: {len(ALL_CLASSES)}")
print(f"Combined Dataset size: {X_full['BLCK_AVG'].shape[0]} samples.")


# --- BOOTSTRAPPING EXECUTION: ENSEMBLE ---

all_ensemble_results = []
classifiers_base = {
    "SVM": LinearSVC(random_state=42, max_iter=5000),
    "MLP": MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50,), random_state=1, max_iter=5000)
}

print(f"\nStarting {NTrial} bootstrap trials for Multi-Class Ensembles...")

for i in range(NTrial):
    # Split Data: We use the indices for a consistent train/test split across features
    indices = np.arange(X_full['BLCK_AVG'].shape[0])
    train_indices, test_indices, _, y_test_true = train_test_split(
        indices, y_full, test_size=0.25, shuffle=True, stratify=y_full, random_state=i
    )
    y_test_true = y_full[test_indices] # The true labels (strings)

    for feature_name in ['BLCK_AVG', 'HOG']:
        X_feat = X_full[feature_name]
        X_train, X_test = X_feat[train_indices], X_feat[test_indices]
        y_train = y_full[train_indices]

        # 1. Train and Predict Individual Models
        
        # --- SVM ---
        # Wrap SVM in CalibratedClassifierCV to get probability-like scores for AUC/ROC
        # This is strictly for the AUC metric, classification relies on decision function.
        M_svm = CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=5000), cv=3, method='isotonic', n_jobs=-1)
        M_svm.fit(X_train, y_train)
        svm_preds = M_svm.predict(X_test)
        
        # For multi-class AUC, we calculate One-vs-Rest (OvR) AUC. Requires numeric labels.
        y_test_numeric = np.array([CLASS_TO_IDX[c] for c in y_test_true])
        
        # Get decision scores for AUC (OvR)
        try:
             svm_scores = M_svm.predict_proba(X_test)
             auc_svm = roc_auc_score(y_test_numeric, svm_scores, multi_class='ovr', average='macro', labels=np.arange(len(ALL_CLASSES)))
        except ValueError:
             # Handle rare cases where probability calibration fails or test set is too small
             auc_svm = 0.0

        # --- MLP ---
        M_mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50,), random_state=1, max_iter=5000)
        M_mlp.fit(X_train, y_train)
        mlp_preds = M_mlp.predict(X_test)

        # Get probability scores for AUC (OvR)
        mlp_scores = M_mlp.predict_proba(X_test)
        auc_mlp = roc_auc_score(y_test_numeric, mlp_scores, multi_class='ovr', average='macro', labels=np.arange(len(ALL_CLASSES)))


        # 2. Ensemble Prediction (Majority Vote on Class Labels)
        ensemble_preds = majority_vote(svm_preds, mlp_preds)
        
        # 3. Ensemble AUC (Simple averaging of prediction scores is common, but complex for combined SVM/MLP. 
        # For simplicity and robust code, we will report AUC for individual models only, as ensemble AUC requires probability combination).
        # We will report the average AUC of the two components for the ensemble entry.
        auc_ensemble = (auc_svm + auc_mlp) / 2
        
        # 4. Evaluation Metrics
        
        # Macro Recall
        recall_svm = recall_score(y_test_true, svm_preds, average='macro', zero_division=0)
        recall_mlp = recall_score(y_test_true, mlp_preds, average='macro', zero_division=0)
        recall_ensemble = recall_score(y_test_true, ensemble_preds, average='macro', zero_division=0)
        
        # Store results for the two ensembles requested:
        all_ensemble_results.append({
            'Metric': 'Macro Recall',
            'Score': recall_ensemble,
            'Classifier': f'Ensemble (SVM+MLP) - {feature_name}'
        })
        all_ensemble_results.append({
            'Metric': 'Macro AUC',
            'Score': auc_ensemble,
            'Classifier': f'Ensemble (SVM+MLP) - {feature_name}'
        })
            
    print(f"Trial {i+1}/{NTrial} complete.")

# Convert results list to DataFrame
df_results_ensemble = pd.DataFrame(all_ensemble_results)


# --- SUMMARY AND PLOTTING ---
print("\n" + "="*50)
print(f"ENSEMBLE CLASSIFICATION RESULTS (31 Classes - {NTrial} Trials)")
print("="*50)

# Calculate Mean and CI for each metric/classifier combination
summary = df_results_ensemble.groupby(['Classifier', 'Metric']).agg(
    Mean_Score=('Score', 'mean'),
    Std_Err=('Score', stats.sem),
    N=('Score', 'count')
).reset_index()

# Calculate 95% Confidence Interval
summary['CI_Lower'] = summary.apply(lambda row: stats.t.interval(0.95, row['N'] - 1, loc=row['Mean_Score'], scale=row['Std_Err'])[0], axis=1)
summary['CI_Upper'] = summary.apply(lambda row: stats.t.interval(0.95, row['N'] - 1, loc=row['Mean_Score'], scale=row['Std_Err'])[1], axis=1)

print("\n--- Summary of Ensemble Performance (Mean ± 95% CI) ---")
summary_output = summary[['Classifier', 'Metric', 'Mean_Score', 'CI_Lower', 'CI_Upper']]
summary_output['Result'] = summary_output.apply(
    lambda row: f"{row['Mean_Score']:.4f} [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]", axis=1
)
print(summary_output[['Classifier', 'Metric', 'Result']].to_markdown(index=False))

# --- PLOTTING: BOXPLOTS of AUC and RECALL ---

plt.figure(figsize=(14, 6))

# Subplot 1: Macro Recall
plt.subplot(1, 2, 1)
sns.boxplot(
    x='Classifier', 
    y='Score', 
    data=df_results_ensemble[df_results_ensemble['Metric'] == 'Macro Recall'], 
    palette='Set1'
)
plt.title(f'Macro Recall Distribution ({NTrial} Trials)')
plt.ylabel('Macro Recall Score')
plt.xlabel('Ensemble (SVM+MLP)')
plt.ylim(df_results_ensemble['Score'].min() * 0.9, 1.0)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()

# Subplot 2: Macro AUC
plt.subplot(1, 2, 2)
sns.boxplot(
    x='Classifier', 
    y='Score', 
    data=df_results_ensemble[df_results_ensemble['Metric'] == 'Macro AUC'], 
    palette='Set1'
)
plt.title(f'Macro AUC Distribution ({NTrial} Trials)')
plt.ylabel('Macro AUC Score (OvR)')
plt.xlabel('Ensemble (SVM+MLP)')
plt.ylim(df_results_ensemble['Score'].min() * 0.9, 1.0)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()

plt.suptitle("Ensemble Performance Comparison (SVM+MLP) by Feature Set", fontsize=16, y=1.02)
plt.show()
print("Boxplot displayed.")