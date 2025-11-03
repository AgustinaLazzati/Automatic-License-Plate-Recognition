# USAGE
# python CharacterClassifier_HoG_Analysis.py

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

def soft_vote_predict(proba_svm, proba_mlp, classes):
    """
    Combines probability scores from two classifiers using equal-weighted averaging (Soft Voting).

    Returns:
        ensemble_preds (np.array): Hard class predictions (string labels).
        ensemble_scores (np.array): Averaged probability scores for AUC calculation.
    """
    # 1. Average the probabilities
    # Note: Assumes both proba arrays are aligned to the same class order (ALL_CLASSES)
    ensemble_scores = (proba_svm + proba_mlp) / 2
    
    # 2. Get the index of the highest probability
    best_class_indices = np.argmax(ensemble_scores, axis=1)
    
    # 3. Map the index back to the string label
    ensemble_preds = classes[best_class_indices]
    
    return ensemble_preds, ensemble_scores

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
NTrial = 30 # Requested number of bootstrap trials

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

# initialize descriptor
descHOG = FeatureHOG()

# --- FEATURE EXTRACTION (HoG ONLY) ---
digitsFeatHOG = []
alphabetFeatHOG = []

print("Extracting HoG features...")
for roi in digitsIms:
     digitsFeatHOG.append(descHOG.extract_image_features(roi))
for roi in alphabetIms_consonants:
     alphabetFeatHOG.append(descHOG.extract_image_features(roi))

# Convert lists to NumPy arrays
X_digits_HOG = np.array(digitsFeatHOG)
X_alpha_HOG = np.array(alphabetFeatHOG)

# --- COMBINE DATASETS ---
X_full_HOG = np.concatenate((X_digits_HOG, X_alpha_HOG))
y_full = np.concatenate((digitsLabels, alphabetLabels_consonants))
ALL_CLASSES = np.unique(y_full)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}
    
print(f"Total Combined Classes: {len(ALL_CLASSES)}")
print(f"Combined Dataset size: {X_full_HOG.shape[0]} samples.")


# --- BOOTSTRAPPING EXECUTION: HOG ANALYSIS ---

all_hog_results = []
classifiers_base = {
    "SVM": LinearSVC(random_state=42, max_iter=5000),
    "MLP": MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50,), random_state=1, max_iter=5000)
}

print(f"\nStarting {NTrial} bootstrap trials for HoG Feature Analysis (Soft/Hard Voting)...")

for i in range(NTrial):
    # Split Data
    indices = np.arange(X_full_HOG.shape[0])
    train_indices, test_indices, _, y_test_true = train_test_split(
        indices, y_full, test_size=0.25, shuffle=True, stratify=y_full, random_state=i
    )
    y_test_true = y_full[test_indices]
    X_train, X_test = X_full_HOG[train_indices], X_full_HOG[test_indices]
    y_train = y_full[train_indices]

    # Use numeric labels for AUC calculation
    y_test_numeric = np.array([CLASS_TO_IDX[c] for c in y_test_true])
    
    svm_preds, mlp_preds = None, None
    svm_scores, mlp_scores = None, None
    auc_svm, auc_mlp = 0.0, 0.0 # Initialize AUCs for Hard Ensemble calculation
    
    # Iterate through individual models
    for clf_name, clf_base in classifiers_base.items():
        
        # --- Train Individual Model ---
        
        if clf_name == "SVM":
            M_clf = CalibratedClassifierCV(clf_base, cv=3, method='isotonic', n_jobs=-1)
            M_clf.fit(X_train, y_train)
            preds = M_clf.predict(X_test)
            scores = M_clf.predict_proba(X_test)
            svm_preds, svm_scores = preds, scores
            
        elif clf_name == "MLP":
            M_clf = clf_base
            M_clf.fit(X_train, y_train)
            preds = M_clf.predict(X_test)
            scores = M_clf.predict_proba(X_test)
            mlp_preds, mlp_scores = preds, scores
            
        # AUC calculation (One-vs-Rest Macro Average)
        try:
             auc_score = roc_auc_score(y_test_numeric, scores, multi_class='ovr', average='macro', labels=np.arange(len(ALL_CLASSES)))
        except ValueError:
             auc_score = 0.0
        
        # Macro Recall
        recall_score_val = recall_score(y_test_true, preds, average='macro', zero_division=0)
        
        # Store AUC for ensemble calculation later
        if clf_name == "SVM":
            auc_svm = auc_score
        else:
            auc_mlp = auc_score
            
        # Store individual model results
        all_hog_results.append({'Metric': 'Macro Recall', 'Score': recall_score_val, 'Classifier': clf_name})
        all_hog_results.append({'Metric': 'Macro AUC', 'Score': auc_score, 'Classifier': clf_name})

    # --- Ensemble Prediction (Soft Voting) ---
    if svm_scores is not None and mlp_scores is not None:
        ensemble_preds_soft, ensemble_scores_soft = soft_vote_predict(svm_scores, mlp_scores, ALL_CLASSES)
        
        # Macro Recall
        recall_ensemble_soft = recall_score(y_test_true, ensemble_preds_soft, average='macro', zero_division=0)
        
        # AUC calculated from averaged probabilities (Soft Voting)
        auc_ensemble_soft = roc_auc_score(y_test_numeric, ensemble_scores_soft, multi_class='ovr', average='macro', labels=np.arange(len(ALL_CLASSES)))
        
        all_hog_results.append({'Metric': 'Macro Recall', 'Score': recall_ensemble_soft, 'Classifier': 'Ensemble (Soft Vote)'})
        all_hog_results.append({'Metric': 'Macro AUC', 'Score': auc_ensemble_soft, 'Classifier': 'Ensemble (Soft Vote)'})
        
        # --- Ensemble Prediction (Majority/Hard Voting) ---
        ensemble_preds_hard = majority_vote(svm_preds, mlp_preds)
        
        # Macro Recall
        recall_ensemble_hard = recall_score(y_test_true, ensemble_preds_hard, average='macro', zero_division=0)
        
        # AUC approximation for Hard Voting (Average of components)
        auc_ensemble_hard = (auc_svm + auc_mlp) / 2
        
        all_hog_results.append({'Metric': 'Macro Recall', 'Score': recall_ensemble_hard, 'Classifier': 'Ensemble (Majority Vote)'})
        all_hog_results.append({'Metric': 'Macro AUC', 'Score': auc_ensemble_hard, 'Classifier': 'Ensemble (Majority Vote)'})

            
    print(f"Trial {i+1}/{NTrial} complete.")

# Convert results list to DataFrame
df_results_hog = pd.DataFrame(all_hog_results)


# --- SUMMARY AND PLOTTING ---
print("\n" + "="*50)
print(f"HOG FEATURE ANALYSIS (SVM, MLP, Soft/Hard Ensemble - {NTrial} Trials)")
print("="*50)

# Calculate Mean and CI for each metric/classifier combination
summary = df_results_hog.groupby(['Classifier', 'Metric']).agg(
    Mean_Score=('Score', 'mean'),
    Std_Err=('Score', stats.sem),
    N=('Score', 'count')
).reset_index()

# Calculate 95% Confidence Interval
summary['CI_Lower'] = summary.apply(lambda row: stats.t.interval(0.95, row['N'] - 1, loc=row['Mean_Score'], scale=row['Std_Err'])[0], axis=1)
summary['CI_Upper'] = summary.apply(lambda row: stats.t.interval(0.95, row['N'] - 1, loc=row['Mean_Score'], scale=row['Std_Err'])[1], axis=1)

print("\n--- Summary of Performance (Mean ± 95% CI) ---")
summary_output = summary[['Classifier', 'Metric', 'Mean_Score', 'CI_Lower', 'CI_Upper']]
summary_output['Result'] = summary_output.apply(
    lambda row: f"{row['Mean_Score']:.4f} [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]", axis=1
)
print(summary_output[['Classifier', 'Metric', 'Result']].to_markdown(index=False))


# --- PLOT 1: BOXPLOTS of AUC and RECALL ---
# This visualizes mean and distribution spread across the 30 trials

plt.figure(figsize=(14, 6))
# Define classifier order for consistent plotting
CLASSIFIERS_ORDER = ['SVM', 'MLP', 'Ensemble (Majority Vote)', 'Ensemble (Soft Vote)']

# Subplot 1: Macro Recall Boxplot
plt.subplot(1, 2, 1)
sns.boxplot(
    x='Classifier', 
    y='Score', 
    data=df_results_hog[df_results_hog['Metric'] == 'Macro Recall'], 
    order=CLASSIFIERS_ORDER,
    palette='Pastel1'
)
# REMOVED TITLE: plt.title(f'Macro Recall Distribution ({NTrial} Trials)')
plt.ylabel('Macro Recall Score')
plt.xlabel('Classifier')
plt.ylim(df_results_hog['Score'].min() * 0.9, 1.0)
plt.xticks(rotation=10, ha='right', fontsize=9)
plt.tight_layout()

# Subplot 2: Macro AUC Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(
    x='Classifier', 
    y='Score', 
    data=df_results_hog[df_results_hog['Metric'] == 'Macro AUC'], 
    order=CLASSIFIERS_ORDER,
    palette='Pastel1'
)
# REMOVED TITLE: plt.title(f'Macro AUC Distribution ({NTrial} Trials)')
plt.ylabel('Macro AUC Score (OvR)')
plt.xlabel('Classifier')
plt.ylim(df_results_hog['Score'].min() * 0.9, 1.0)
plt.xticks(rotation=10, ha='right', fontsize=9)
plt.tight_layout()

# REMOVED SUPER TITLE: plt.suptitle("HoG Feature Performance Comparison: Individual vs. Soft Ensemble", fontsize=16, y=1.02)
plt.savefig('hog_boxplot_comparison_soft_hard.png')
plt.close()
print("Boxplot saved to 'hog_boxplot_comparison_soft_hard.png'")

# --- PLOT 2: HISTOGRAMS with Fitted PDF and CI ---
# This visualizes the distribution shape, CI, and Normal approximation

metrics_to_plot = ['Macro Recall', 'Macro AUC']
colors = {'SVM': 'b', 'MLP': 'r', 'Ensemble (Majority Vote)': 'orange', 'Ensemble (Soft Vote)': 'g'}
linestyles = {'SVM': '-', 'MLP': '--', 'Ensemble (Majority Vote)': ':', 'Ensemble (Soft Vote)': '-.'}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax_idx, metric in enumerate(metrics_to_plot):
    ax = axes[ax_idx]
    data_metric = df_results_hog[df_results_hog['Metric'] == metric]
    
    # Calculate overall min/max for the x-axis
    xmin = data_metric['Score'].min() * 0.95
    xmax = data_metric['Score'].max() * 1.05
    x = np.linspace(xmin, xmax, 100)

    for clf_name in CLASSIFIERS_ORDER:
        data = data_metric[data_metric['Classifier'] == clf_name]['Score']
        color = colors[clf_name]
        
        # 1. Plot Histogram (Density)
        ax.hist(data, bins=8, alpha=0.4, color=color, density=True, label=f'{clf_name} (Hist)')
        
        # 2. Overlay Normal PDF
        mu, std = data.mean(), data.std()
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, color=color, linestyle=linestyles[clf_name], linewidth=2, label=f'{clf_name} PDF')
        
        # 3. Add Mean and CI lines
        ci_low, ci_high = calculate_ci(data)
        ax.axvline(mu, color=color, linestyle='-', linewidth=1.5)
        ax.axvspan(ci_low, ci_high, color=color, alpha=0.1, label=f'95% CI')
        
    # REMOVED TITLE: ax.set_title(f'{metric} Distribution Density (HoG)')
    ax.set_xlabel(f'{metric} Score')
    ax.set_ylabel('Probability Density')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(ax.get_xticks(), rotation=10, ha='right', fontsize=9)


plt.tight_layout()
# REMOVED SUPER TITLE: plt.suptitle(f"HoG Performance Distributions with Fitted Normal PDF (Soft Ensemble - {NTrial} Trials)", fontsize=16, y=1.02)
plt.savefig('hog_histogram_distributions_soft_hard.png')
plt.close()
print("Histograms saved to 'hog_histogram_distributions_soft_hard.png'")
