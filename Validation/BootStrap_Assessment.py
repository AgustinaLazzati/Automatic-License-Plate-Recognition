# USAGE
# python train_simple.py --fonts input/example_fonts --char-classifier output/simple_char.cpickle \
#	--digit-classifier output/simple_digit.cpickle

##### PYTHON PACKAGES
# Generic
import pickle
import cv2
import imutils
import numpy as np
import os
from matplotlib import pyplot as plt
import scipy
from scipy import stats
import pandas

# Classifiers
# include differnet classifiers
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,  roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support,precision_score, recall_score, f1_score, accuracy_score, classification_report
import seaborn as sns

# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)

# Function to calculate 95% CI
def calculate_ci(data):
    """Calculates the 95% Confidence Interval for the mean of the data."""
    # Ensure there's enough data to calculate CI
    if len(data) < 2:
        return (np.mean(data) if data else 0, np.mean(data) if data else 0)
    # degrees of freedom: df = N - 1
    # loc = sample mean
    # scale = standard error of the mean (SEM)
    # Using the user-provided snippet structure
    return stats.t.interval(
        confidence=0.95, 
        df=len(data) - 1, 
        loc=np.mean(data), 
        scale=stats.sem(data)
    )


#### STEP0. EXP-SET UP

# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
ResultsDir='data/descriptors'
# Load Font DataSets
fileout=os.path.join(ResultsDir,'AlphabetDescriptors')+'.pkl'     
f=open(fileout, 'rb')
data=pickle.load(f)
f.close()  
alphabetFeat=data['alphabetFeat']
alphabetLabels=data['alphabetLabels']
    

fileout=os.path.join(ResultsDir,'DigitsDescriptors')+'.pkl'    
f=open(fileout, 'rb')
data=pickle.load(f)
f.close()    
digitsFeat=data['digitsFeat']
digitsLabels=data['digitsLabels']


#### DEFINE BINARY DATASET
DescriptorsTags=list(digitsFeat.keys())
targetFeat=DescriptorsTags[0]

digits=np.stack(digitsFeat[targetFeat])
digitsLab=np.zeros(digits.shape[0])
chars=np.stack(alphabetFeat[targetFeat])
charsLab=np.ones(chars.shape[0])

X=np.concatenate((digits,chars))
y=np.concatenate((digitsLab,charsLab))

### STEP1. TRAIN BINARY CLASSIFIERS [CHARACTER VS DIGITS]
NTrial=30

averages = ['micro', 'macro', 'weighted']

aucMLP, aucSVC, aucKNN = [], [], []  
accMLP, accSVC, accKNN = [], [], []
rocSVC, rocKNN, rocMLP = [], [], []

scores = ['micro', 'macro', 'weighted', 'class0', 'class1', 'fscore']
# Initialize dictionary keys as empty lists for storing multiple trials
precSVC = {avg: [] for avg in scores}
recSVC  = {avg: [] for avg in scores}
precKNN = {avg: [] for avg in scores}
recKNN  = {avg: [] for avg in scores}
precMLP = {avg: [] for avg in scores}
recMLP  = {avg: [] for avg in scores}

# Store all predictions and true labels for confusion matrices
y_true_all = {'SVC': [], 'KNN': [], 'MLP': []}
y_pred_all = {'SVC': [], 'KNN': [], 'MLP': []}


#-----------------------------------------------------------------------------
# We will train each model for all the different trials 
#----------------------------------------------------------------------------
for kTrial in np.arange(NTrial):
    # Random Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y)
    
    ##### SVM
    ## Train Model
    ModelSVC = SVC(C=1.0,class_weight='balanced') #compute loss with weights accounting for class unbalancing
    # Use CalibratedClassifierCV to calibrate probabilites
    ModelSVC = CalibratedClassifierCV(ModelSVC,n_jobs=-1)
    ModelSVC.fit(X_train, y_train)
    ## Evaluate Model
    pSVC = ModelSVC.predict_proba(X_test)
    
    ## Metrics
    auc=roc_auc_score(y_test, pSVC[:,1])
    aucSVC.append(auc)
    fpr_svc, tpr_svc, _ = roc_curve(y_test, pSVC[:,1])  
    rocSVC.append((fpr_svc, tpr_svc))  

    # Precision & Recall for different averages
    y_pred=(pSVC[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                             zero_division=0)
    
    # Store predictions for confusion matrix
    y_true_all['SVC'].extend(y_test)
    y_pred_all['SVC'].extend(y_pred)  
    
    accSVC.append(accuracy_score(y_test, y_pred))
    precSVC['fscore'].append(np.mean(prec))
    recSVC['fscore'].append(np.mean(rec))
    
    #now, per class
    precSVC['class0'].append(prec[0])
    precSVC['class1'].append(prec[1])
    recSVC['class0'].append(rec[0])
    recSVC['class1'].append(rec[1])


    # Loop through averaging methods
    for avg_method in averages:
        prec_val = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        rec_val  = recall_score(y_test, y_pred, average=avg_method, zero_division=0)

        precSVC[avg_method].append(prec_val)
        recSVC[avg_method].append(rec_val)

    ##### KNN
    ## Train Model
    ModelKNN = KNeighborsClassifier(n_neighbors=10)
    ModelKNN = CalibratedClassifierCV(ModelKNN,n_jobs=-1)
    ModelKNN.fit(X_train, y_train)
    ## Evaluate Model
    pKNN = ModelKNN.predict_proba(X_test)
    # Metrics
    auc=roc_auc_score(y_test, pKNN[:,1])
    aucKNN.append(auc)
    fpr_knn, tpr_knn, _ = roc_curve(y_test, pKNN[:,1])   
    rocKNN.append((fpr_knn, tpr_knn))

    y_pred=(pKNN[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                             zero_division=0)
    
    # Store predictions for confusion matrix
    y_true_all['KNN'].extend(y_test)
    y_pred_all['KNN'].extend(y_pred)

    accKNN.append(accuracy_score(y_test, y_pred))
    precKNN['fscore'].append(np.mean(prec))
    recKNN['fscore'].append(np.mean(rec))

    #now, per class
    recKNN['class0'].append(rec[0])
    recKNN['class1'].append(rec[1])
    precKNN['class0'].append(prec[0])
    precKNN['class1'].append(prec[1])

    # Loop through averaging methods
    for avg_method in averages:
        prec_val = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        rec_val  = recall_score(y_test, y_pred, average=avg_method, zero_division=0)

        precKNN[avg_method].append(prec_val)
        recKNN[avg_method].append(rec_val)

    #### MLP
    ## Train Model
    ModelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15, 20), random_state=1,max_iter=100000)
    ModelMLP = CalibratedClassifierCV(ModelMLP,n_jobs=-1)
    ModelMLP.fit(X_train, y_train)
    ## Evaluate Model
    pMLP = ModelMLP.predict_proba(X_test)
    # Metrics
    auc=roc_auc_score(y_test, pMLP[:,1])
    aucMLP.append(auc)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, pMLP[:,1])  
    rocMLP.append((fpr_mlp, tpr_mlp))

    y_pred=(pMLP[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                             zero_division=0)
    
    # Store predictions for confusion matrix
    y_true_all['MLP'].extend(y_test)
    y_pred_all['MLP'].extend(y_pred)

    accMLP.append(accuracy_score(y_test, y_pred))
    precMLP['fscore'].append(np.mean(prec))
    recMLP['fscore'].append(np.mean(rec))

    #now, per class
    recMLP['class0'].append(rec[0])
    recMLP['class1'].append(rec[1])
    precMLP['class0'].append(prec[0])
    precMLP['class1'].append(prec[1])

    # Loop through averaging methods
    for avg_method in averages:
        prec_val = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        rec_val  = recall_score(y_test, y_pred, average=avg_method, zero_division=0)

        precMLP[avg_method].append(prec_val)
        recMLP[avg_method].append(rec_val)


# T-test for AUC: SVC vs KNN
t_stat_svc_knn_auc, p_svc_knn_auc = stats.ttest_ind(aucSVC, aucKNN)
print(f"\nAUC t-test (SVM vs KNN): t={t_stat_svc_knn_auc:.5f}, p={p_svc_knn_auc:.5f}")

# T-test for AUC: SVC vs MLP
t_stat_svc_mlp_auc, p_svc_mlp_auc = stats.ttest_ind(aucSVC, aucMLP)
print(f"AUC T-test (SVM vs MLP): t={t_stat_svc_mlp_auc:.5f}, p={p_svc_mlp_auc:.5f}")

# T-test for AUC: KNN vs MLP
t_stat_knn_mlp_auc, p_knn_mlp_auc = stats.ttest_ind(aucKNN, aucMLP)
print(f"AUC T-test (KNN vs MLP): t={t_stat_knn_mlp_auc:.5f}, p={p_knn_mlp_auc:.5f}")

# T-test for Accuracy: SVC vs KNN
t_stat_svc_knn_acc, p_svc_knn_acc = stats.ttest_ind(accSVC, accKNN)
print(f"\nAccuracy T-test (SVM vs KNN): t={t_stat_svc_knn_acc:.5f}, p={p_svc_knn_acc:.5f}")

# T-test for Accuracy: SVC vs MLP
t_stat_svc_mlp_acc, p_svc_mlp_acc = stats.ttest_ind(accSVC, accMLP)
print(f"Accuracy T-test (SVM vs MLP): t={t_stat_svc_mlp_acc:.5f}, p={p_svc_mlp_acc:.5f}")

# T-test for Accuracy: KNN vs MLP
t_stat_knn_mlp_acc, p_knn_mlp_acc = stats.ttest_ind(accKNN, accMLP)
print(f"Accuracy T-test (KNN vs MLP): t={t_stat_knn_mlp_acc:.5f}, p={p_knn_mlp_acc:.5f}")


#### STEP2. ANALYZE RESULTS
# ------------------------------------------------------
# In this step we will compute the ROC, AUC and CONFUSION MATRIX
# ------------------------------------------------------
"""
## Visual Exploration
recSVC=np.stack(recSVC)
recKNN=np.stack(recKNN)
recMLP=np.stack(recMLP)

precSVC = np.stack(precSVC)
precKNN = np.stack(precKNN)
precMLP = np.stack(precMLP)
"""

#### Plots accoss trials (random splits)
plt.figure()
plt.plot(np.arange(NTrial),aucSVC,marker='o',c='b',markersize=2, alpha=0.5)
plt.plot(np.arange(NTrial),aucKNN,marker='o',c='r',markersize=2, alpha=0.5)
plt.plot(np.arange(NTrial),aucMLP,marker='o',c='g',markersize=2, alpha=0.5)
plt.legend(['SVM','KNN','MLP'])
plt.xticks(np.arange(0, NTrial + 1, NTrial/10), fontsize=10) # Adjust x-ticks for 100 trials
plt.xlabel("Trial", fontsize=15)
plt.ylabel("AUC", fontsize=15)
# Showing
#plt.savefig(os.path.join(ResultsDir, "auc_plot.png"))
plt.show()


# Calculate CIs
ci_aucSVC = calculate_ci(aucSVC)
ci_aucKNN = calculate_ci(aucKNN)
ci_aucMLP = calculate_ci(aucMLP)

ci_accSVC = calculate_ci(accSVC)
ci_accKNN = calculate_ci(accKNN)
ci_accMLP = calculate_ci(accMLP)

# --- NEW: Calculate CI for Macro Precision ---
ci_precSVC = calculate_ci(precSVC['macro'])
ci_precKNN = calculate_ci(precKNN['macro'])
ci_precMLP = calculate_ci(precMLP['macro'])

# --- CI Visualization Data Setup ---
models = ['SVM', 'KNN', 'MLP']

# AUC Data
mean_auc = [np.mean(aucSVC), np.mean(aucKNN), np.mean(aucMLP)]
ci_auc = [ci_aucSVC, ci_aucKNN, ci_aucMLP]
# Margin of Error (Distance from Mean to Lower Bound)
err_auc = [mean_auc[i] - ci_auc[i][0] for i in range(len(models))]

# Accuracy Data
mean_acc = [np.mean(accSVC), np.mean(accKNN), np.mean(accMLP)]
ci_acc = [ci_accSVC, ci_accKNN, ci_accMLP]
# Margin of Error (Distance from Mean to Lower Bound)
err_acc = [mean_acc[i] - ci_acc[i][0] for i in range(len(models))]

# --- CI PLOTS ---

#EXERCISE 2A) ------------------
# Boxplot for AUC
plt.figure(figsize=(8,5))
plt.boxplot([aucSVC, aucKNN, aucMLP], tick_labels=['SVM','KNN','MLP'])
plt.ylabel("AUC", fontsize=12)
plt.title("Boxplot of AUC Across Trials")
plt.show()

# Histogram for AUC
plt.figure(figsize=(8,5))
plt.hist(aucSVC, bins=10, alpha=0.5, label='SVM', color='b')
plt.hist(aucKNN, bins=10, alpha=0.5, label='KNN', color='r')
plt.hist(aucMLP, bins=10, alpha=0.5, label='MLP', color='g')
# Changed to density=True for Normal PDF overlay
plt.hist(aucSVC, bins=10, alpha=0.5, label='SVM', color='b', density=True)
plt.hist(aucKNN, bins=10, alpha=0.5, label='KNN', color='purple', density=True)
plt.hist(aucMLP, bins=10, alpha=0.5, label='MLP', color='c', density=True)

ax = plt.gca()
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)

# Overlay Normal PDF for SVM (Blue)
mu, std = np.mean(aucSVC), np.std(aucSVC)
p = stats.norm.pdf(x, mu, std)
ax.plot(x, p, 'b--', linewidth=2, label=f'SVM Normal Fit')

# Overlay Normal PDF for KNN (Red/Purple)
mu, std = np.mean(aucKNN), np.std(aucKNN)
p = stats.norm.pdf(x, mu, std)
ax.plot(x, p, 'r--', linewidth=2, label=f'KNN Normal Fit') # Using red for consistency

# Overlay Normal PDF for MLP (Green/Cyan)
mu, std = np.mean(aucMLP), np.std(aucMLP)
p = stats.norm.pdf(x, mu, std)
ax.plot(x, p, 'g--', linewidth=2, label=f'MLP Normal Fit')

# Add Mean and CI lines for AUC
# SVM (Blue)
ax.axvline(np.mean(aucSVC), color='b', linestyle='-', linewidth=1.5, label='SVM Mean')
ax.axvline(ci_aucSVC[0], color='b', linestyle=':', linewidth=1)
ax.axvline(ci_aucSVC[1], color='b', linestyle=':', linewidth=1)

# KNN (Red)
ax.axvline(np.mean(aucKNN), color='r', linestyle='-', linewidth=1.5, label='KNN Mean')
ax.axvline(ci_aucKNN[0], color='r', linestyle=':', linewidth=1)
ax.axvline(ci_aucKNN[1], color='r', linestyle=':', linewidth=1)

# MLP (Green)
ax.axvline(np.mean(aucMLP), color='g', linestyle='-', linewidth=1.5, label='MLP Mean')
ax.axvline(ci_aucMLP[0], color='g', linestyle=':', linewidth=1)
ax.axvline(ci_aucMLP[1], color='g', linestyle=':', linewidth=1)

plt.legend(loc='upper left', fontsize=8) # Lower fontsize to fit more labels
plt.xlabel("AUC", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.title("Probability Density of AUC Across Trials with Fitted Normal Distribution and 95% CI")
plt.show()

# Print summary
print("Average AUCs:")
print("SVM:", np.mean(aucSVC).round(4))
print("KNN:", np.mean(aucKNN).round(4))
print("MLP:", np.mean(aucMLP).round(4))

# ROC CURVE --------------------------------------------
# ROC CURVE (averaged across all trials)
mean_fpr = np.linspace(0, 1, 100)

def compute_mean_roc(roc_list, auc_list):
    """Compute mean ROC by interpolating TPRs at common FPR points."""
    tprs = []
    for fpr, tpr in roc_list:
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc_list)
    return mean_tpr, mean_auc

# Compute averaged ROC for each model
mean_tpr_svc, mean_auc_svc = compute_mean_roc(rocSVC, aucSVC)
mean_tpr_knn, mean_auc_knn = compute_mean_roc(rocKNN, aucKNN)
mean_tpr_mlp, mean_auc_mlp = compute_mean_roc(rocMLP, aucMLP)

# Plot averaged ROC
plt.figure(figsize=(8,6))
plt.plot(mean_fpr, mean_tpr_svc, color='b', lw=2, label=f'SVM (AUC = {mean_auc_svc:.3f})')
plt.plot(mean_fpr, mean_tpr_knn, color='r', lw=2, label=f'KNN (AUC = {mean_auc_knn:.3f})')
plt.plot(mean_fpr, mean_tpr_mlp, color='g', lw=2, label=f'MLP (AUC = {mean_auc_mlp:.3f})')
plt.plot([0,1], [0,1], 'k--', lw=1, label='Chance')

plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Average ROC Curve Across Trials", fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# CONFUSION MATRIX ---------------------------------------
# Merged across all trials
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, model_name in zip(axes, ['SVC', 'KNN', 'MLP']):
    cm = confusion_matrix(y_true_all[model_name], y_pred_all[model_name])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Digit (0)', 'Char (1)'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{model_name} - Combined Confusion Matrix')
    ax.grid(False)

plt.suptitle(f"Confusion Matrices Aggregated Across {NTrial} Trials", fontsize=14, y=0.96)
plt.tight_layout()
plt.show()



#### STEP3. SUMMARY AND VISUALIZATION ####
# --------------------------------------------------------
# in this step, we will compute BOX PLOTS, BAR PLOTS, and HISTOGRAMS. 
# --------------------------------------------------------
"""
# BOX PLOTS + HISTOGRAMS OF PRECISION AND RECALL ---------
# Boxplot for Precision (macro) across trials
plt.figure(figsize=(8,5))
box = plt.boxplot([precSVC['macro'], precKNN['macro'], precMLP['macro']], 
             tick_labels=['SVM','KNN','MLP'])
plt.ylabel("Precision (macro)", fontsize=12)
plt.title("Boxplot of Precision Across Trials")
plt.show()

# Histogram for Precision (macro) across trials
plt.figure(figsize=(8,5))
plt.hist(precSVC['macro'], bins=10, alpha=0.6, label='SVM', color='b')
plt.hist(precKNN['macro'], bins=10, alpha=0.6, label='KNN', color='r')
plt.hist(precMLP['macro'], bins=10, alpha=0.6, label='MLP', color='g')
# Changed to density=True for Normal PDF overlay
plt.hist(precSVC['macro'], bins=10, alpha=0.6, label='SVM', color='b', density=True)
plt.hist(precKNN['macro'], bins=10, alpha=0.6, label='KNN', color='purple', density=True)
plt.hist(precMLP['macro'], bins=10, alpha=0.6, label='MLP', color='c', density=True)

ax = plt.gca()
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)

# Overlay Normal PDF for SVM (Blue)
mu, std = np.mean(precSVC['macro']), np.std(precSVC['macro'])
p = stats.norm.pdf(x, mu, std)
ax.plot(x, p, 'b--', linewidth=2, label=f'SVM Normal Fit')

# Overlay Normal PDF for KNN (Red/Purple)
mu, std = np.mean(precKNN['macro']), np.std(precKNN['macro'])
p = stats.norm.pdf(x, mu, std)
ax.plot(x, p, 'r--', linewidth=2, label=f'KNN Normal Fit')

# Overlay Normal PDF for MLP (Green/Cyan)
mu, std = np.mean(precMLP['macro']), np.std(precMLP['macro'])
p = stats.norm.pdf(x, mu, std)
ax.plot(x, p, 'g--', linewidth=2, label=f'MLP Normal Fit')

# Add Mean and CI lines for Macro Precision
# SVM (Blue)
ax.axvline(np.mean(precSVC['macro']), color='b', linestyle='-', linewidth=1.5, label='SVM Mean')
ax.axvline(ci_precSVC[0], color='b', linestyle=':', linewidth=1)
ax.axvline(ci_precSVC[1], color='b', linestyle=':', linewidth=1)

# KNN (Red)
ax.axvline(np.mean(precKNN['macro']), color='r', linestyle='-', linewidth=1.5, label='KNN Mean')
ax.axvline(ci_precKNN[0], color='r', linestyle=':', linewidth=1)
ax.axvline(ci_precKNN[1], color='r', linestyle=':', linewidth=1)

# MLP (Green)
ax.axvline(np.mean(precMLP['macro']), color='g', linestyle='-', linewidth=1.5, label='MLP Mean')
ax.axvline(ci_precMLP[0], color='g', linestyle=':', linewidth=1)
ax.axvline(ci_precMLP[1], color='g', linestyle=':', linewidth=1)

plt.legend(loc='upper left', fontsize=8)
>>>>>>> afa116e (some pipe)
plt.xlabel("Precision (macro)", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.title("Probability Density of Precision (macro) Across Trials with Fitted Normal Distribution and 95% CI")
plt.show()

# Boxplot for Recall (macro) across trials
plt.figure(figsize=(8,5))
box = plt.boxplot([recSVC['macro'], recKNN['macro'], recMLP['macro']], 
                  tick_labels=['SVM','KNN','MLP'])
plt.ylabel("Precision (macro)", fontsize=12)
plt.title("Boxplot of Recall Across Trials")
plt.show()

# Histogram for Precision (macro) across trials
plt.figure(figsize=(8,5))
plt.hist(recSVC['macro'], bins=10, alpha=0.6, label='SVM', color='b')
plt.hist(recKNN['macro'], bins=10, alpha=0.6, label='KNN', color='r')
plt.hist(recMLP['macro'], bins=10, alpha=0.6, label='MLP', color='g')
plt.xlabel("Recall (macro)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Histogram of Recall Across Trials")
plt.legend()
plt.show()

#EXERCISE 1D)  MICRO MACRO -----------------
# Accuracy
plt.figure()
plt.plot(np.arange(NTrial), accSVC, marker='o', c='b', markersize=2, alpha=0.5)
plt.plot(np.arange(NTrial), accKNN, marker='o', c='r', markersize=2, alpha=0.5)
plt.plot(np.arange(NTrial), accMLP, marker='o', c='g', markersize=2, alpha=0.5)
plt.legend(['SVM','KNN','MLP'])
plt.xticks(np.arange(0, NTrial + 1, NTrial/10), fontsize=10)
plt.xlabel("Trial", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.show()

# Recall (macro average)
plt.figure()
plt.plot(np.arange(NTrial), recSVC['macro'], marker='o', c='b', markersize=2, alpha=0.5)
plt.plot(np.arange(NTrial), recKNN['macro'], marker='o', c='r', markersize=2, alpha=0.5)
plt.plot(np.arange(NTrial), recMLP['macro'], marker='o', c='g', markersize=2, alpha=0.5)
plt.legend(['SVM','KNN','MLP'])
plt.xticks(np.arange(0, NTrial + 1, NTrial/10), fontsize=10)
plt.xlabel("Trial", fontsize=15)
plt.ylabel("Recall (macro)", fontsize=15)
plt.show()

# Precision (macro average)
plt.figure()
plt.plot(np.arange(NTrial), precSVC['macro'], marker='o', c='b', markersize=2, alpha=0.5)
plt.plot(np.arange(NTrial), precKNN['macro'], marker='o', c='r', markersize=2, alpha=0.5)
plt.plot(np.arange(NTrial), precMLP['macro'], marker='o', c='g', markersize=2, alpha=0.5)
plt.legend(['SVM','KNN','MLP'])
plt.xticks(np.arange(0, NTrial + 1, NTrial/10), fontsize=10)
plt.xlabel("Trial", fontsize=15)
plt.ylabel("Precision (macro)", fontsize=15)
plt.show()

=======
# Print summary
print("Average AUCs:")
print("SVM:", np.mean(aucSVC).round(4))
print("KNN:", np.mean(aucKNN).round(4))
print("MLP:", np.mean(aucMLP).round(4))

# Print Summary CIs
print("\n" + "="*65)
print(" 95% CONFIDENCE INTERVALS (CI) FOR MEAN AUC, ACCURACY, AND MACRO PRECISION")
print("="*65)

print("\n--- AUC Confidence Intervals ---")
print(f"SVM AUC CI (95%): ({ci_aucSVC[0]:.4f}, {ci_aucSVC[1]:.4f}) | Mean: {np.mean(aucSVC):.4f}")
print(f"KNN AUC CI (95%): ({ci_aucKNN[0]:.4f}, {ci_aucKNN[1]:.4f}) | Mean: {np.mean(aucKNN):.4f}")
print(f"MLP AUC CI (95%): ({ci_aucMLP[0]:.4f}, {ci_aucMLP[1]:.4f}) | Mean: {np.mean(aucMLP):.4f}")

print("\n--- Accuracy Confidence Intervals ---")
print(f"SVM Accuracy CI (95%): ({ci_accSVC[0]:.4f}, {ci_accSVC[1]:.4f}) | Mean: {np.mean(accSVC):.4f}")
print(f"KNN Accuracy CI (95%): ({ci_accKNN[0]:.4f}, {ci_accKNN[1]:.4f}) | Mean: {np.mean(accKNN):.4f}")
print(f"MLP Accuracy CI (95%): ({ci_accMLP[0]:.4f}, {ci_accMLP[1]:.4f}) | Mean: {np.mean(accMLP):.4f}")

# --- NEW: Print CI for Macro Precision ---
print("\n--- Precision (macro) Confidence Intervals ---")
print(f"SVM Precision CI (95%): ({ci_precSVC[0]:.4f}, {ci_precSVC[1]:.4f}) | Mean: {np.mean(precSVC['macro']):.4f}")
print(f"KNN Precision CI (95%): ({ci_precKNN[0]:.4f}, {ci_precKNN[1]:.4f}) | Mean: {np.mean(precKNN['macro']):.4f}")
print(f"MLP Precision CI (95%): ({ci_precMLP[0]:.4f}, {ci_precMLP[1]:.4f}) | Mean: {np.mean(precMLP['macro']):.4f}")

# Comparing Precision, Recall, and Accuracy  for exercise 1A)
#### STEP3. SUMMARY AND VISUALIZATION ####
#all_averages = averages + ['fscore']
results = []

# Define models and collect metrics
models = {
    'SVC': (precSVC, recSVC, accSVC),
    'KNN': (precKNN, recKNN, accKNN),
    'MLP': (precMLP, recMLP, accMLP)
}

for model_name, (prec_dict, rec_dict, acc_list) in models.items():
    for avg_method in scores:  # now includes 'mean'
        results.append({
            'Model': model_name,
            'Average': avg_method,
            'Precision': np.mean(prec_dict[avg_method]),
            'Recall': np.mean(rec_dict[avg_method]),
            'Accuracy': np.mean(acc_list)
        })


results_df = pandas.DataFrame(results)

# Nicely formatted printout
print("\n" + "="*65)
print(" COMPARISON OF PRECISION, RECALL, AND ACCURACY (Averaged + Mean)")
print("="*65)
for avg_method in scores:
    subset = results_df[results_df["Average"] == avg_method]
    print(f"\n>> {avg_method.upper()} average:")
    print(subset[['Model', 'Precision', 'Recall', 'Accuracy']].round(3).to_string(index=False))

# Plot Recall comparison (excluding per-class)
plt.figure(figsize=(10,5))
for i, avg_method in enumerate(['micro', 'macro', 'weighted', 'fscore']):
    subset = results_df[results_df['Average'] == avg_method]
    plt.bar(np.arange(len(subset)) + i*0.2, subset['Recall'], width=0.2, label=f'Recall ({avg_method})')

plt.xticks(np.arange(len(subset)) + 0.3, subset['Model'].unique())
plt.ylabel("Recall", fontsize=12)
plt.ylim(0, 1)
plt.legend()
plt.title("Recall Comparison Across Models and Averages")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# PER-CLASS ANALYSIS ---------------------------------------
# per-class RECALL distinction separately BARPLOT
plt.figure(figsize=(8,4))
for i, cls in enumerate(['class0', 'class1']):
    subset = results_df[results_df['Average'] == cls]
    plt.bar(np.arange(len(subset)) + i*0.3, subset['Recall'], width=0.3, label=f'Recall ({cls})')

plt.xticks(np.arange(len(subset)) + 0.15, subset['Model'].unique())
plt.ylabel("Recall", fontsize=12)
plt.ylim(0, 1)
plt.legend()
plt.title("Recall Comparison Across Models (Per-Class)")
plt.grid(alpha=0.3)
plt.tight_layout()
<<<<<<< HEAD
plt.show()

# Per-Class RECALL Comparison BOXPLOT
plt.figure(figsize=(10, 6))

recall_data = [
    recSVC['class0'], recSVC['class1'],
    recKNN['class0'], recKNN['class1'],
    recMLP['class0'], recMLP['class1']
]

# Define labels (grouped visually by model)
labels = [
    'SVM\nClass 0', 'SVM\nClass 1',
    'KNN\nClass 0', 'KNN\nClass 1',
    'MLP\nClass 0', 'MLP\nClass 1'
]

box = plt.boxplot(recall_data, tick_labels=labels, patch_artist=False, widths=0.6)
plt.ylabel("Recall", fontsize=12)
plt.title("Per-Class Recall Distribution Across Models", fontsize=14)
plt.tight_layout()
plt.show()
"""
#plt.savefig(os.path.join(ResultsDir, "precision_comparison_per_class.png"))
plt.show()
