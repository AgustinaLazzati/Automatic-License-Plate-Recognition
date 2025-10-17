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
import pandas

# Classifiers
# include differnet classifiers
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support,precision_score, recall_score, f1_score, accuracy_score, classification_report

# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)


#### STEP0. EXP-SET UP

# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
ResultsDir='data/Results'
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
NTrial=30   #for exercise 1a) it was ask 1 trial. 

averages = ['micro', 'macro', 'weighted']

aucMLP, aucSVC, aucKNN = [], [], [] 
accMLP, accSVC, accKNN = [], [], []

scores = ['micro', 'macro', 'weighted', 'class0', 'class1', 'fscore']
# Initialize dictionary keys as empty lists for storing multiple trials
precSVC = {avg: [] for avg in scores}
recSVC  = {avg: [] for avg in scores}
precKNN = {avg: [] for avg in scores}
recKNN  = {avg: [] for avg in scores}
precMLP = {avg: [] for avg in scores}
recMLP  = {avg: [] for avg in scores}

"""
# Add a 'mean' key to store the average across classes for each trial
precSVC['mean'] = []
precKNN['mean'] = []
precMLP['mean'] = []
recSVC['mean'] = []
recKNN['mean'] = []
recMLP['mean'] = []
"""

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

    # Precision & Recall for different averages
    y_pred=(pSVC[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                       zero_division=0)
    
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

    y_pred=(pKNN[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                       zero_division=0)
    
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

    y_pred=(pMLP[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                       zero_division=0)
    
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


#### STEP2. ANALYZE RESULTS
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
plt.plot(np.arange(NTrial),aucSVC,marker='o',c='b',markersize=10)
plt.plot(np.arange(NTrial),aucKNN,marker='o',c='r',markersize=10)
plt.plot(np.arange(NTrial),aucMLP,marker='o',c='g',markersize=10)
plt.legend(['SVM','KNN','MLP'])
plt.xticks(np.arange(NTrial), fontsize=10)
plt.xlabel("Trial", fontsize=15)
plt.ylabel("AUC", fontsize=15)
# Showing
#plt.savefig(os.path.join(ResultsDir, "auc_plot.png"))
plt.show()


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
plt.hist(aucKNN, bins=10, alpha=0.5, label='KNN', color='purple')
plt.hist(aucMLP, bins=10, alpha=0.5, label='MLP', color='c')
plt.xlabel("AUC", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Histogram of AUC Across Trials")
plt.legend()
plt.show()

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
plt.hist(precKNN['macro'], bins=10, alpha=0.6, label='KNN', color='purple')
plt.hist(precMLP['macro'], bins=10, alpha=0.6, label='MLP', color='c')
plt.xlabel("Precision (macro)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Histogram of Precision Across Trials")
plt.legend()
plt.show()


#EXERCISE 1D)  -----------------
# Accuracy
plt.figure()
plt.plot(np.arange(NTrial), accSVC, marker='o', c='b', markersize=10)
plt.plot(np.arange(NTrial), accKNN, marker='o', c='r', markersize=10)
plt.plot(np.arange(NTrial), accMLP, marker='o', c='g', markersize=10)
plt.legend(['SVM','KNN','MLP'])
plt.xticks(np.arange(NTrial), fontsize=10)
plt.xlabel("Trial", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.show()

# Recall (macro average)
plt.figure()
plt.plot(np.arange(NTrial), recSVC['macro'], marker='o', c='b', markersize=10)
plt.plot(np.arange(NTrial), recKNN['macro'], marker='o', c='r', markersize=10)
plt.plot(np.arange(NTrial), recMLP['macro'], marker='o', c='g', markersize=10)
plt.legend(['SVM','KNN','MLP'])
plt.xticks(np.arange(NTrial), fontsize=10)
plt.xlabel("Trial", fontsize=15)
plt.ylabel("Recall (macro)", fontsize=15)
plt.show()

# Precision (macro average)
plt.figure()
plt.plot(np.arange(NTrial), precSVC['macro'], marker='o', c='b', markersize=10)
plt.plot(np.arange(NTrial), precKNN['macro'], marker='o', c='r', markersize=10)
plt.plot(np.arange(NTrial), precMLP['macro'], marker='o', c='g', markersize=10)
plt.legend(['SVM','KNN','MLP'])
plt.xticks(np.arange(NTrial), fontsize=10)
plt.xlabel("Trial", fontsize=15)
plt.ylabel("Precision (macro)", fontsize=15)
plt.show()
#---------------------------

# Print summary
print("Average AUCs:")
print("SVM:", np.mean(aucSVC).round(4))
print("KNN:", np.mean(aucKNN).round(4))
print("MLP:", np.mean(aucMLP).round(4))


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

# Plot Precision comparison (excluding per-class)
plt.figure(figsize=(10,5))
for i, avg_method in enumerate(['micro', 'macro', 'weighted', 'fscore']):
    subset = results_df[results_df['Average'] == avg_method]
    plt.bar(np.arange(len(subset)) + i*0.2, subset['Precision'], width=0.2, label=f'Precision ({avg_method})')

plt.xticks(np.arange(len(subset)) + 0.3, subset['Model'].unique())
plt.ylabel("Precision", fontsize=12)
plt.ylim(0, 1)
plt.legend()
plt.title("Precision Comparison Across Models and Averages")
plt.grid(alpha=0.3)
plt.tight_layout()
#plt.savefig(os.path.join(ResultsDir, "precision_comparison_summary.png"))
plt.show()

# Optional: Plot per-class Precision distinction separately
plt.figure(figsize=(8,4))
for i, cls in enumerate(['class0', 'class1']):
    subset = results_df[results_df['Average'] == cls]
    plt.bar(np.arange(len(subset)) + i*0.3, subset['Precision'], width=0.3, label=f'Precision ({cls})')

plt.xticks(np.arange(len(subset)) + 0.15, subset['Model'].unique())
plt.ylabel("Precision", fontsize=12)
plt.ylim(0, 1)
plt.legend()
plt.title("Precision Comparison Across Models (Per-Class)")
plt.grid(alpha=0.3)
plt.tight_layout()
#plt.savefig(os.path.join(ResultsDir, "precision_comparison_per_class.png"))
plt.show()