##### PYTHON PACKAGES
# Generic
import pickle
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns

# Classifiers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve


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

#### DEFINE DIGIT DATASET
DescriptorsTags = list(digitsFeat.keys())
targetFeat = DescriptorsTags[0]  # choose descriptor type

# Features
X_digits = np.stack(digitsFeat[targetFeat])

# Labels (convert string labels to integers)
y_digits = np.array([int(lab) for lab in digitsLabels])



### STEP1. TRAIN DIGIT CLASSIFIERS [0–9 DIGITS]
NTrial = 30  # number of random splits/trials

# Dictionaries to hold results
acc_all = {'SVC': [], 'KNN': [], 'MLP': []}
conf_all = {'SVC': [], 'KNN': [], 'MLP': []}

aucMLP, aucSVC, aucKNN = [], [], [] 
roc_all = {'SVC': [], 'KNN': [], 'MLP': []}  # store fpr/tpr for aggregation

for kTrial in np.arange(NTrial):
    # Random Train-test split
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_digits, y_digits, test_size=0.2, stratify=y_digits
    )
    
    ##### SVM
    ModelSVC = SVC(C=1.0, class_weight='balanced', probability=True)
    ModelSVC = CalibratedClassifierCV(ModelSVC, n_jobs=-1)
    ModelSVC.fit(X_train_d, y_train_d)
    y_prob_svc = ModelSVC.predict_proba(X_test_d)
    y_pred_svc = np.argmax(y_prob_svc, axis=1)
    acc_all['SVC'].append(accuracy_score(y_test_d, y_pred_svc))
    conf_all['SVC'].append(confusion_matrix(y_test_d, y_pred_svc, labels=np.arange(10)))
    
    auc = roc_auc_score(y_test_d, y_prob_svc, multi_class='ovr')
    aucSVC.append(auc)
    # store per-class ROC
    fpr = dict()
    tpr = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve((y_test_d==i).astype(int), y_prob_svc[:,i])
    roc_all['SVC'].append((fpr, tpr))


    ##### KNN
    ModelKNN = KNeighborsClassifier(n_neighbors=10)
    ModelKNN = CalibratedClassifierCV(ModelKNN, n_jobs=-1)
    ModelKNN.fit(X_train_d, y_train_d)
    y_prob_knn = ModelKNN.predict_proba(X_test_d)
    y_pred_knn = np.argmax(y_prob_knn, axis=1)
    acc_all['KNN'].append(accuracy_score(y_test_d, y_pred_knn))
    conf_all['KNN'].append(confusion_matrix(y_test_d, y_pred_knn, labels=np.arange(10)))
    
    auc = roc_auc_score(y_test_d, y_prob_knn, multi_class='ovr')
    aucKNN.append(auc)

    fpr = dict(); tpr = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve((y_test_d==i).astype(int), y_prob_knn[:,i])
    roc_all['KNN'].append((fpr, tpr))

    ##### MLP
    ModelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 20), max_iter=100000, random_state=1)
    ModelMLP = CalibratedClassifierCV(ModelMLP, n_jobs=-1)
    ModelMLP.fit(X_train_d, y_train_d)
    y_prob_mlp = ModelMLP.predict_proba(X_test_d)
    y_pred_mlp = np.argmax(y_prob_mlp, axis=1)
    acc_all['MLP'].append(accuracy_score(y_test_d, y_pred_mlp))
    conf_all['MLP'].append(confusion_matrix(y_test_d, y_pred_mlp, labels=np.arange(10)))
    
    auc = roc_auc_score(y_test_d, y_prob_mlp, multi_class='ovr')
    aucMLP.append(auc)

    fpr = dict(); tpr = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve((y_test_d==i).astype(int), y_prob_mlp[:,i])
    roc_all['MLP'].append((fpr, tpr))

        


#### STEP2. AGGREGATE RESULTS
mean_cm_all = {}
std_cm_all = {}
mean_acc_all = {}
std_acc_all = {}

for model in ['SVC', 'KNN', 'MLP']:
    mean_cm_all[model] = np.mean(conf_all[model], axis=0)
    std_cm_all[model] = np.std(conf_all[model], axis=0)
    mean_acc_all[model] = np.mean(acc_all[model])
    std_acc_all[model] = np.std(acc_all[model])
    # Normalize mean confusion matrix
    mean_cm_all[model] = mean_cm_all[model] / mean_cm_all[model].sum(axis=1, keepdims=True)

    print(f"\n{model} Average Accuracy over {NTrial} trials: {mean_acc_all[model]:.4f} ± {std_acc_all[model]:.4f}")


#### STEP3. VISUALIZATION
model_names = ['SVC', 'KNN', 'MLP']
colors = {'SVC':'Blues', 'KNN':'Reds', 'MLP':'Greens'}

for model in model_names:
    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_cm_all[model], annot=True, fmt=".2f", cmap=colors[model],
                xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(f"{model} Confusion Matrix (Averaged over {NTrial} Trials)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()

#AUC
#### Plots accoss trials (random splits)
plt.figure()
plt.plot(np.arange(NTrial),aucSVC,marker='o',c='b',markersize=10)
plt.plot(np.arange(NTrial),aucKNN,marker='o',c='r',markersize=10)
plt.plot(np.arange(NTrial),aucMLP,marker='o',c='g',markersize=10)
plt.legend(['SVM','KNN','MLP'])
plt.xticks(np.arange(NTrial), fontsize=10)
plt.xlabel("Trial", fontsize=15)
plt.ylabel("AUC", fontsize=15)
plt.show()


#### AGGREGATED ROC CURVE
# Average TPR over trials for each class and each model
mean_fpr = np.linspace(0,1,100)

def compute_mean_roc(roc_list):
    tprs = []
    for fpr_dict, tpr_dict in roc_list:
        # average across classes
        tpr_avg = np.zeros_like(mean_fpr)
        for i in range(10):
            tpr_avg += np.interp(mean_fpr, fpr_dict[i], tpr_dict[i])
        tpr_avg /= 10  # average across classes
        tprs.append(tpr_avg)
    return np.mean(tprs, axis=0)

mean_tpr_svc = compute_mean_roc(roc_all['SVC'])
mean_tpr_knn = compute_mean_roc(roc_all['KNN'])
mean_tpr_mlp = compute_mean_roc(roc_all['MLP'])

plt.figure(figsize=(8,6))
plt.plot(mean_fpr, mean_tpr_svc, c='b', lw=2, label=f'SVC')
plt.plot(mean_fpr, mean_tpr_knn, c='r', lw=2, label=f'KNN')
plt.plot(mean_fpr, mean_tpr_mlp, c='g', lw=2, label=f'MLP')
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Aggregated ROC Curve Across Trials (Digits)")
plt.legend()
plt.show()