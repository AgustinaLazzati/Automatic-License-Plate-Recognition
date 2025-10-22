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
from sklearn.metrics import confusion_matrix, accuracy_score

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

for kTrial in np.arange(NTrial):
    # Random Train-test split
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_digits, y_digits, test_size=0.2, stratify=y_digits
    )
    
    ##### SVM
    ModelSVC = SVC(C=1.0, class_weight='balanced', probability=True)
    ModelSVC = CalibratedClassifierCV(ModelSVC, n_jobs=-1)
    ModelSVC.fit(X_train_d, y_train_d)
    y_pred_svc = ModelSVC.predict(X_test_d)
    acc_all['SVC'].append(accuracy_score(y_test_d, y_pred_svc))
    conf_all['SVC'].append(confusion_matrix(y_test_d, y_pred_svc, labels=np.arange(10)))
    
    ##### KNN
    ModelKNN = KNeighborsClassifier(n_neighbors=10)
    ModelKNN = CalibratedClassifierCV(ModelKNN, n_jobs=-1)
    ModelKNN.fit(X_train_d, y_train_d)
    y_pred_knn = ModelKNN.predict(X_test_d)
    acc_all['KNN'].append(accuracy_score(y_test_d, y_pred_knn))
    conf_all['KNN'].append(confusion_matrix(y_test_d, y_pred_knn, labels=np.arange(10)))
    
    ##### MLP
    ModelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 20), max_iter=100000, random_state=1)
    ModelMLP = CalibratedClassifierCV(ModelMLP, n_jobs=-1)
    ModelMLP.fit(X_train_d, y_train_d)
    y_pred_mlp = ModelMLP.predict(X_test_d)
    acc_all['MLP'].append(accuracy_score(y_test_d, y_pred_mlp))
    conf_all['MLP'].append(confusion_matrix(y_test_d, y_pred_mlp, labels=np.arange(10)))


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