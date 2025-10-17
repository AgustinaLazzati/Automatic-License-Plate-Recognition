# USAGE
# python train_simple.py --fonts input/example_fonts --char-classifier output/simple_char.cpickle \
#	--digit-classifier output/simple_digit.cpickle

##### PYTHON PACKAGES
# Generic
import pickle
import cv2
import imutils
import numpy as np
import pandas
import os
from matplotlib import pyplot as plt
import scipy

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
NTrial=1   #for exercise 1a) it was ask 1 trial. 
aucMLP=[]
aucSVC=[]
aucKNN=[]
recMLP=[]
recSVC=[]
recKNN=[]
precMLP=[]
precSVC=[]
precKNN=[]

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
    y_pred=(pSVC[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                       zero_division=0)
    recSVC.append(rec)
    precSVC.append(prec)

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
    recKNN.append(rec)
    precKNN.append(prec)

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
    recMLP.append(rec)
    precMLP.append(prec)

#### STEP2. ANALYZE RESULTS
## Visual Exploration
recSVC=np.stack(recSVC)
recKNN=np.stack(recKNN)
recMLP=np.stack(recMLP)

precSVC = np.stack(precSVC)
precKNN = np.stack(precKNN)
precMLP = np.stack(precMLP)

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
plt.savefig(os.path.join(ResultsDir, "auc_plot.png"))
plt.show()

# Print summary
print("Average AUCs:")
print("SVM:", np.mean(aucSVC))
print("KNN:", np.mean(aucKNN))
print("MLP:", np.mean(aucMLP))


