# USAGE
# python CharacterClassifier.py

##### PYTHON PACKAGES
# Generic
import pickle
import numpy as np
import os

# Classifiers
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)
from descriptors.blockbinarypixelsum import FeatureBlockBinaryPixelSum
from descriptors.lbp import FeatureLBP
from descriptors.hog import FeatureHOG

#### EXP-SET UP
# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
DataDir=r'/home/tomiock/uni2025/license/example_fonts'

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

digitsFeat={}
digitsFeat['BLCK_AVG']=[]
digitsFeat['HOG'] = []
digitsFeat['LBP'] = []

# initialize descriptors
descBlckAvg = FeatureBlockBinaryPixelSum()
descHOG = FeatureHOG()
descLBP = FeatureLBP()

### EXTRACT FEATURES
# Digits
for roi in digitsIms:
     # extract features
     digitsFeat['BLCK_AVG'].append(descBlckAvg.extract_image_features(roi))
     digitsFeat['HOG'].append(descHOG.extract_image_features(roi))
     digitsFeat['LBP'].append(descLBP.extract_image_features(roi))

### CLASSIFICATION
classifiers = {
    "SVM": LinearSVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(random_state=42)
}

for feature_name, features in digitsFeat.items():
    print(f"--- Classification for {feature_name} features ---")
    
    X = np.array(features)
    y = digitsLabels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    for clf_name, clf in classifiers.items():
        print(f"Training {clf_name}...")
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)
        
        print(f"Results for {clf_name}:")
        print(classification_report(y_test, predictions))
        print("-" * 30)
