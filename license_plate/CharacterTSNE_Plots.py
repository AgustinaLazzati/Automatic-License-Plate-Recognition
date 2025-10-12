# USAGE
# python train_simple.py --fonts input/example_fonts --char-classifier output/simple_char.cpickle \
#	--digit-classifier output/simple_digit.cpickle

##### PYTHON PACKAGES
# Generic
from imutils import paths
import argparse
import pickle
import cv2
import imutils
import numpy as np
import pandas
import os
from matplotlib import pyplot as plt

# Classifiers
# include differnet classifiers
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)
from descriptors.blockbinarypixelsum import FeatureBlockBinaryPixelSum
from descriptors.intensity import FeatureIntensity
from descriptors.lbp import FeatureLBP
from descriptors.hog import FeatureHOG

#### EXP-SET UP
# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
DataDir=r'/home/tomiock/uni2025/license/example_fonts'
ResultsDir=r'D:\Teaching\Grau\GrauIA\V&L\Challenges\Matricules\Results'
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
alphabetFeat={}

digitsFeat['BLCK_AVG']=[]
digitsFeat['HOG'] = []
digitsFeat['LBP'] = []


# initialize descriptors
blockSizes =((5, 5),)#((5, 5), (5, 10), (10, 5), (10, 10))
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


### VISUALIZE FEATURE SPACES
color=['r','m','g','cyan','y','k','orange','lime','b']
from sklearn.manifold import TSNE,trustworthiness


for targetFeat in digitsFeat.keys():
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(np.stack(digitsFeat[targetFeat]))
    
    plt.figure()
    plt.scatter(embeddings_2d[digitsLabels=='0', 0], embeddings_2d[digitsLabels=='0', 1], 
                marker='s')
    k=0
    for num in np.unique(digitsLabels)[1::]:
        plt.scatter(embeddings_2d[digitsLabels==num, 0], embeddings_2d[digitsLabels==num, 1], 
                     marker='o',color=color[k])
        k=k+1
    plt.legend(np.unique(digitsLabels))
    plt.title(targetFeat)
    plt.show()
  #  plt.savefig(os.path.join(ResultsDir,targetFeat+'DigitsFeatSpace.png'))
    
### VISUALIZE FEATURES IMAGES

## LBP Images for Digits


# HOG Images for Digits


