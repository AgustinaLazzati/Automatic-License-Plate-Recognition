# USAGE
# python train_simple.py --fonts input/example_fonts --char-classifier output/simple_char.cpickle \
# --digit-classifier output/simple_digit.cpickle

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
DataDir = r"/home/tomiock/uni2025/license/example_fonts"
ResultsDir = r"D:\Teaching\Grau\GrauIA\V&L\Challenges\Matricules\Results"
# Load Font DataSets
fileout = os.path.join(DataDir, "alphabetIms") + ".pkl"
f = open(fileout, "rb")
data = pickle.load(f)
f.close()
alphabetIms = data["alphabetIms"]
alphabetLabels = np.array(data["alphabetLabels"])


fileout = os.path.join(DataDir, "digitsIms") + ".pkl"
f = open(fileout, "rb")
data = pickle.load(f)
f.close()
digitsIms = data["digitsIms"]
digitsLabels = np.array(data["digitsLabels"])

digitsFeat = {}
alphabetFeat = {}

digitsFeat["BLCK_AVG"] = []


# initialize descriptors
blockSizes = ((5, 5),)  # ((5, 5), (5, 10), (10, 5), (10, 10))
descBlckAvg = FeatureBlockBinaryPixelSum()

### EXTRACT FEATURES
# Digits
for roi in digitsIms:
    # extract features
    digitsFeat["BLCK_AVG"].append(descBlckAvg.extract_image_features(roi))


### VISUALIZE FEATURE SPACES
color = ["r", "m", "g", "cyan", "y", "k", "orange", "lime", "b"]
from sklearn.manifold import TSNE, trustworthiness

tsne = TSNE(n_components=2, random_state=42)

for targetFeat in digitsFeat.keys():
    embeddings_2d = tsne.fit_transform(np.stack(digitsFeat[targetFeat]))

    plt.figure()
    plt.scatter(
        embeddings_2d[digitsLabels == "0", 0],
        embeddings_2d[digitsLabels == "0", 1],
        marker="s",
    )
    k = 0
    for num in np.unique(digitsLabels)[1::]:
        plt.scatter(
            embeddings_2d[digitsLabels == num, 0],
            embeddings_2d[digitsLabels == num, 1],
            marker="o",
            color=color[k],
        )
        k = k + 1
    plt.legend(np.unique(digitsLabels))
    plt.title(targetFeat)
    plt.show()
#  plt.savefig(os.path.join(ResultsDir,targetFeat+'DigitsFeatSpace.png'))

### VISUALIZE FEATURES IMAGES

## LBP Images for Digits
descHOG = FeatureHOG()

num_examples = 10
example_images = digitsIms[:num_examples]
example_labels = digitsLabels[:num_examples]

fig, axes = plt.subplots(num_examples, 2, figsize=(16, 4 * num_examples))
fig.suptitle("HOG Feature Visualization", fontsize=10)

for i in range(num_examples):
    roi = example_images[i]
    label = example_labels[i]

    # Ensure ROI is grayscale for HOG
    if len(roi.shape) > 2:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi

    # Extract HOG image
    hog_image = descHOG.extract_pixel_features(roi_gray)

    # Plot original image
    axes[i, 0].imshow(roi, cmap=plt.cm.gray)
    axes[i, 0].set_title(f"Original Digit: {label}")
    axes[i, 0].axis("off")

    # Plot HOG image
    axes[i, 1].imshow(hog_image, cmap=plt.cm.gray)
    axes[i, 1].set_title("HOG Image")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()

## LBP Images for Digits
descLBP = FeatureLBP(
    radius=5
)

num_examples = 10
example_images = digitsIms[:num_examples]
example_labels = digitsLabels[:num_examples]

fig, axes = plt.subplots(num_examples, 2, figsize=(16, 4 * num_examples))
fig.suptitle("LBP Feature Visualization", fontsize=10)

for i in range(num_examples):
    roi = example_images[i]
    label = example_labels[i]

    if len(roi.shape) > 2:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi

    hog_image = descLBP.extract_pixel_features(roi_gray)

    # Plot original image
    axes[i, 0].imshow(roi, cmap=plt.cm.gray)
    axes[i, 0].set_title(f"Original Digit: {label}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(hog_image, cmap=plt.cm.gray)
    axes[i, 1].set_title("LBP Image")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()
