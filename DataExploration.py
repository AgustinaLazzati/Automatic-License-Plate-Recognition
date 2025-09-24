"""
This is the pipeline for the introduction to data properties exploration. 

Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres"
__license__ = "GPL"
__email__ = "debora,gtorres@cvc.uab.es"

"""

##### PYTHON PACKAGES
# import the necessary packages
import numpy as np
import cv2
import glob
import os
from imutils import perspective
from matplotlib import pyplot as plt

# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)
from LicensePlateDetector import detectPlates


"""
#### EXP-SET UP
# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
DataDir=r'data/Patentes'
Views=['FrontalAugmented','LateralAugmented']

"""
#### COMPUTE PROPERTIES FOR EACH VIEW
def computeProperties (DataDir,Views):
    plateArea={}
    plateAngle={}
    imageColor={}
    imageIlluminance={}
    imageSaturation={}

    for View in Views:

        ImageFiles=sorted(glob.glob(os.path.join(DataDir,View,'*.jpg')))
        plateArea[View]=[]
        plateAngle[View]=[]
        imageColor[View]=[]
        imageIlluminance[View]=[]
        imageSaturation[View]=[]
        # loop over the images
        for imagePath in ImageFiles:
            # load the image
            image = cv2.imread(imagePath)
            # Image Color and Illuminance properties
            imageColor[View].append(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,0].flatten()))
            imageIlluminance[View].append(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,2].flatten()))
            imageSaturation[View].append(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1].flatten()))
            # Image ViewPoint (orientation with respect frontal view and focal distance)
            regions, _ =detectPlates(image)  #OUR detectPlates RETURNS (regions, image)
            for reg in regions:
                # Region Properties
                reg = np.array(reg, dtype=np.float32)   # ensuring type is correct
                rect = cv2.minAreaRect(reg)
                
                plateArea[View].append(np.prod(rect[1]))
                # Due to the way cv2.minAreaRect computes the sides of the rectangle
                # (https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/)
                # Depending on view point, the estimated rectangle has not
                # the largest side along the horizontal axis. This cases are corrected to ensure that orientations
                # are always with respect the largest side 
                if (rect[1][0]<rect[1][1]):
                    plateAngle[View].append(rect[2]-90)
                else:
                    plateAngle[View].append(rect[2])

    return plateArea, plateAngle, imageColor, imageIlluminance, imageSaturation            


#### VISUALLY EXPLORE PROPERTIES DISTRIBUTION FOR EACH VIEW
# -------------------------------------------------------------
def Visualplots(name, Views, plateAngle, imageColor, 
          imageIlluminance, imageSaturation, SHOW_SEPARATE=False):

    co=['b','c']  # colors for views
    
    # -------------------------------------------------------------
    ## Color Distribution
    if SHOW_SEPARATE:
        for k, view in enumerate(Views):
            plt.figure()
            plt.hist(imageColor[view], bins=20, edgecolor='k', color=co[k], alpha=0.7)
            plt.title(f'{name} Color Distribution - {view}')
    else:
        plt.figure()
        for k, view in enumerate(Views):
            plt.hist(imageColor[view], bins=20, edgecolor='k', color=co[k], alpha=0.7)
        plt.title(f'{name} Color Distribution (Combined)')
        plt.legend(Views)

    # Boxplot
    plt.figure()
    x=[imageColor[v] for v in Views]
    bpC = plt.boxplot(x, patch_artist=True, tick_labels=Views)
    for patch, color in zip(bpC['boxes'], co):
        patch.set_facecolor(color)
    for median in bpC['medians']:
        median.set(color='black', linewidth=2)
    plt.title(f'{name} Color Distribution')
    

    # -------------------------------------------------------------
    ## Saturation
    if SHOW_SEPARATE:
        for k, view in enumerate(Views):
            plt.figure()
            plt.hist(imageSaturation[view], bins=20, edgecolor='k', color=co[k], alpha=0.7)
            plt.title(f'Saturation Distribution - {view}')
    else:
        plt.figure()
        for k, view in enumerate(Views):
            plt.hist(imageSaturation[view], bins=20, edgecolor='k', color=co[k], alpha=0.7)
        plt.title(f'{name} Saturation Distribution (Combined)')
        plt.legend(Views)

    # Boxplot
    plt.figure()
    x=[imageSaturation[v] for v in Views]
    bpS = plt.boxplot(x, patch_artist=True, tick_labels=Views)
    for patch, color in zip(bpS['boxes'], co):
        patch.set_facecolor(color)
    for median in bpS['medians']:
        median.set(color='black', linewidth=2)
    plt.title(f'{name} Saturation Distribution')


    # -------------------------------------------------------------
    ## Brightness
    if SHOW_SEPARATE:
        for k, view in enumerate(Views):
            plt.figure()
            plt.hist(imageIlluminance[view], bins=20, edgecolor='k', color=co[k], alpha=0.7)
            plt.title(f'{name} Brightness Distribution - {view}')
    else:
        plt.figure()
        for k, view in enumerate(Views):
            plt.hist(imageIlluminance[view], bins=20, edgecolor='k', color=co[k], alpha=0.7)
        plt.title(f'{name} Brightness Distribution (Combined)')
        plt.legend(Views)

    # Boxplot
    plt.figure()
    x=[imageIlluminance[v] for v in Views]
    bpB = plt.boxplot(x, patch_artist=True, tick_labels=Views)
    for patch, color in zip(bpB['boxes'], co):
        patch.set_facecolor(color)
    for median in bpB['medians']:
        median.set(color='black', linewidth=2)
    plt.title(f'{name} Brightness Distribution')

    # -------------------------------------------------------------
    ## Camera ViewPoint
    if SHOW_SEPARATE:
        for k, view in enumerate(Views):
            plt.figure()
            plt.hist(plateAngle[view], bins=20, edgecolor='k', color=co[k], alpha=0.7)
            plt.title(f'{name} View Point Distribution - {view}')
    else:
        plt.figure()
        for k, view in enumerate(Views):
            plt.hist(plateAngle[view], bins=20, edgecolor='k', color=co[k], alpha=0.7)
        plt.title(f'{name} View Point Distribution (Combined)')
        plt.legend(Views)

    # Boxplot
    plt.figure()
    x=[plateAngle[v] for v in Views]
    bpV = plt.boxplot(x, patch_artist=True, tick_labels=Views)
    for patch, color in zip(bpV['boxes'], co):
        patch.set_facecolor(color)
    for median in bpV['medians']:
        median.set(color='black', linewidth=2)
    plt.title(f'{name} View Point Distribution')

    plt.show()  # final display of all plots



def main():
    # WE WILL COMPARE THREE DATASETS: real_plates, our own plates, and
    # an augmented version of our plates (modiffied properties).  
    
    SHOW_SEPARATE = False
    # DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
    Real_DataDir=r'data'
    Views=['Frontal','Lateral']
    Name = 'Real Data'
    R_plateArea, R_plateAngle, R_imageColor, R_imageIlluminance, R_imageSaturation = computeProperties(Real_DataDir, Views)
    Visualplots(Name, R_plateArea, R_plateAngle, R_imageColor, R_imageIlluminance, R_imageSaturation)


    Own_DataDir=r'data/Patentes'
    Name = 'Own Data'
    O_plateArea, O_plateAngle, O_imageColor, O_imageIlluminance, O_imageSaturation = computeProperties(Own_DataDir, Views)
    Visualplots(Name, O_plateArea, O_plateAngle, O_imageColor, O_imageIlluminance, O_imageSaturation)
    
    Augmented_DataDir=r'data/Patentes'
    Name = 'Augmented Data'
    Views_A=['FrontalAugmented','LateralAugmented']
    A_plateArea, A_plateAngle, A_imageColor, A_imageIlluminance, A_imageSaturation = computeProperties(Augmented_DataDir, Views_A)
    Visualplots(Name, A_plateArea, A_plateAngle, A_imageColor, A_imageIlluminance, A_imageSaturation)



if __name__ == "__main__":
    main()
