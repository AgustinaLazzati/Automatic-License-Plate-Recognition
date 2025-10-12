# import the necessary packages
import numpy as np
import cv2
import os

import matplotlib.pyplot as plt

class FeatureBlockBinaryPixelSum:
    def __init__(self, targetSize=(30, 15), blockSizes=((5, 5),)):
        # store the target size of the image to be described along with the set of block sizes
        self.targetSize = targetSize
        self.blockSizes = blockSizes
    
    def extract_pixel_features(self, image):
        raise Exception("This functionality is not implemented.")
        
    def extract_image_features(self, image):
        # resize the image to the target size and initialize the feature vector
        image = cv2.resize(image, (self.targetSize[1], self.targetSize[0]))
        
        # Ensure the image is binary
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        features = []

        stride_env = os.environ.get("STRIDE")
        stride = None
        if stride_env:
            try:
                stride = int(stride_env)
            except (ValueError, TypeError):
                pass

        if stride and stride > 0:
            # Use convolution with the given stride.
            # Convert image to float (0.0-1.0) for convolution
            img_float = image / 255.0
            
            for (blockW, blockH) in self.blockSizes:
                # Create a kernel for summation
                kernel = np.ones((blockH, blockW), dtype=np.float32)
                
                # Perform convolution. anchor=(0,0) means the kernel's top-left corner is aligned with the pixel.
                sum_image = cv2.filter2D(img_float, -1, kernel, anchor=(0, 0))

                # Subsample the result with the given stride
                for y in range(0, sum_image.shape[0] - blockH + 1, stride):
                    for x in range(0, sum_image.shape[1] - blockW + 1, stride):
                        total = sum_image[y, x] / (blockW * blockH)
                        features.append(total)
        else:
            # Default implementation (non-overlapping blocks)
            # loop over the block sizes
            for (blockW, blockH) in self.blockSizes:
                # loop over the image for the current block size
                for y in range(0, image.shape[0], blockH):
                    for x in range(0, image.shape[1], blockW):
                        # extract the current ROI, count the total number of non-zero pixels in the
                        # ROI, normalizing by the total size of the block
                        roi = image[y:y + blockH, x:x + blockW]
                        total = cv2.countNonZero(roi) / float(roi.shape[0] * roi.shape[1])

                        # update the feature vector
                        features.append(total)

        # return the features
        features = np.array(features)
        return features
