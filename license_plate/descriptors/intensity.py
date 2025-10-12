#!/usr/bin/env python

"""
Use image original intensity.

M5. Smart Data Knowledge / Analytics
Master in Internet of Things for eHealth
Universitat Autonoma de Barcelona

__author__ = "David Geronimo"
__license__ = "GPL"
__email__ = "research@davidgeronimo.com"
__year__ = "2019"
"""

import numpy as np


class FeatureIntensity:

    def __init__(self):
        self.threshold = 128

    def extract_pixel_features(self, image):
        """ 
        Receives a 2D grayscale image, returns a W x H x 1 (every pixel)
        """
        image = np.expand_dims(image, axis=2)
        return image

    def extract_image_features(self, image):
        """
        Calls the pixel features extractor and then flattens the matrix into a vector
        """
        #pixel_features = self.extract_pixel_features(image)
        return 0
        