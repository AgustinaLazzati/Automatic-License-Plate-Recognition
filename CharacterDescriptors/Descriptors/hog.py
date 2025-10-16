#!/usr/bin/env python

"""
Histograms of Oriented Gradients feature extraction
wrapper.


Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres"
__license__ = "GPL"
__email__ = "debora,gtorres@cvc.uab.es"


"""

from skimage import feature, transform
import numpy as np
import cv2


class FeatureHOG:
    """
    Class for the computation of HOG descriptor (using feature.hog):
        orientations: Number of orientations for gradient computation
        Default: orientations=8
        pixels_per_cell: Size of the cell for orientation computation (default, (4,4))
        cells_per_block: Cells for each block for cell histograms normalization
    """

    def __init__(
        self,
        orientations=8,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True,
        targetSize=(30, 15),
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = "L2-Hys"
        self.feature_vector = feature_vector
        self.targetSize = targetSize

    def extract_pixel_features(self, image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.targetSize[1], self.targetSize[0]))
        image = np.pad(image, 5, mode="constant")
        image = transform.rescale(image, 2)
        feats, feats_im = feature.hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            visualize=True,
            feature_vector=self.feature_vector,
        )
        return feats_im

    def extract_image_features(self, image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.targetSize[1], self.targetSize[0]))
        image = np.pad(image, 5, mode="constant")
        image = transform.rescale(image, 2)
        feats = feature.hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            visualize=False,
            feature_vector=True,
        )
        return feats