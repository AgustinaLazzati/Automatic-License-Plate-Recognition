#!/usr/bin/env python

"""
Local Binary Patterns feature extractor
wrapper.

M5. Smart Data Knowledge / Analytics
Master in Internet of Things for eHealth
Universitat Autonoma de Barcelona

__author__ = "David Geronimo"
__license__ = "GPL"
__email__ = "research@davidgeronimo.com"
__year__ = "2019"
"""

from skimage import  feature
import numpy as np


class FeatureLBP:
    """
    Class for the computation of LBP descriptor: 
        radius: radius to the central pìxel for computation of LBP
        Default: radius=3
        method: Method to determine the pattern in feature.local_binary_pattern:
            ``default``
                Original local binary pattern which is grayscale invariant but not
                rotation invariant.
            ``ror``
                Extension of default pattern which is grayscale invariant and
                rotation invariant.
            ``uniform`` (default value)
                Uniform pattern which is grayscale invariant and rotation
                invariant, offering finer quantization of the angular space.
                For details, see [1]_.
            ``nri_uniform``
                Variant of uniform pattern which is grayscale invariant but not
                rotation invariant. For details, see [2]_ and [3]_.
        lbp_type:method used to aggregate LBP images into a single vector:
            ``simple``
                (default value) 
                Aggregation using a single histogram for the whole image
            ``block_lbp``
                Concatenation of block-wise histograms
        n_bins: Number of bins for histogram computation. 
        If no value is specified it is authomatically computed
    """
    def __init__(self,radius=3,method='uniform',n_bins=None,lbp_type='simple'):
        
        self.radius = radius
        self.n_points = 8 * self.radius
        self.method = method
        if n_bins is None:
          self.hist_bins()
        self.norm='l2' 
        self.lbp_type=lbp_type
        
    def hist_bins(self):
        
        # Determine bins/range
        if self.method in ('nri_uniform','uniform'):
            self.n_bins = self.n_points + 2
            self.hist_range = (0,self.n_bins + 2)
        elif self.method in ('default','ror'):
            self.n_bins = 2 ** self.n_points
            self.hist_range = (0, 2 **self.n_bins)
    
    def block_lbp(self,lbp,grid=(4,4)):
        H, W, K = lbp.shape
        gy, gx = grid
        ys = np.linspace(0, H, gy+1, dtype=int)
        xs = np.linspace(0, W, gx+1, dtype=int)
        desc_parts = []
        for iy in range(gy):
            for ix in range(gx):
                patch = lbp[ys[iy]:ys[iy+1], xs[ix]:xs[ix+1]]
                hist, _ = np.histogram(patch.ravel(), bins=self.n_bins, range=self.hist_range)
                # normalize per block
                if self.norm == 'l1':
                    s = hist.sum()
                    hist = hist / s if s > 0 else hist
                elif self.norm == 'l2':
                    s = np.linalg.norm(hist)
                    hist = hist / s if s > 0 else hist
                desc_parts.append(hist)

        return np.concatenate(desc_parts)

    def simple_lbp(self,lbp):
        
        hist, _ = np.histogram(lbp.ravel(), bins=self.n_bins, 
                               range=self.hist_range)

       # normalize
        if  self.norm == 'l1':
           s = hist.sum()
           hist = hist / s if s > 0 else hist
        elif self.norm == 'l2':
           s = np.linalg.norm(hist)
           hist = hist / s if s > 0 else hist
       
        return hist
    
    def extract_pixel_features(self, image):
        """ 
        Receives a 2D grayscale image, returns a W x H x 1 (every pixel)
        """
        features = feature.local_binary_pattern(image, self.n_points, self.radius, self.method)
        features = np.expand_dims(features, axis=2)
        return features

    def extract_image_features(self, image):
        """
        Calls the pixel features extractor and then aggregates the matrix into a vector using 
        either a global histogram (lbp_type='simple') or a concatenation of block-wise histograms (lbp_type='block_lbp')
        """
        pixel_features = self.extract_pixel_features(image)
        if self.lbp_type=='simple':
            lbp=self.simple_lbp(pixel_features)
        else:
            lbp=self.block_lbp(pixel_features)
        
        return lbp
        