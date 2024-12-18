#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains functions to generate Point Spread Functions (PSF).
PSFs are used in image processing for deconvolution algorithms.

Functions:
- fspecial_gauss2D: Generates a 2D Gaussian PSF.
- fspecial_sinc2D: Generates a 2D Sinc PSF.
"""

import numpy as np

# Avoid warnings related to division by zero
np.seterr(divide='ignore', invalid='ignore')


def fspecial_gauss2D(shape=(3, 3), sigmax=0.5, sigmay=0.5):
    """
    Generates a 2D Gaussian Point Spread Function (PSF).
    
    Args:
    - shape: Tuple, the shape of the PSF (height, width).
    - sigmax: Float, standard deviation in the x-direction (controls the width).
    - sigmay: Float, standard deviation in the y-direction (controls the height).

    Returns:
    - h: 2D numpy array, the normalized Gaussian PSF.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x**2 / (2 * sigmax**2) + y**2 / (2 * sigmay**2)))

    # Apply threshold to remove small values
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    # Normalize the PSF
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return h


def fspecial_sinc2D(shape=(3, 3), sigmax=0.5, sigmay=0.5):
    """
    Generates a 2D Sinc Point Spread Function (PSF).
    
    Args:
    - shape: Tuple, the shape of the PSF (height, width).
    - sigmax: Float, standard deviation in the x-direction.
    - sigmay: Float, standard deviation in the y-direction.

    Returns:
    - h: 2D numpy array, the normalized Sinc PSF.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    rad = np.sqrt(x**2 * sigmax**2 + y**2 * sigmay**2)
    h = np.sinc(rad)

    # Apply threshold to remove small values
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    # Normalize the PSF
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return h
