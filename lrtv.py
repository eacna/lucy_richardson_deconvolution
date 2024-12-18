#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the implementation of the Lucy-Richardson Deconvolution
algorithm with Total Variation regularization.

Functions:
- entropy: Calculates the entropy of the image.
- deconvRLTV: Main function to perform deconvolution with regularization.
"""

import cv2
import numpy as np
import warnings
import os
from scipy.ndimage import convolve
from tqdm import tqdm

# Suppress specific warnings related to invalid or divide errors
np.seterr(divide='ignore', invalid='ignore')


def entropy(labels, base=None):
    """
    Calculate the entropy of an image.
    
    Args:
    - labels: Input image or array.
    - base: Base of logarithm for entropy calculation (default is None).
    
    Returns:
    - Entropy value of the input image.
    """
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()

    # Default to base e if not provided
    base = np.e if base is None else base

    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def deconvRLTV(image, psf, lamb=0.002, epsilon=1e-4, maxiter=20000, lapse=2000, imagename="processed.png"):
    """
    Perform Lucy-Richardson deconvolution with Total Variation regularization.
    
    Args:
    - image: Grayscale input image as a numpy array.
    - psf: Point Spread Function for deconvolution.
    - lamb: Regularization parameter, default is 0.002.
    - epsilon: Convergence threshold, default is 1e-4.
    - maxiter: Maximum number of iterations, default is 20000.
    - lapse: Frequency for saving intermediate results, default is 2000.
    - imagename: Output image filename.

    Returns:
    - The final deconvolved image saved to a file.
    """
    cont = 1  # Counter for saving images
    fn = np.double(image)
    psfbar = np.double(np.fliplr(np.flipud(psf)))  # Flipped PSF
    i = 0

    # Sobel filters for gradient calculation
    sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sy = np.transpose(sx)

    for i in tqdm (range (maxiter), desc="Iteration: "):
    # while i <= maxiter:
        gradx = cv2.filter2D(fn, -1, sx)
        grady = cv2.filter2D(fn, -1, sy)

        # Compute gradient magnitude
        gradabs = np.sqrt(np.square(gradx) + np.square(grady))
        gradabs[gradabs == 0] = 1  # Avoid division by zero

        # Normalize gradients
        gradx /= gradabs
        grady /= gradabs

        # Compute divergence of the gradient
        divgradx = cv2.filter2D(gradx, -1, sx)
        divgrady = cv2.filter2D(grady, -1, sy)
        divgrad = divgradx + divgrady
        regularization = 1 - lamb * divgrad
        minreg = np.min(regularization)

        # Check for potential issues with regularization parameter
        if minreg <= 0:
            warnings.warn('Please check the lambda value; lambda is the variation parameter.')
            print(f"Warning: Lambda is too high. Minimum regularization value: {minreg}")

        Den = cv2.filter2D(fn, -1, psf)
        Den[Den == 0] = 1e-99  # Avoid division by zero
        PreviousRes = image / Den

        # Calculate entropy for stopping criteria
        Sn0 = entropy(fn, 2)
        fn1 = fn * cv2.filter2D(PreviousRes, -1, psfbar) / regularization
        Sn1 = entropy(fn1, 2)

        # Check for convergence (based on image average)
        crit = np.abs(np.mean(fn1) - np.mean(fn)) / np.mean(fn)
        if crit < epsilon:
            print(f"Convergence reached after {i+1} iterations.")
            cv2.imwrite(f"{imagename}_deconv.png", fn1)
            return fn1

        fn = fn1

        # Save intermediate results at specified intervals
        if i % lapse == 0:
            cv2.imwrite(f"{imagename}_iter_{i+1}.png", fn)


    print("Deconvolution did not converge within the maximum number of iterations.")
    return fn  # Return final image
