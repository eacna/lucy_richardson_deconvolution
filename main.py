#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script applies the Lucy-Richardson deconvolution algorithm with Total Variation regularization
to an image. The user can choose to process the image in grayscale or in RGB channels.

Usage:
    python script.py -i <image_path> --mode <grayscale|rgb>

Arguments:
    -i : Path to the input image
    --mode : Mode of processing: 'grayscale' for single channel (intensity), 'rgb' for all channels (R, G, B).
"""

import argparse
import cv2
import numpy as np
import os
import time
import warnings
import ntpath
from psfs import *  # Import PSF generation functions
from deconvRLTV import *  # Import deconvolution function

# Avoid warnings related to invalid or divide errors
np.seterr(divide='ignore', invalid='ignore')


def process_image(image_path, mode='grayscale'):
    """
    Load an image, process it in either grayscale or RGB mode, and perform Lucy-Richardson deconvolution.

    Args:
    - image_path: Path to the image file.
    - mode: 'grayscale' or 'rgb', defines whether to process the image in grayscale or all channels (RGB).

    Returns:
    - None (saves output image to file).
    """
    # Load image
    print("Loading image...")
    image = cv2.imread(image_path)
    head, tail = ntpath.split(image_path)
    first_name = os.path.splitext(tail)[0]

    # Check if image is loaded
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    print(f"Loaded image: {tail}")
    
    # Separate channels if mode is 'rgb', otherwise convert to grayscale
    if mode == 'rgb':
        r, g, b = image[:, :, 2], image[:, :, 1], image[:, :, 0]
    elif mode == 'grayscale':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        r = g = b = image  # Use the grayscale image for all channels
    else:
        raise ValueError("Invalid mode. Choose 'grayscale' or 'rgb'.")

    # PSF Estimation (Gaussian PSF in this case)
    print("Estimating PSF...")
    W = 363
    sigmax, sigmay = 43, 37
    PSF = fspecial_gauss2D((W, W), sigmax, sigmay)

    # Deconvolution parameters
    maxiter = 10000
    lapse = 2000
    lamb = 0.00201
    epsilon = 1e-7

    print("\nStarting Lucy-Richardson deconvolution...")

    # Start timer for performance tracking
    start = time.process_time()

    # Perform deconvolution for each channel (R, G, B) if in 'rgb' mode
    if mode == 'rgb':
        print("Processing R channel...")
        deconvRLTV(r, PSF, lamb, epsilon, maxiter, lapse, f"{first_name}_R.png")
        
        print("Processing G channel...")
        deconvRLTV(g, PSF, lamb, epsilon, maxiter, lapse, f"{first_name}_G.png")
        
        print("Processing B channel...")
        deconvRLTV(b, PSF, lamb, epsilon, maxiter, lapse, f"{first_name}_B.png")

        # After processing each channel, combine the results
        print("Combining R, G, B channels...")
        result = cv2.merge([cv2.imread(f"{first_name}_R.png"), 
                            cv2.imread(f"{first_name}_G.png"), 
                            cv2.imread(f"{first_name}_B.png")])

        # Save the combined result
        cv2.imwrite(f"{first_name}_deconv_rgb.png", result)

    elif mode == 'grayscale':
        # Perform deconvolution for grayscale image
        deconvRLTV(r, PSF, lamb, epsilon, maxiter, lapse, f"{first_name}_grayscale.png")

    print("\nDeconvolution completed.")
    print(f"Elapsed time: {time.process_time() - start} seconds")


def main():
    # Set up argument parsing
    ap = argparse.ArgumentParser(description="Lucy-Richardson Deconvolution with Total Variation Regularization")
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("--mode", choices=['grayscale', 'rgb'], default='grayscale', 
                    help="Mode for processing: 'grayscale' or 'rgb'. Default is 'grayscale'.")
    args = vars(ap.parse_args())

    # Process the image based on the provided arguments
    process_image(args["image"], args["mode"])


if __name__ == "__main__":
    main()
