#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script applies the Lucy-Richardson deconvolution algorithm with Total Variation regularization
to an image. The user can choose to process the image in grayscale or in RGB channels.
Additionally, the user can specify the PSF type (Gaussian or Sinc), and the PSF size parameters.

Usage:
    python script.py -i <image_path> --mode <grayscale|rgb> --psf_type <gaussian|sinc> --psf_size W,SX,SY

Arguments:
    -i : Path to the input image
    --mode : Mode of processing: 'grayscale' for single channel (intensity), 'rgb' for all channels (R, G, B).
    --psf_type : Type of PSF: 'gaussian' or 'sinc'
    --psf_size : PSF size and standard deviation parameters in the form W,SX,SY (Width, Sigmax, Sigmay)
    --maxiter : 
"""

import argparse
import cv2
import numpy as np
import os
import time
import warnings
import ntpath
from psfs import *  # Import PSF generation functions
from lrtv import *  # Import deconvolution function

# Avoid warnings related to invalid or divide errors
np.seterr(divide='ignore', invalid='ignore')


def process_image(image_path, mode='grayscale', psf_type='gaussian', psf_size=(363, 43, 37), maxiter = 100, lapse = 10, lambda_var = 0.00201):
    """
    Load an image, process it in either grayscale or RGB mode, and perform Lucy-Richardson deconvolution.

    Args:
    - image_path: Path to the image file.
    - mode: 'grayscale' or 'rgb', defines whether to process the image in grayscale or all channels (RGB).
    - psf_type: Type of PSF ('gaussian' or 'sinc').
    - psf_size: Tuple of (W, Sigmax, Sigmay) for PSF size and parameters.

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

    # PSF Estimation
    W, sigmax, sigmay = psf_size
    print(f"Generating {psf_type} PSF with size ({W}, {sigmax}, {sigmay})...")
    if psf_type == 'gaussian':
        PSF = fspecial_gauss2D((W, W), sigmax, sigmay)
    elif psf_type == 'sinc':
        PSF = fspecial_sinc2D((W, W), sigmax, sigmay)
    else:
        raise ValueError("Invalid PSF type. Choose 'gaussian' or 'sinc'.")

    # Deconvolution parameters
    
    epsilon = 1e-7

    print("\nStarting Lucy-Richardson deconvolution...")

    # Start timer for performance tracking
    start = time.process_time()

    # Perform deconvolution for each channel (R, G, B) if in 'rgb' mode
    if mode == 'rgb':
        print("Processing R channel...")
        deconvRLTV(r, PSF, lambda_var, epsilon, maxiter, lapse, f"{head}/{first_name}_R")
        
        print("Processing G channel...")
        deconvRLTV(g, PSF, lambda_var, epsilon, maxiter, lapse, f"{head}/{first_name}_G")
        
        print("Processing B channel...")
        deconvRLTV(b, PSF, lambda_var, epsilon, maxiter, lapse, f"{head}/{first_name}_B")

        # After processing each channel, combine the results
        print("Combining R, G, B channels...")
        result = cv2.merge([cv2.imread(f"{head}/{first_name}_R"), 
                            cv2.imread(f"{head}/{first_name}_G"), 
                            cv2.imread(f"{head}/{first_name}_B")])

        # Save the combined result
        cv2.imwrite(f"{head}/{first_name}_deconv_rgb.png", result)

    elif mode == 'grayscale':
        # Perform deconvolution for grayscale image
        deconvRLTV(r, PSF, lambda_var, epsilon, maxiter, lapse, f"{head}/{first_name}_grayscale")

    print("\nDeconvolution completed.")
    print(f"Elapsed time: {time.process_time() - start} seconds")


def main():
    # Set up argument parsing
    ap = argparse.ArgumentParser(description="Lucy-Richardson Deconvolution with Total Variation Regularization")
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("--mode", choices=['grayscale', 'rgb'], default='grayscale', 
                    help="Mode for processing: 'grayscale' or 'rgb'. Default is 'grayscale'.")
    ap.add_argument("--psf_type", choices=['gaussian', 'sinc'], default='gaussian', 
                    help="Type of PSF: 'gaussian' or 'sinc'. Default is 'gaussian'.")
    ap.add_argument("--psf_size", type=str, default="363,43,37", 
                    help="PSF size parameters (W,SX,SY) as a comma-separated string. Default is '363,43,37'.")
    ap.add_argument("--maxiter", type=int, default=100, 
                    help="Number of iterations, default is 100.")
    ap.add_argument("--lapse", type=int, default=10, 
                    help="Number of iterations to wait to save images, default is 10.")
    ap.add_argument("--lambda_var", type=float, default=0.00201, 
                    help="Number of iterations to wait to save images, default is 10.")

    args = vars(ap.parse_args())

    # Parse PSF size
    psf_size = tuple(map(int, args["psf_size"].split(',')))
    print('')
    print('Args', args)

    # Process the image based on the provided arguments
    process_image(args["image"], args["mode"], args["psf_type"], psf_size, args["maxiter"], args["lapse"], args["lambda_var"])


if __name__ == "__main__":
    main()
