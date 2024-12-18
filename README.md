# Lucy-Richardson Deconvolution with Total Variation Regularization (Python, OpenCV)

This project provides an implementation of the **Lucy-Richardson Deconvolution Algorithm** with **Total Variation (TV)** regularization. It can be used to deblur images, either in grayscale or in individual RGB channels. Additionally, the script allows users to specify the **Point Spread Function (PSF)** used in the deconvolution process, with options to choose a **Gaussian** or **Sinc** PSF and customize its size.

## Features

- Perform deconvolution on grayscale or RGB images.
- Support for two PSF types: **Gaussian** and **Sinc**.
- Customizable PSF size (Width, Sigmax, Sigmay).
- Flexible parameters for the deconvolution process (e.g., regularization parameter `lambda`, tolerance `epsilon`).
- Save the resulting deconvolved images in different formats.
  
## Requirements

To run this project, you'll need:

- Python 3.x
- Required Python libraries:
  - **OpenCV** (`cv2`)
  - **NumPy**
  - **SciPy**
  - **argparse**
  - **matplotlib** (optional, for visualization)

You can install the necessary dependencies using `pip`:

```bash
pip install opencv-python numpy scipy argparse matplotlib
```

# Files in the Project
## 1. deconvRLTV.py
This file contains the main implementation of the Lucy-Richardson Deconvolution Algorithm with Total Variation regularization. The function deconvRLTV() is used to process images with the deconvolution algorithm.

## 2. psfs.py
This file contains functions to generate different Point Spread Functions (PSFs), specifically Gaussian and Sinc. The function fspecial_gauss2D() creates a 2D Gaussian PSF, and fspecial_sinc2D() creates a 2D Sinc PSF.

## 3. main.py (or any script you choose to run)
This file serves as the main entry point for the deconvolution process. It takes command-line arguments, loads the image, generates the appropriate PSF, and performs the deconvolution on the image.

# How to Use
## 1. Prepare Your Image
Make sure you have an image that you want to deconvolve. The image can be in any format supported by OpenCV (e.g., .jpg, .png).

## 2. Run the Script
You can run the script using the following command:

```bash
python main.py -i <image_path> --mode <grayscale|rgb> --psf_type <gaussian|sinc> --psf_size <W,SX,SY>
```

Where:
```-i <image_path>:``` Path to the input image file.
``` --mode <grayscale|rgb>: ``` Specifies whether to process the image in grayscale or RGB mode. Default is grayscale.
``` --psf_type <gaussian|sinc>:```  Specifies the type of Point Spread Function (PSF). Choose between gaussian or sinc. Default is gaussian.
``` --psf_size <W,SX,SY>:```  The size of the PSF. Provide three comma-separated integers:
``` W:```  Width and Height of the PSF.
``` SX:```  Sigma (standard deviation) in the X direction.
``` SY:```  Sigma (standard deviation) in the Y direction. Example: --psf_size 363,43,37.
W, SX, and SY are obligated parameters when --psf_type gaussian

# Example 1: Deconvolution in Grayscale with a Gaussian PSF
```bash
python main.py -i Examples/image.png --mode grayscale --psf_type sinc  --psf_size 350,45,45 --maxiter 500 --lapse 2 --lambda_var 0.0019
```
Check Results in Examples directory

## 3. Check the Output
After running the script, the output images will be saved in the same directory as the input image with the following naming conventions:

For grayscale mode, the resulting image will be saved as image_deconv_grayscale.png.
For RGB mode, the resulting images for each channel (R, G, B) will be saved as:
image_deconv_R.png
image_deconv_G.png
image_deconv_B.png
The combined result will be saved as image_deconv_rgb.png.
## 4. Parameters for Deconvolution
You can adjust the following parameters for the deconvolution process:

```maxiter:``` The maximum number of iterations (default is 10000).
```lapse:``` How often to save intermediate images (default is 2000 iterations).
```lambda:``` Regularization parameter (default is 0.00201).
```epsilon:``` Convergence threshold (default is 1e-7).

## 5. Visualization (Optional)
If you would like to visualize the intermediate steps of the deconvolution process, you can modify the script to display images using matplotlib.

# How the Code Works
## PSF Generation
The Point Spread Function (PSF) is used in the deconvolution process to reverse the blurring applied to the image. You can generate a PSF in two ways:

Gaussian PSF: Created using the function fspecial_gauss2D(), which generates a 2D Gaussian kernel.
Sinc PSF: Created using the function fspecial_sinc2D(), which generates a 2D Sinc kernel.

## Deconvolution Process
The deconvolution is performed using the Lucy-Richardson algorithm with Total Variation regularization to improve the quality of the result. The algorithm iteratively refines the estimate of the sharp image by considering both the observed blurred image and the PSF.

The regularization term controls how much sharpness is enforced on the output image.

# Notes
Ensure that the input image is of good quality. A low-quality or excessively noisy image may lead to poor deconvolution results.
The script is flexible and can handle both single-channel (grayscale) and multi-channel (RGB) images.
The PSF size and sigma parameters are crucial for the deconvolution quality. Experiment with different values to get the best result for your image.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgements
The Lucy-Richardson deconvolution algorithm is a well-established technique for image deblurring, and it is referenced in numerous image processing works.
The Total Variation (TV) regularization used in this project helps to remove noise while preserving important image details.

