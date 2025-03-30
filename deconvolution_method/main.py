"""Main script to run the whole pipeline"""

import sys

from richardson_lucy_deconvolution import deconvolution_richardson_methods
from wiener_deconvolution import deconvolution_wiener_methods
from blurred_functions import spatially_variant_convolution
from moffat_psf import moffat_psf_local
from find_psf_parameters import find_parameters

def create_psf_generator(image_shape, alpha_0=3, beta_0=2.5, k_alpha=1, k_beta=0.02, 
                         beta_min=2.5, alpha_max=6):
    """
    Creates a PSF generator function for spatially variant deconvolution
    
    Parameters:
    - image_shape: dimensions of the full image (height, width)
    - alpha_0, beta_0: baseline Moffat parameters at image center
    - k_alpha, k_beta: rate of parameter change with distance
    - beta_min, alpha_max: parameter limits
    
    Returns:
    - Function that generates PSFs for specific positions
    """
    def generator(shape, center_y, center_x):
        return moffat_psf_local(
            shape, center_y, center_x, image_shape,
            alpha_0=alpha_0, beta_0=beta_0,
            k_alpha=k_alpha, k_beta=k_beta,
            beta_min=beta_min, alpha_max=alpha_max
        )
    
    return generator

if __name__ == "__main__":
    """Run script with optional arguments:  
    Usage:  
    python script.py <original_image_path> <parameters_to_find>
    <original_image_path>: Path to the original image (required)
    <parameters_to_find>: 'yes' or 'no' to find Moffat parameters (optional, default is 'no')

    Arguments:    
    """
    original_image_path = sys.argv[1]
    if len(sys.argv) > 2:
        parameters_to_find_str = sys.argv[2].lower()
        parameters_to_find = parameters_to_find_str == 'yes'
    else:
        parameters_to_find = False
    
    print("Starting convolution")
    original_image, blurred_image, constant_blurred_image = spatially_variant_convolution(original_image_path, kernel_size=50)
    
    if parameters_to_find == True :
        print("Finding Moffat parameters")
        moffat_parameters = find_parameters(blurred_image, verbose=False)
        alpha_0 = moffat_parameters['Parameter0']['intercept']
        beta_0 = moffat_parameters['Parameter1']['intercept']
        k_alpha =moffat_parameters['Parameter0']['slope']
        k_beta =moffat_parameters['Parameter1']['slope']
        moffat_generator = create_psf_generator((50,50),alpha_0=alpha_0, beta_0=beta_0,
                k_alpha=k_alpha, k_beta=k_beta)
    else:
        print("Using default values for alpha and beta")
        moffat_generator = create_psf_generator((50,50),alpha_0=1, beta_0=7,
                k_alpha=0.1, k_beta=0.02,
                beta_min=2.5, alpha_max=6)
    print("Starting deconvolution")
    richardson_deconvolved_images = deconvolution_richardson_methods(blurred_image, moffat_generator, original_image=original_image, kernel_size=15, step=20, iterations = 20)
    wiener_deconvolved_images = deconvolution_wiener_methods(blurred_image, moffat_generator, original_image=original_image, kernel_size=15, step=20)