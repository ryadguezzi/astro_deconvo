import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from pre_processing import resize_image, renormalize_to_minus_one_to_one
from psf_creation import moffat_psf
from plot import plot_psf, plot_original_blurred_images, plot_blurred_deblurred_images, plot_original_deblurred_images
from skimage.restoration import unsupervised_wiener

import sys

def create_blurred_image(image, psf, mode = "same"):
    """Create a blurred image by convolving the image with the PSF.
    
    Inputs :
    image : 2D array, the original image.
    psf : 2D array, the point spread function.
    target_size : tuple, size of the target image (default = (128, 128)).
    mode : str, mode of the convolution (default = 'same').
    
    Returns :
    blurred_image : 2D array, the blurred image.
    """
    blurred_image = convolve2d(image, psf, mode=mode)
    return blurred_image

def wiener_deconvolution(blurred_image, psf):
    """Apply Wiener deconvolution to the blurred image.
    
    Inputs :
    blurred_image : 2D array, the blurred image.
    psf : 2D array, the point spread function.
    
    Returns :
    deconvolved_image : 2D array, the deconvolved image.
    """
    H = np.fft.fft2(np.fft.fftshift(psf), s=blurred_image.shape)
    G = np.fft.fft2(blurred_image)
    
    # Wiener filter : adjustable regularizer (small_value to limit artifacts)
    small_value = 1e-2
    deconvolved_image = np.real(np.fft.ifft2(G * np.conj(H) / (H * np.conj(H) + small_value)))
    return deconvolved_image

def upgreated_wiener(blurred_image, psf):
    """Deconvolution with a Wiener-Hunt approach, where the hyperparameters are automatically estimated.
    
    Inputs : 
    blurred_image : 2D array, the blurred image.
    psf : 2D array, the point spread function.
    
    Returns :
    deconvolved_image : 2D array, the deconvolved image.
    chains : dict, the keys noise and prior contain the chain list of noise and prior precision respectively."""
    renormalized_blurred_image = renormalize_to_minus_one_to_one(blurred_image)
    deconvolved_image, chains = unsupervised_wiener(renormalized_blurred_image, psf)
    return deconvolved_image, chains

if __name__ == "__main__":
    """Main function to deblur an image with Wiener deconvolution.
    To run use <python constant_psf.py image_filename target_size>"""
    filename = sys.argv[1]
    image = plt.imread(filename)
    target_size = (128,128)
    resized_image = resize_image(image, target_size)
    if resized_image.ndim == 3:
        resized_image = np.mean(resized_image, axis=2)
    psf = moffat_psf(target_size)
    blurred_image = create_blurred_image(resized_image, psf)
    deblurred_image= upgreated_wiener(blurred_image, psf)[0]
    plot_psf(psf, filename.split('/')[-1].split('.')[0])
    plot_original_blurred_images(resized_image, blurred_image, filename.split('/')[-1].split('.')[0])
    plot_blurred_deblurred_images(blurred_image, deblurred_image, filename.split('/')[-1].split('.')[0])
    plot_original_deblurred_images(resized_image, deblurred_image, filename.split('/')[-1].split('.')[0])
