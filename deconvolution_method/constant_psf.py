import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from pre_processing import resize_image
from psf_creation import moffat_psf
from plot import plot_psf, plot_original_blurred_images, plot_blurred_deblurred_images
import sys

def create_blurred_image(image, psf, target_size = (128, 128), mode = "same"):
    """Create a blurred image by convolving the image with the PSF.
    
    Inputs :
    image : 2D array, the original image.
    psf : 2D array, the point spread function.
    target_size : tuple, size of the target image (default = (128, 128)).
    mode : str, mode of the convolution (default = 'same').
    
    Returns :
    blurred_image : 2D array, the blurred image.
    """
    resized_image = resize_image(image, target_size)
    if resized_image.ndim == 3:
        resized_image = np.mean(resized_image, axis=2)
    blurred_image = convolve2d(resized_image, psf, mode=mode)
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
    small_value = 1e-1
    deconvolved_image = np.real(np.fft.ifft2(G * np.conj(H) / (H * np.conj(H) + small_value)))
    return deconvolved_image

if __name__ == "__main__":
    """Main function to deblur an image with Wiener deconvolution.
    To run use <python constant_psf.py image_filename target_size>"""
    filename = sys.argv[1]
    image = plt.imread(filename)
    target_size = (128,128)
    psf = moffat_psf(target_size)
    blurred_image = create_blurred_image(image, psf, target_size)
    deblurred_image = wiener_deconvolution(blurred_image, psf)
    plot_psf(psf, filename.split('/')[-1].split('.')[0])
    plot_original_blurred_images(image, blurred_image, filename.split('/')[-1].split('.')[0])
    plot_blurred_deblurred_images(blurred_image, deblurred_image, filename.split('/')[-1].split('.')[0])
