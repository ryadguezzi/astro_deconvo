"""File to plot blurred and deblurred images."""
import numpy as np
import matplotlib.pyplot as plt

def plot_psf(psf, name):
    """Plot the PSF"""

    # Affichage de la PSF
    plt.figure(figsize=(5, 5))
    plt.title("PSF radiale (2D)")
    plt.imshow(psf, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(f'./deconvolution_method/{name}.png')

def plot_original_blurred_images(original_image, blurred_image, name):
    """Plot the original and blurred images"""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image originale")
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Image floue")
    plt.imshow(blurred_image, cmap='gray')
    plt.axis('off')
    plt.savefig(f'./images/blurred/{name}.png')

def plot_blurred_deblurred_images(blurred_image, deblurred_image, name):
    """Plot the blurred and deblurred images"""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image floue")
    plt.imshow(blurred_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Image défloutée")
    plt.imshow( deblurred_image, cmap='gray')
    plt.axis('off')
    plt.savefig(f'./images/deblurred/{name}.png')
