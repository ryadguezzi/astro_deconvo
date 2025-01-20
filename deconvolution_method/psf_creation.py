"""File to create the power spectral density (PSD) of the image."""
import numpy as np

def generate_radial_psf(image_shape, radial_function, normalize=True):
    """
    Comppute radial PSF.

    Inputs:
    image_shape : tuple, shape of the image.
    radial_function : function, function defining the radial dependence of the PSF.
    normalize : bool, normalize the PSF so that the sum = 1.

    Returns :
    psf : 2D array of shape image_shape representing the radial PSF.
    """
    h, w = image_shape
    y, x = np.meshgrid(np.linspace(-h//2, h//2, h), np.linspace(-w//2, w//2, w), indexing='ij')

    r = np.sqrt(x**2 + y**2)
    psf = radial_function(r) #for example lambda r: np.exp(-r**2 / (2 * sigma**2)) for gaussian psf
    if normalize:
        psf /= np.sum(psf)
    return psf

def moffat_psf(image_shape, alpha=3, beta=2.5, normalize=True):
    """
    Compute Moffat PSF.

    Inputs:
    image_shape : tuple, shape of the image.
    alpha : float, width parameter of the PSF (related to the FWHM).
    beta : float, parameter of the PSF wing decay.
    normalize : bool, normalize the PSF so that the sum = 1.

    Returns :
    psf : 2D array of shape image_shape representing the Moffat PSF.
    """
    h, w = image_shape
    
    y, x = np.meshgrid(np.linspace(-h//2, h//2, h), np.linspace(-w//2, w//2, w), indexing='ij')
    r2 = x**2 + y**2
    
    psf = (1 + r2 / alpha**2) ** -beta
    if normalize:
        psf /= np.sum(psf)
    return psf