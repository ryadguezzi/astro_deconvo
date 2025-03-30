"""File to create the power spectral density (PSD) of the image."""
import numpy as np

def moffat_psf(image_shape, alpha=3, beta=2.5, normalize=True):
    """
    Compute Moffat PSF.

    Inputs:
    image_shape : tuple, shape of the image.
    alpha : float (in pixel), width parameter of the PSF (related to the FWHM).
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
        psf /= np.max(psf)
    return psf

def moffat_psf_local(shape, center_y, center_x, image_shape, alpha_0=3, beta_0=2.5, k_alpha=0.001, k_beta=0.005, 
                     beta_min=-np.inf, alpha_max=np.inf, normalize=True):
    """
    Generate a local Moffat PSF kernel for a specific position.
    Alpha increases and beta decreases with distance from the image center.
    
    Parameters:
    - shape: tuple, shape of the kernel
    - center_y, center_x: position in the padded image where this kernel will be applied
    - image_shape: tuple, shape of the original image (h, w)
    - alpha_0: float, central width parameter at the image center
    - beta_0: float, central wing decay parameter at the image center
    - k_alpha: float, rate of alpha increase with distance from center
    - k_beta: float, rate of beta decrease with distance from center
    - beta_min: float, minimum value for beta
    - alpha_max: float, maximum value for alpha
    - normalize: bool, normalize the PSF so that the sum = 1
    """
    h, w = shape
    img_h, img_w = image_shape
    
    # Create coordinate grid for the kernel
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Calculate distance from the specified center within the kernel
    r_kernel = np.sqrt((y - h//2)**2 + (x - w//2)**2)
    
    # Calculate distance from the image center to this position
    # Adjust coordinates to be relative to the original image center
    img_center_y, img_center_x = img_h//2, img_w//2
    
    # Calculate position in the original image (accounting for padding)
    pad = h // 2
    orig_pos_y = center_y - pad
    orig_pos_x = center_x - pad
    
    # Distance from image center to this position
    r_from_img_center = np.sqrt((orig_pos_y - img_center_y)**2 + (orig_pos_x - img_center_x)**2)
    
    # Define alpha and beta based on distance from image center
    # Alpha increases with distance from center
    alpha = np.minimum(alpha_0 + k_alpha * r_from_img_center, alpha_max)
    
    # Beta decreases with distance from center
    beta = np.maximum(beta_0 - k_beta * r_from_img_center, beta_min)
    
    # Compute Moffat PSF using the kernel's internal distance grid
    psf = (1 + (r_kernel / alpha) ** 2) ** (-beta)
    
    # Normalize the PSF
    if normalize:
        psf /= np.sum(psf)
    
    return psf