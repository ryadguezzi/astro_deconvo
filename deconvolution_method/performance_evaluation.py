import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_psnr(original, restored):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        original (numpy.ndarray): The original reference image.
        restored (numpy.ndarray): The restored (deblurred) image.

    Returns:
        float: PSNR value in decibels (dB).
    """
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    max_pixel = 255.0  # Assuming 8-bit images
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


def compute_ssim(original, restored):
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Parameters:
        original (numpy.ndarray): The original reference image.
        restored (numpy.ndarray): The restored (deblurred) image.

    Returns:
        float: SSIM value (ranges from -1 to 1, where 1 indicates perfect similarity).
    """
    return ssim(original, restored, data_range=original.max() - original.min())