"""File to preprocess images before applying deconvolution methods."""
import numpy as np

def renormalize_to_minus_one_to_one(image):
    """
    Normalize image pixels to be between -1 and 1 (could be useful for some techniques).

    Input:
    image : 2D array
    
    Returns:
    renormalized_image : 2D array, image with pixels between -1 and 1.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return image

    renormalized_image = 2 * (image - min_val) / (max_val - min_val) - 1
    return renormalized_image
