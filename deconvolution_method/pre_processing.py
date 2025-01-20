"""File to preprocess images before applying deconvolution methods."""

from skimage.transform import resize
import numpy as np

def resize_image(image, target_size=(255, 255)):
    """
    Resize image to target size.

    Inputs:
    image : 2D array
    target_size : tuple, new size of the image (height, width).

    Returns :
    resized_image : resized 2D array.
    """
    if image.shape[:2] != target_size:
        resized_image = resize(image, target_size, preserve_range=True, anti_aliasing=True)
        return resized_image.astype(image.dtype)
    return image

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

    renormalized_image = 2 * (image - min_val) / (max_val - min_val) - 1
    return renormalized_image