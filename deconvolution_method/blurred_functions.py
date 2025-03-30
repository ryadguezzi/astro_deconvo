"""File to spatially variant convolve image with Moffat PSF"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d

from moffat_psf import moffat_psf_local

def spatially_variant_convolve(image, kernel_size=15, step=None, alpha_0=3, beta_0=2.5, 
                             k_alpha=0.001, k_beta=0.005, beta_min=1.5, alpha_max=5):
    """
    Apply spatially variant convolution to an image with improved patch handling
    
    Parameters:
    - image: 2D array, input image
    - kernel_size: size of the PSF kernels (odd number)
    - step: step size for sampling PSFs (if None, use kernel_size//2)
    - alpha_0, beta_0: baseline parameters at image center
    - k_alpha: rate of increase for alpha with distance from center
    - k_beta: rate of decrease for beta with distance from center
    
    Returns:
    - convolved image with Moffat PSFs
    """
    if step is None:
        step = kernel_size // 2
        
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    h, w = image.shape
    pad = kernel_size // 2
    
    # Create weight map for blending
    weight_sum = np.zeros((h, w), dtype=float)
    result = np.zeros((h, w), dtype=float)
    
    # Pad the input image
    padded_image = np.pad(image, pad, mode='symmetric')
    
    # Create a grid of PSF sampling points - ensure we cover the full image
    y_positions = np.arange(0, h, step)
    if y_positions[-1] < h - 1:
        y_positions = np.append(y_positions, h - 1)
    
    x_positions = np.arange(0, w, step)
    if x_positions[-1] < w - 1:
        x_positions = np.append(x_positions, w - 1)
    
    # Create a wrapper function for the PSF generation
    def psf_wrapper(shape, center_y, center_x):
        return moffat_psf_local(
            shape, center_y, center_x, image.shape,
            alpha_0=alpha_0, beta_0=beta_0,
            k_alpha=k_alpha, k_beta=k_beta,
            beta_min=beta_min, alpha_max=alpha_max
        )
    
    # For each sampling position
    for y_pos in y_positions:
        for x_pos in x_positions:
            # Generate the local PSF for this position
            #cy, cx = y_pos + pad, x_pos + pad  # Adjust for padding
            local_psf = psf_wrapper((kernel_size, kernel_size), y_pos, x_pos)
            
            # Define the patch region to process
            y_start = max(0, y_pos - step)
            y_end = min(h, y_pos + step + 1)
            x_start = max(0, x_pos - step)
            x_end = min(w, x_pos + step + 1)
            
            # Extract the region from padded image (accounting for padding)
            pad_y_start = y_start + pad
            pad_y_end = y_end + pad
            pad_x_start = x_start + pad
            pad_x_end = x_end + pad
            
            local_region = padded_image[pad_y_start - pad:pad_y_end + pad, 
                                        pad_x_start - pad:pad_x_end + pad]
            
            # Convolve the local region
            local_result = convolve2d(local_region, local_psf, mode='same', boundary='symm')
            
            # Create a weight map for smooth blending - higher weight near the center
            # Using a radial Gaussian weight
            y_grid, x_grid = np.mgrid[y_start:y_end, x_start:x_end]
            
            # Calculate distance from current position
            dist_y = y_grid - y_pos
            dist_x = x_grid - x_pos
            distances = np.sqrt(dist_y**2 + dist_x**2)
            
            # Convert to weights - closer points get higher weight
            sigma = step/2  # Controls falloff rate
            weights = np.exp(-(distances**2) / (2 * sigma**2))
            
            # Add the weighted contribution
            # We need to extract the central part of the convolution result
            local_y_size = y_end - y_start
            local_x_size = x_end - x_start
            
            center_result = local_result[pad:pad+local_y_size, pad:pad+local_x_size]
            
            result[y_start:y_end, x_start:x_end] += center_result * weights
            weight_sum[y_start:y_end, x_start:x_end] += weights
    
    # Normalize by weights to get final image
    # Avoid division by zero
    weight_sum = np.where(weight_sum > 0, weight_sum, 1)
    result = result / weight_sum
    
    return result



################################# TEST #################################
# Test the spatially variant convolution function
def spatially_variant_convolution(image_path, kernel_size=21):
    """ Convolve an image with a spatially variant Moffat PSF
    Input:
    - image_path (str): path to the input image
    - kernel_size: size of the PSF kernel (odd number)
    
    Returns:
    - original image (2darray)
    - convolved image (2darray) with spatially variant PSF
    - convolved image (2darray) with standard PSF"""
    alpha_0=1      # Alpha at center 
    beta_0=7     # Beta at center
    k_alpha=0.1  # Rate of alpha increase 
    k_beta=0.02   # Rate of beta decrease
    beta_min=2.5  # Minimum beta value
    alpha_max=6

    image = plt.imread(image_path)
    image = image.mean(axis=2)
    size = image.shape[0]

    # Apply spatially variant convolution with increasing alpha and decreasing beta
    result = spatially_variant_convolve(
        image, 
        kernel_size=kernel_size, 
        alpha_0=alpha_0,      # Alpha at center 
        beta_0=beta_0,     # Beta at center
        k_alpha=k_alpha,   # Rate of alpha increase 
        k_beta=k_beta,    # Rate of beta decrease
        beta_min=beta_min,   # Minimum beta value
        alpha_max=6     # Maximum alpha value
    )
    
    # For comparison, also do standard convolution with a single PSF from the center
    center_kernel_size = kernel_size
    center_psf = moffat_psf_local(
        (center_kernel_size, center_kernel_size), 
        center_kernel_size//2, center_kernel_size//2, 
        image.shape,
        alpha_0=alpha_0, beta_0=beta_0, 
        k_alpha=k_alpha, k_beta=k_beta,
        beta_min=beta_min, alpha_max=alpha_max
    )
    standard_result = convolve2d(image, center_psf, mode='same', boundary='symm')
    
    # Visualize PSFs at different locations to show the variation
    psf_size = kernel_size
    center_y, center_x = size//2, size//2  # Image center
    edge_y, edge_x = size-1, size-1        # Bottom-right corner
    
    center_psf = moffat_psf_local(
        (psf_size, psf_size), 
        size//2, size//2, 
        image.shape,
        alpha_0=alpha_0, beta_0=beta_0, 
        k_alpha=k_alpha, k_beta=k_beta,
        beta_min=beta_min, alpha_max=alpha_max
    )
    
    edge_psf = moffat_psf_local(
        (psf_size, psf_size), 
        edge_y, edge_x, 
        image.shape,
        alpha_0=alpha_0, beta_0=beta_0, 
        k_alpha=k_alpha, k_beta=k_beta,
        beta_min=beta_min, alpha_max=alpha_max
    )

    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='grey')
    plt.title('Original Image')
    plt.colorbar()
    
    plt.subplot(2, 3, 2)
    plt.imshow(standard_result, cmap='grey')
    plt.title('Standard Convolution')
    plt.colorbar()
    
    plt.subplot(2, 3, 3)
    plt.imshow(result, cmap='grey')
    plt.title('Spatially Variant Convolution\n(α↑, β↓ from center)')
    plt.colorbar()
    
    plt.subplot(2, 3, 4)
    plt.imshow(center_psf, cmap='hot')
    plt.title('PSF at Image Center\n(α={:.2f}, β={:.2f})'.format(1, 10))
    plt.colorbar()
    
    # Calculate the actual alpha, beta at edge
    center_dist = np.sqrt((edge_y - center_y)**2 + (edge_x - center_x)**2)
    edge_alpha = min(alpha_0 +k_alpha * center_dist, alpha_max)
    edge_beta = max(beta_0 - k_beta * center_dist, beta_min)
    
    plt.subplot(2, 3, 5)
    plt.imshow(edge_psf, cmap='hot')
    plt.title('PSF at Image Edge\n(α={:.2f}, β={:.2f})'.format(edge_alpha, edge_beta))
    plt.colorbar()
    
    plt.subplot(2, 3, 6)
    # Show cross-section of both PSFs
    plt.plot(center_psf[psf_size//2, :], 'b-', label='Center PSF')
    plt.plot(edge_psf[psf_size//2, :], 'r-', label='Edge PSF')
    plt.title('PSF Cross-sections')
    plt.legend()
    
    plt.savefig('./deconvolution_method/plots/spatially_variant_convolution.png')
    
    return image, result, standard_result