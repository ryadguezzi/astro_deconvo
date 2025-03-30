import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from performance_evaluation import compute_psnr, compute_ssim

def regularized_richardson_lucy(blurred_image, psf_generator, kernel_size=15, step=None,
                              iterations=20, lambda_reg=0.01):
    """
    Apply Richardson-Lucy deconvolution with regularization for spatially variant PSF
    
    Parameters:
    - blurred_image: 2D array, blurred input image
    - psf_generator: function that generates PSF for a specific position
    - kernel_size: size of the PSF kernels
    - step: step size for sampling PSFs
    - iterations: number of Richardson-Lucy iterations
    - lambda_reg: regularization parameter
    
    Returns:
    - deconvolved image
    """
    if step is None:
        step = kernel_size // 2
        
    h, w = blurred_image.shape
    pad = kernel_size // 2
    
    # Create a grid of PSF sampling points
    y_positions = np.arange(0, h, step)
    x_positions = np.arange(0, w, step)
    
    # Initialize estimate
    estimate = np.ones_like(blurred_image) * np.mean(blurred_image)
    
    # For each iteration
    for iter in range(iterations):
        # Construct total error correction term
        relative_blur = np.zeros_like(blurred_image)
        weights = np.zeros_like(blurred_image) + 1e-10
        
        # For each position
        for y_idx, cy in enumerate(y_positions):
            for x_idx, cx in enumerate(x_positions):
                # Generate the local PSF for this position
                local_psf = psf_generator((kernel_size, kernel_size), cy + pad, cx + pad)
                # Make sure PSF is flipped for correlation
                local_psf_flipped = np.flip(np.flip(local_psf, 0), 1)
                
                # Define the region for this PSF
                if y_idx < len(y_positions) - 1:
                    y_end = min(y_positions[y_idx + 1], h)
                else:
                    y_end = h
                    
                if x_idx < len(x_positions) - 1:
                    x_end = min(x_positions[x_idx + 1], w)
                else:
                    x_end = w
                
                y_start = y_positions[y_idx]
                x_start = x_positions[x_idx]
                
                # Extract the local region
                local_region = blurred_image[y_start:y_end, x_start:x_end]
                local_estimate = estimate[y_start:y_end, x_start:x_end]
                
                # Pad for convolution
                padded_region = np.pad(local_region, pad, mode='symmetric')
                padded_estimate = np.pad(local_estimate, pad, mode='symmetric')
                
                # Forward model: re-blur the current estimate
                reblurred = convolve2d(padded_estimate, local_psf, mode='same', boundary='symm')
                reblurred = reblurred[pad:-pad, pad:-pad]
                
                # Calculate the ratio of observed to re-blurred (handle zeros)
                ratio = np.zeros_like(local_region)
                mask = reblurred > 1e-10
                ratio[mask] = local_region[mask] / reblurred[mask]
                
                # Pad the ratio for backward correlation
                padded_ratio = np.pad(ratio, pad, mode='symmetric')
                
                # Backward operation (correlation with PSF)
                error_term = convolve2d(padded_ratio, local_psf_flipped, mode='same', boundary='symm')
                error_term = error_term[pad:-pad, pad:-pad]
                
                # Calculate weight based on distance from center of this region
                y_grid, x_grid = np.meshgrid(
                    np.arange(y_end - y_start), 
                    np.arange(x_end - x_start), 
                    indexing='ij'
                )
                region_center_y = (y_end - y_start) / 2
                region_center_x = (x_end - x_start) / 2
                dist_from_center = np.sqrt((y_grid - region_center_y)**2 + (x_grid - region_center_x)**2)
                max_dist = np.sqrt(region_center_y**2 + region_center_x**2) + 1e-10
                weight = np.maximum(0, 1 - dist_from_center / max_dist)**2
                
                # Add to error correction term and weights
                relative_blur[y_start:y_end, x_start:x_end] += error_term * weight
                weights[y_start:y_end, x_start:x_end] += weight
        
        # Normalize by weights
        relative_blur /= weights
        
        # Add regularization term (Total Variation)
        if lambda_reg > 0:
            # Calculate gradients
            dx = np.zeros_like(estimate)
            dy = np.zeros_like(estimate)
            dx[:, 1:-1] = estimate[:, 2:] - estimate[:, :-2]
            dy[1:-1, :] = estimate[2:, :] - estimate[:-2, :]
            
            # Calculate the gradient magnitude
            grad_mag = np.sqrt(dx**2 + dy**2 + 1e-10)
            
            # Calculate the divergence of the normalized gradient
            dxx = np.zeros_like(estimate)
            dyy = np.zeros_like(estimate)
            dxx[:, 1:-1] = dx[:, 2:] / grad_mag[:, 2:] - dx[:, :-2] / grad_mag[:, :-2]
            dyy[1:-1, :] = dy[2:, :] / grad_mag[2:, :] - dy[:-2, :] / grad_mag[:-2, :]
            
            # Add regularization term
            reg_term = 1 - lambda_reg * (dxx + dyy)
        else:
            reg_term = 1
        
        # Update the estimate
        estimate *= relative_blur * reg_term
    
    return estimate

def improved_richardson_lucy(blurred_image, psf_generator, kernel_size=15, step=None,
                             iterations=20, lambda_reg=0.01, lambda_smooth=0.001,
                             overlap_factor=2, early_stopping=True):
    """
    Improved Richardson-Lucy deconvolution with better handling of point sources
    
    Parameters:
    - blurred_image: 2D array, blurred input image
    - psf_generator: function that generates PSF for a specific position
    - kernel_size: size of the PSF kernels
    - step: step size for sampling PSFs (smaller = more overlap)
    - iterations: maximum number of Richardson-Lucy iterations
    - lambda_reg: regularization parameter for Total Variation
    - lambda_smooth: smoothing parameter to prevent overconvergence
    - overlap_factor: controls patch overlap (higher = more overlap)
    - early_stopping: whether to use early stopping based on convergence
    
    Returns:
    - deconvolved image
    """
    import numpy as np
    from scipy.signal import convolve2d
    import numpy.fft as fft
    
    # Helper function for Gaussian window
    def gaussian_window(size_y, size_x, sigma=0.5):
        y = np.linspace(-1, 1, size_y)
        x = np.linspace(-1, 1, size_x)
        Y, X = np.meshgrid(y, x, indexing='ij')
        window = np.exp(-(X**2 + Y**2) / sigma)
        return window
    
    # Helper function for FFT-based convolution (faster than convolve2d)
    def fft_convolve2d(image, kernel):
        s1 = image.shape
        s2 = kernel.shape
        size = (s1[0] + s2[0] - 1, s1[1] + s2[1] - 1)
        
        # Pad inputs
        image_pad = np.pad(image, ((0, size[0] - s1[0]), (0, size[1] - s1[1])), mode='constant')
        kernel_pad = np.pad(kernel, ((0, size[0] - s2[0]), (0, size[1] - s2[1])), mode='constant')
        
        # FFT and multiply
        image_fft = fft.fft2(image_pad)
        kernel_fft = fft.fft2(kernel_pad)
        result = fft.ifft2(image_fft * kernel_fft)
        
        # Crop and return real part
        result = result[0:s1[0], 0:s1[1]].real
        return result
    
    if step is None:
        step = max(kernel_size // overlap_factor, 1)
        
    h, w = blurred_image.shape
    pad = kernel_size // 2
    
    # Apply mild smoothing to input image to reduce noise
    from scipy.ndimage import gaussian_filter
    blurred_smooth = gaussian_filter(blurred_image, sigma=0.5)
    
    # Create a grid of PSF sampling points with more overlap
    y_positions = np.arange(0, h, step)
    x_positions = np.arange(0, w, step)
    
    # Initialize estimate with a smoothed version of the input
    estimate = blurred_smooth.copy()
    
    # To track convergence
    prev_estimate = estimate.copy()
    
    # For each iteration
    for iter in range(iterations):
        # Construct total error correction term
        relative_blur = np.zeros_like(blurred_image)
        weights = np.zeros_like(blurred_image) + 1e-10
        
        # For each position
        for y_idx, cy in enumerate(y_positions):
            for x_idx, cx in enumerate(x_positions):
                # Define the region with more overlap
                y_start = max(0, cy - pad)
                y_end = min(h, cy + pad + step)
                x_start = max(0, cx - pad)
                x_end = min(w, cx + pad + step)
                
                # Skip tiny regions
                if (y_end - y_start) < 3 or (x_end - x_start) < 3:
                    continue
                
                # Generate the local PSF for this position
                local_psf = psf_generator((kernel_size, kernel_size), cy + pad, cx + pad)
                
                # Ensure PSF is normalized
                if np.sum(local_psf) > 0:
                    local_psf = local_psf / np.sum(local_psf)
                
                # Make sure PSF is flipped for correlation
                local_psf_flipped = np.flip(np.flip(local_psf, 0), 1)
                
                # Extract the local region
                local_region = blurred_image[y_start:y_end, x_start:x_end]
                local_estimate = estimate[y_start:y_end, x_start:x_end]
                
                # Pad for convolution
                pad_y = min(kernel_size, (y_end - y_start) // 2 + 1)
                pad_x = min(kernel_size, (x_end - x_start) // 2 + 1)
                padded_region = np.pad(local_region, ((pad_y, pad_y), (pad_x, pad_x)), mode='symmetric')
                padded_estimate = np.pad(local_estimate, ((pad_y, pad_y), (pad_x, pad_x)), mode='symmetric')
                
                # Forward model: re-blur the current estimate
                reblurred = convolve2d(padded_estimate, local_psf, mode='same', boundary='symm')
                reblurred = reblurred[pad_y:-pad_y, pad_x:-pad_x]
                
                # Calculate the ratio of observed to re-blurred (handle zeros)
                ratio = np.ones_like(local_region)
                mask = reblurred > 1e-10
                ratio[mask] = local_region[mask] / reblurred[mask]
                
                # Pad the ratio for backward correlation
                padded_ratio = np.pad(ratio, ((pad_y, pad_y), (pad_x, pad_x)), mode='symmetric')
                
                # Backward operation (correlation with PSF)
                error_term = convolve2d(padded_ratio, local_psf_flipped, mode='same', boundary='symm')
                error_term = error_term[pad_y:-pad_y, pad_x:-pad_x]
                
                # Use a Gaussian window for better blending
                window = gaussian_window(y_end - y_start, x_end - x_start)
                
                # Add to error correction term and weights
                relative_blur[y_start:y_end, x_start:x_end] += error_term * window
                weights[y_start:y_end, x_start:x_end] += window
        
        # Normalize by weights
        relative_blur /= weights
        
        # Add regularization term (Total Variation)
        if lambda_reg > 0:
            # Calculate gradients with better boundary handling
            dx = np.zeros_like(estimate)
            dy = np.zeros_like(estimate)
            dx[:, :-1] = estimate[:, 1:] - estimate[:, :-1]
            dy[:-1, :] = estimate[1:, :] - estimate[:-1, :]
            
            # Calculate the gradient magnitude with stability term
            grad_mag = np.sqrt(dx**2 + dy**2 + 1e-6)
            
            # Calculate the divergence of the normalized gradient
            dxx = np.zeros_like(estimate)
            dyy = np.zeros_like(estimate)
            
            # Normalize gradients
            norm_dx = dx / (grad_mag + 1e-6)
            norm_dy = dy / (grad_mag + 1e-6)
            
            # Calculate divergence
            dxx[:, 1:] = norm_dx[:, 1:] - norm_dx[:, :-1]
            dyy[1:, :] = norm_dy[1:, :] - norm_dy[:-1, :]
            
            # Add regularization term
            reg_term = 1 - lambda_reg * (dxx + dyy)
        else:
            reg_term = 1
        
        # Update the estimate with a smoothness constraint to prevent overconvergence
        update = relative_blur * reg_term
        
        # Apply smoothness constraint to prevent overconvergence
        if lambda_smooth > 0:
            update_smooth = gaussian_filter(update, sigma=lambda_smooth)
            update = (1 - lambda_smooth) * update + lambda_smooth * update_smooth
        
        # Update the estimate
        estimate *= update
        
        # Check for convergence
        if early_stopping and iter > 0:
            change = np.sum(np.abs(estimate - prev_estimate)) / np.sum(np.abs(estimate))
            if change < 0.001:  # Convergence criterion
                print(f"Early stopping at iteration {iter+1}")
                break
        
        prev_estimate = estimate.copy()
    
    return estimate


######################### MAIN #################################
#to call deconvolution with richardson lucy methods

def deconvolution_richardson_methods(blurred_image, psf_generator, original_image=None, kernel_size=15, step=20, iterations =20):
    """
    Demonstrate different deconvolution methods
    
    Parameters:
    - blurred_image: 2D array, blurred input image
    - psf_generator: function that generates PSF for a specific position
    - original_image: 2D array, original unblurred image (if available, for comparison)
    
    Returns:
    - dictionary of results from different methods
    """
    results = {}

    # 1. Wiener Deconvolution
    print("Applying RL Deconvolution...")
    results['richardson_lucy'] = regularized_richardson_lucy(
        blurred_image, psf_generator, kernel_size=kernel_size, step=step, iterations=iterations)
    if original_image is not None:
        print('PSNR =', compute_psnr(original_image, results['richardson_lucy']))
        print('SSIM =', compute_ssim(original_image, results['richardson_lucy']))
    # 2. Improved Wiener Deconvolution
    print("Applying improved RL Deconvolution...")
    results['improved_richardson_lucy'] = improved_richardson_lucy(
        blurred_image, psf_generator, kernel_size=kernel_size, step=step, iterations=iterations)
    if original_image is not None:
        print('PSNR =', compute_psnr(original_image, results['improved_richardson_lucy']))
        print('SSIM =', compute_ssim(original_image, results['improved_richardson_lucy']))

    # Plot results
    plt.figure(figsize=(10, 10))
    
    # Original and blurred images
    plt.subplot(2, 2, 1)
    if original_image is not None:
        plt.imshow(original_image, cmap='grey')
        plt.title('Original Image')
    else:
        plt.imshow(blurred_image, cmap='grey')
        plt.title('Blurred Image')
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.imshow(blurred_image, cmap='grey')
    plt.title('Blurred Image')
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.imshow(results['richardson_lucy'], cmap='grey')
    plt.title('Richardson-Lucy Deconvolution')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(results['improved_richardson_lucy'], cmap='grey')
    plt.title('Improved Richardson-Lucy')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('./deconvolution_method/plots/deblurred_image_richardson_lucy.png')
    
    return results