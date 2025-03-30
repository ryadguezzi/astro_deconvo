import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unsupervised_wiener, wiener
from pre_processing import renormalize_to_minus_one_to_one
from performance_evaluation import compute_psnr, compute_ssim

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

def upgraded_wiener(blurred_image, psf):
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

def sectional_wiener_deconvolution(blurred_image, psf_generator, kernel_size=15, step=None, iterations=1):
    """
    Apply sectional Wiener deconvolution to a spatially variant blurred image
    
    Parameters:
    - blurred_image: 2D array, blurred input image
    - psf_generator: function that generates PSF for a specific position
    - kernel_size: size of the PSF kernels (odd number)
    - step: step size for sampling PSFs (if None, use kernel_size//2)
    - noise_level: estimate of the noise level for the Wiener filter
    - iterations: number of refinement iterations
    
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
    
    # Initial estimate is the blurred image
    estimate = blurred_image.copy()
    
    # Refinement iterations
    for iteration in range(iterations):
        # Create a residual image (starts as zeros)
        residual = np.zeros_like(blurred_image)
        weights = np.zeros_like(blurred_image) + 1e-10  # Small value to avoid division by zero
        
        # For each sampling position
        for y_idx, cy in enumerate(y_positions):
            for x_idx, cx in enumerate(x_positions):
                # Generate the local PSF for this position
                local_psf = psf_generator((kernel_size, kernel_size), cy + pad, cx + pad)
                
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
                local_estimate = estimate[y_start:y_end, x_start:x_end]
                
                # Pad for convolution
                padded_estimate = np.pad(local_estimate, pad, mode='symmetric')
                
                # Apply Wiener deconvolution to the local region
                local_deconv = unsupervised_wiener(renormalize_to_minus_one_to_one(padded_estimate), local_psf)[0]
                
                # Remove padding
                local_deconv = local_deconv[pad:-pad, pad:-pad]
                
                # Calculate weight based on distance from center of this region
                # (higher weight near the center of the region)
                y_grid, x_grid = np.meshgrid(
                    np.arange(y_end - y_start), 
                    np.arange(x_end - x_start), 
                    indexing='ij'
                )
                region_center_y = (y_end - y_start) / 2
                region_center_x = (x_end - x_start) / 2
                dist_from_center = np.sqrt((y_grid - region_center_y)**2 + (x_grid - region_center_x)**2)
                max_dist = np.sqrt(region_center_y**2 + region_center_x**2)
                weight = np.maximum(0, 1 - dist_from_center / max_dist)**2
                
                # Add to residual and weights
                residual[y_start:y_end, x_start:x_end] += local_deconv * weight
                weights[y_start:y_end, x_start:x_end] += weight
        
        # Update estimate using weighted average
        estimate = residual / weights
    
    return estimate

def improved_sectional_wiener_deconvolution(blurred_image, psf_generator, kernel_size=15, step=None, 
                                           iterations=2, noise_level=0.01, overlap_factor=2):
    """
    Apply improved sectional Wiener deconvolution to a spatially variant blurred image
    with better handling of artifacts
    
    Parameters:
    - blurred_image: 2D array, blurred input image
    - psf_generator: function that generates PSF for a specific position
    - kernel_size: size of the PSF kernels (odd number)
    - step: step size for sampling PSFs (if None, use kernel_size//overlap_factor)
    - noise_level: estimate of the noise level for the Wiener filter (0.01-0.1 typically works well)
    - iterations: number of refinement iterations
    - overlap_factor: controls patch overlap (higher = more overlap)
    
    Returns:
    - deconvolved image
    """
    import numpy as np
    from skimage.restoration import wiener
    
    # Define a function for Wiener filter with better noise handling
    def robust_wiener(image, psf, noise_level=noise_level):
        # Normalize PSF
        psf = psf / np.sum(psf)
        
        # Apply Wiener filter
        print(image.shape, psf.shape, noise_level)
        deconvolved = wiener(image, psf, noise_level)
        
        # Apply mild bilateral filter to reduce ringing
        try:
            from skimage.restoration import denoise_bilateral
            deconvolved = denoise_bilateral(deconvolved, sigma_color=0.05, sigma_spatial=1.0)
        except:
            # Fall back if bilateral filter is not available
            pass
            
        return deconvolved
    
    # Calculate step size for overlapping patches
    if step is None:
        step = max(kernel_size // overlap_factor, 1)
    
    h, w = blurred_image.shape
    pad = kernel_size // 2
    
    # Use a Gaussian window for better blending between patches
    def gaussian_window(size):
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        window = np.exp(-(X**2 + Y**2) / 0.5)
        return window
    
    # Create a grid of PSF sampling points with more overlap
    y_positions = np.arange(0, h, step)
    x_positions = np.arange(0, w, step)
    
    # Initial estimate is the blurred image
    estimate = blurred_image.copy()
    
    # Refinement iterations
    for iteration in range(iterations):
        # Create a residual image and weights
        residual = np.zeros_like(blurred_image)
        weights = np.zeros_like(blurred_image) + 1e-10  # Small value to avoid division by zero
        
        # For each sampling position
        for y_idx, cy in enumerate(y_positions):
            for x_idx, cx in enumerate(x_positions):
                # Generate the local PSF for this position
                local_psf = psf_generator((kernel_size, kernel_size), cy + pad, cx + pad)
                
                # Ensure PSF is properly normalized
                if np.sum(local_psf) > 0:
                    local_psf = local_psf / np.sum(local_psf)
                
                # Define the region for this PSF with overlap
                y_start = max(0, cy - pad)
                y_end = min(h, cy + pad + step)
                x_start = max(0, cx - pad)
                x_end = min(w, cx + pad + step)
                
                # Skip very small regions
                if (y_end - y_start) < 3 or (x_end - x_start) < 3:
                    continue
                
                # Extract the local region
                local_estimate = estimate[y_start:y_end, x_start:x_end]
                
                # Apply robust Wiener deconvolution
                local_deconv = robust_wiener(local_estimate, local_psf)
                
                # Create position-dependent Gaussian window for smooth blending
                window = gaussian_window(max(y_end - y_start, x_end - x_start))
                window = window[:y_end - y_start, :x_end - x_start]
                
                # Add to residual and weights using Gaussian window
                residual[y_start:y_end, x_start:x_end] += local_deconv * window
                weights[y_start:y_end, x_start:x_end] += window
        
        # Update estimate using weighted average
        estimate = residual / weights
        
        # Apply mild total variation regularization to reduce artifacts
        try:
            from skimage.restoration import denoise_tv_chambolle
            estimate = denoise_tv_chambolle(estimate, weight=0.01)
        except:
            pass
    
    return estimate

def improved_sectional_wiener_deconvolution(
    blurred_image, 
    psf_generator, 
    kernel_size=15, 
    step=None, 
    noise_level=0.01, 
    overlap_factor=2,
):
    """
    Enhanced sectional Wiener deconvolution with more robust parameters and diagnostics
    
    Additional features:
    - Default PSF generator
    - More flexible noise handling
    - Optional verbose mode for diagnostics
    """
    h, w = blurred_image.shape
    pad = kernel_size // 2
    
    # Adaptive noise level estimation
    noise_level = np.std(blurred_image) * 0.1  # Dynamic noise estimation
    
    def robust_wiener(image, psf, noise_level=noise_level):
        # Ensure PSF is the right size for the current patch
        if image.shape[0] < psf.shape[0] or image.shape[1] < psf.shape[1]:
            # Resize PSF to fit the current image patch
            resized_psf = np.zeros((min(image.shape[0], psf.shape[0]), min(image.shape[1], psf.shape[1])))
            resized_psf = psf[:resized_psf.shape[0], :resized_psf.shape[1]]
            # Renormalize the PSF
            resized_psf = resized_psf / np.sum(resized_psf)
            psf = resized_psf
        else:
            psf = psf / np.sum(psf)
        
        deconvolved = wiener(image, psf, noise_level)
        try:
            deconvolved = denoise_bilateral(deconvolved, sigma_color=0.1, sigma_spatial=1.5)
        except:
            pass
        
        return deconvolved

    # Gaussian window for smoother blending
    def gaussian_window(size):
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        return np.exp(-(X**2 + Y**2) / 0.5)
    
    # Compute step size
    if step is None:
        step = max(kernel_size // overlap_factor, 1)
    
    # Position sampling
    y_positions = np.arange(0, h, step)
    x_positions = np.arange(0, w, step)
    
    # Initial estimate
    estimate = blurred_image.copy()
    

    residual = np.zeros_like(blurred_image)
    weights = np.zeros_like(blurred_image) + 1e-10
    
    for y_idx, cy in enumerate(y_positions):
        for x_idx, cx in enumerate(x_positions):
            # Generate local PSF
            local_psf = psf_generator((kernel_size, kernel_size), cy + pad, cx + pad)
            
            # Region of interest
            y_start = max(0, cy - pad)
            y_end = min(h, cy + pad + step)
            x_start = max(0, cx - pad)
            x_end = min(w, cx + pad + step)
            
            if (y_end - y_start) < 3 or (x_end - x_start) < 3:
                continue
            
            local_estimate = estimate[y_start:y_end, x_start:x_end]
            local_deconv = robust_wiener(local_estimate, local_psf)
            
            # Gaussian blending
            window = gaussian_window(max(y_end - y_start, x_end - x_start))
            window = window[:y_end - y_start, :x_end - x_start]
            
            residual[y_start:y_end, x_start:x_end] += local_deconv * window
            weights[y_start:y_end, x_start:x_end] += window
    
    # Update estimate
    estimate = residual / weights
    
    # Total variation denoising
    try:
        estimate = denoise_tv_chambolle(estimate, weight=0.02)
    except :
        pass

    
    return estimate



######################### MAIN #################################
#to call deconvolution with wiener methods

def deconvolution_wiener_methods(blurred_image, psf_generator, original_image=None, kernel_size=15, step=20):
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
    print("Applying classical Wiener Deconvolution...")
    results['classical_wiener'] = wiener_deconvolution(blurred_image, psf_generator(blurred_image.shape, blurred_image.size//2, blurred_image.size//2))
    if original_image is not None:
        print('PSNR =', compute_psnr(original_image, results['classical_wiener']))
        print('SSIM =', compute_ssim(original_image, results['classical_wiener']))
    
    # 2. Improved Wiener Deconvolution
    print("Applying upgreaded Wiener Deconvolution...")
    results['upgraded_wiener'] = upgraded_wiener(
        blurred_image, psf_generator(blurred_image.shape, blurred_image.size//2, blurred_image.size//2)
    )[0]
    if original_image is not None:
        print('PSNR =', compute_psnr(original_image, results['upgraded_wiener']))
        print('SSIM =', compute_ssim(original_image, results['upgraded_wiener']))
    
    # 3. Sectional Wiener Deconvolution
    print("Applying sectional Wiener Deconvolution...")
    results['sectional_wiener'] = sectional_wiener_deconvolution(
        blurred_image, psf_generator, kernel_size=kernel_size, step=step
    )
    if original_image is not None:
        print('PSNR =', compute_psnr(original_image, results['sectional_wiener']))
        print('SSIM =', compute_ssim(original_image, results['sectional_wiener']))

    # 4. Improved sectional Wiener Deconvolution
    print("Applying improved sectional Wiener Deconvolution...")
    results['improved_sectional_wiener'] = improved_sectional_wiener_deconvolution(
        blurred_image, psf_generator, kernel_size=kernel_size, step=step
    )
    if original_image is not None:
        print('PSNR =', compute_psnr(original_image, results['sectional_wiener']))
        print('SSIM =', compute_ssim(original_image, results['sectional_wiener']))
    

    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Original and blurred images
    plt.subplot(2, 3, 1)
    if original_image is not None:
        plt.imshow(original_image, cmap='grey')
        plt.title('Original Image')
    else:
        plt.imshow(blurred_image, cmap='grey')
        plt.title('Blurred Image')
    plt.colorbar()
    
    plt.subplot(2, 3, 2)
    plt.imshow(blurred_image, cmap='grey')
    plt.title('Blurred Image')
    plt.colorbar()
    
    plt.subplot(2, 3, 3)
    plt.imshow(results['classical_wiener'], cmap='grey')
    plt.title('Wiener Deconvolution')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.imshow(results['upgraded_wiener'], cmap='grey')
    plt.title('Improved Wiener')
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(results['sectional_wiener'], cmap='grey')
    plt.title('Sectional Wiener')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.imshow(results['improved_sectional_wiener'], cmap='grey')
    plt.title('Improved sectional Wiener')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('./deconvolution_method/plots/deblurred_image_wiener.png')
    
    return results