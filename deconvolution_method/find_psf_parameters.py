
"""File to find alpha and beta parameters of the Moffat PSF for any image."""
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from scipy.optimize import minimize, differential_evolution
from astropy.stats import sigma_clipped_stats
from skimage.feature import peak_local_max
from skimage.metrics import structural_similarity as ssim

def moffat_psf(image_shape, alpha=3, beta=2.5, normalize=True):
    """
    Compute Moffat PSF.
    
    Parameters:
    - image_shape: Tuple of image dimensions
    - alpha: Core width parameter
    - beta: Power-law index
    - normalize: Whether to normalize the PSF
    
    Returns:
    - 2D Moffat PSF
    """
    h, w = image_shape
    y, x = np.meshgrid(np.linspace(-h//2, h//2, h), 
                       np.linspace(-w//2, w//2, w), 
                       indexing='ij')
    r2 = x**2 + y**2
    
    # Improved numerical stability
    psf = (1 + r2 / (alpha**2)) ** -beta
    
    if normalize:
        psf /= np.sum(psf)
    
    return psf

def advanced_star_detection(image, sigma_threshold=3, fwhm=3):
    """
    Advanced star detection with multiple robustness features
    
    Parameters:
    - image: Input image
    - sigma_threshold: Detection threshold in sigma units
    - fwhm: Approximate Full Width at Half Maximum
    
    Returns:
    - Detected star sources
    """
    # Robust background estimation
    mean, _,std = sigma_clipped_stats(image)
    
    # Detect local peaks using difference of Gaussian
    dog = ndimage.gaussian_filter(image, sigma=fwhm) - \
          ndimage.gaussian_filter(image, sigma=fwhm*1.5)
    
    # Find local maxima above threshold
    threshold = mean + (sigma_threshold * std)
    peaks = peak_local_max(
        dog, 
        min_distance=int(fwhm), 
        threshold_abs=threshold,
        exclude_border=True
    )
    
    # Compute centroid and additional star characteristics
    star_sources = []
    for peak in peaks:
        y, x = peak
        # Extract local region
        region = image[max(0, y-fwhm):min(image.shape[0], y+fwhm+1),
                      max(0, x-fwhm):min(image.shape[1], x+fwhm+1)]
        
        # Compute centroid with weighted center of mass
        total = np.sum(region)
        y_coords, x_coords = np.indices(region.shape)
        weighted_y = np.sum(region * y_coords) / total + (y - fwhm)
        weighted_x = np.sum(region * x_coords) / total + (x - fwhm)
        
        star_sources.append({
            'xcentroid': weighted_x,
            'ycentroid': weighted_y,
            'peak_value': np.max(region)
        })
    
    print("Recreating star field for visualization")
    star_field = np.zeros(image.shape, dtype=np.float32)
    for star in star_sources:
        y = int(star['ycentroid'])  # Convert to integer to index the array
        x = int(star['xcentroid'])  # Convert to integer to index the array
        star_field[y, x] = 255
    plt.figure(figsize=(6, 6))
    plt.title('Detected Stars')
    plt.imshow(star_field, cmap='grey')
    plt.savefig('./deconvolution_method/plots/optimization/detected_stars.png')
    
    return star_sources

def multi_objective_psf_cost(params, averaged_psf, methods=['structural_similarity']):
    """
    Multi-objective cost function for PSF parameter optimization
    
    Parameters:
    - params: [alpha, beta] to optimize
    - averaged_psf: Target PSF to match
    - methods: List of comparison methods
    
    Returns:
    - Composite cost
    """
    if np.any(np.isnan(params)):
        print("Warning: NaN values detected in parameters:", params)
        return 1e10  # Return a very high cost to penalize NaN values
    alpha, beta = params
    model_psf = moffat_psf(averaged_psf.shape, alpha, beta)
    
    costs = []
    
    # L2 Norm (Least Squares)
    if 'l2' in methods:
        l2_cost = np.mean((averaged_psf - model_psf)**2)
        costs.append(l2_cost)
    
    # Structural Similarity Index
    if 'structural_similarity' in methods:
        ssim_cost = 1 - ssim(averaged_psf, model_psf, data_range=1.0)
        costs.append(ssim_cost)
    
    # Weighted combination
    return np.mean(costs)

def advanced_moffat_parameter_optimization(averaged_psf, optimization_method='hybrid', verbose = False):
    """
    Advanced parameter optimization with multiple strategies
    
    Parameters:
    - averaged_psf: Target PSF to match
    - optimization_method: Optimization approach
    
    Returns:
    - Optimized parameters, optimization details
    """
    # Define parameter bounds
    bounds = [(0.1, 10), (0.1, 10)]  # alpha, beta
    
    if optimization_method == 'differential_evolution':
        # Global search with differential evolution
        result = differential_evolution(
            multi_objective_psf_cost, 
            bounds=bounds, 
            args=(averaged_psf,),
            popsize=30,
            maxiter=100,
            tol=1e-7
        )
    
    else:  # Hybrid approach
        # Local search with global initialization
        global_result = differential_evolution(
            multi_objective_psf_cost, 
            bounds=bounds, 
            args=(averaged_psf,),
            popsize=20,
            maxiter=50
        )
        
        # Refine with local search
        result = minimize(
            multi_objective_psf_cost, 
            global_result.x, 
            bounds=bounds, 
            method='L-BFGS-B',
            args=(averaged_psf,)
        )
    
    if verbose :
        print("Plotting optimization results")
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.title('Original Averaged PSF')
        plt.imshow(averaged_psf, cmap='hot')
        plt.colorbar()
        
        plt.subplot(132)
        optimized_psf = moffat_psf(averaged_psf.shape, *result.x)
        plt.title(f'Optimized Moffat PSF\nα={result.x[0]:.2f}, β={result.x[1]:.2f}')
        plt.imshow(optimized_psf, cmap='hot')
        plt.colorbar()
        
        plt.subplot(133)
        plt.title('Difference')
        plt.imshow(np.abs(averaged_psf - optimized_psf), cmap='viridis')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('./deconvolution_method/plots/optimization/comparing_optimized_psf.png')
    
    return result.x, result

def comprehensive_psf_estimation_per_patch(image):
    """
    PSF estimation pipeline that finds optimal parameters for each patch
    using multiple optimization methods and stores results from each method
    
    Parameters:
    - image (2d array): Input image
    
    Returns:
    - List of dictionaries containing patch locations and their optimized PSF parameters for each method
    - Original patches for reference
    """
    # Detect stars with advanced method
    sources = advanced_star_detection(image)
    
    # Extract PSF patches and optimize each one individually
    patch_results = []
    patches = []
    print('Starting optimization for each patch')
    # Define optimization methods
    optimization_methods = ['differential_evolution', 'hybrid']
    
    for star in sources:
        x, y = int(star['xcentroid']), int(star['ycentroid'])
        crop_size = 15
        hsize = crop_size // 2
        
        # Check if patch is fully within image bounds
        if (hsize < x < image.shape[1]-hsize and 
            hsize < y < image.shape[0]-hsize):
            
            # Extract patch
            patch = image[y-hsize:y+hsize+1, x-hsize:x+hsize+1]
            
            # Normalize patch
            patch_normalized = patch / np.sum(patch)
            
            # Store original patch for reference
            patches.append({
                'x': x, 
                'y': y, 
                'patch': patch_normalized
            })
            
            # Store results for each optimization method
            method_results = {}
            
            for method in optimization_methods:
                params, result = advanced_moffat_parameter_optimization(
                    patch_normalized,
                    optimization_method=method
                )
                
                method_results[method] = {'params': params, 'cost': result.fun}

            
            # Store all results along with patch location
            patch_results.append({
                'x': x,
                'y': y,
                'method_results': method_results})
    return patch_results, patches

######################### PLOTS #################################

def plot_landscape(method):
    """
    Plot the cost function landscape for visualization
    Parameters:
    - cost_function: Function to evaluate
    - bounds: Bounds for the parameters
    - num_points: Number of points in each dimension
    """
    alpha_vals = np.linspace(0.1, 10, 50) 
    beta_vals = np.linspace(0.1, 10, 50)

    A, B = np.meshgrid(alpha_vals, beta_vals)
    Z = np.zeros_like(A)

    averaged_psf = moffat_psf((50, 50), alpha=1, beta=7) # Example PSF

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            params = [A[i, j], B[i, j]]
            Z[i, j] = multi_objective_psf_cost(params, averaged_psf, methods=[method])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, Z, cmap='viridis', edgecolor='none')

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_zlabel("Coût")
    ax.set_title("3D landscape of cost function")
    plt.savefig(f'./deconvolution_method/plots/optimization/cost_function_landscape{method}.png')


def plot_parameter_vs_radial_distance_per_method(patch_results, image_shape, param_names=None):
    """
    Visualizes PSF parameters as a function of radial distance for each method.

    Parameters:
    - patch_results: List of dictionaries containing positions and optimized parameters.
    - image_shape: Tuple (height, width) of the image.
    - param_names: List of parameter names (optional).
    """
    # Computing the center of the image
    center_y, center_x = image_shape[0] / 2, image_shape[1] / 2
    
    # Extracting x and y coordinates from patch results
    x_coords = np.array([result['x'] for result in patch_results])
    y_coords = np.array([result['y'] for result in patch_results])
    
    # Computing the squared radial distance from the center
    radial_distance_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
    
    # Getting the list of methods from the first result
    methods = list(patch_results[0]['method_results'].keys())
    num_params = len(patch_results[0]['method_results'][methods[0]]['params'])
    
    # If param_names is not provided, generate default names
    if param_names is None:
        param_names = [f'Parameter{i}' for i in range(num_params)]
    
    trend_coefficients = {}
    
    # For each method, create a plot for each parameter
    for method in methods:
        trend_coefficients[method] = {}
        
        fig, axes = plt.subplots(nrows=num_params, figsize=(10, 4*num_params))
        fig.suptitle(f'PSF parameters vs radial distance - method: {method}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        
        for param_index in range(num_params):
            param_values = np.array([
                result['method_results'][method]['params'][param_index] 
                for result in patch_results
            ])
            
            axes[param_index].scatter(
                np.sqrt(radial_distance_sq),
                param_values, 
                alpha=0.7,
                label='Measured values'
            )
            
            # Adding a polynomial trend line
            z = np.polyfit(np.sqrt(radial_distance_sq), param_values, 1)
            p = np.poly1d(z)
            x_range = np.linspace(np.min(np.sqrt(radial_distance_sq)), np.max(np.sqrt(radial_distance_sq)), 100)
            axes[param_index].plot(
                x_range, 
                p(x_range), 
                'r--',
                label={f'Tendancy: {z[0]:.2e}x + {z[1]:.2e}'}
            )

            trend_coefficients[method][param_names[param_index]] = {
                'slope': z[0],
                'intercept': z[1]
            }

            axes[param_index].set_xlabel('radial distance to the center (pixels)')
            axes[param_index].set_ylabel(param_names[param_index])
            axes[param_index].set_title(f'{param_names[param_index]} vs radial distance')
            axes[param_index].legend()
            axes[param_index].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'./deconvolution_method/plots/optimization/parameters_vs_radial_distance_{method}.png')
        plt.close(fig)
    
    # Create comparative plots
    for param_index in range(num_params):
        plt.figure(figsize=(12, 8))
        
        for method in methods:
            param_values = np.array([
                result['method_results'][method]['params'][param_index] 
                for result in patch_results
            ])
            
            plt.scatter(
                np.sqrt(radial_distance_sq), 
                param_values, 
                alpha=0.6,
                label=f'Method: {method}'
            )
            
            z = np.polyfit(np.sqrt(radial_distance_sq), param_values, 1)
            p = np.poly1d(z)
            x_range = np.linspace(np.min(np.sqrt(radial_distance_sq)), np.max(np.sqrt(radial_distance_sq)), 100)
            plt.plot(x_range, p(x_range), '--', label=f'Tendancy {method}')

        x_range = np.linspace(np.min(np.sqrt(radial_distance_sq)), np.max(np.sqrt(radial_distance_sq)), 100)
        if param_index == 0:
            alpha_0 = 1
            k_alpha = 0.1
            alpha_max = 6
            true_values = alpha_0 + np.sqrt(radial_distance_sq)
            true_values = np.minimum(alpha_0 + k_alpha * x_range, alpha_max)
        elif param_index == 1:
            beta_0 = 7
            k_beta = 0.02
            beta_min = 2.5
            true_values = beta_0 - k_beta * np.sqrt(radial_distance_sq)
            true_values = np.maximum(beta_0 - k_beta * x_range, beta_min)

        plt.plot(
            x_range, 
            true_values, 
            'k-', 
            linewidth=2, 
            label='True values'
        )
        
        plt.xlabel('Radial distance to the center (pixels)')
        plt.ylabel(param_names[param_index])
        plt.title(f'Comparing {param_names[param_index]} vs radial distance for each method')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f'./deconvolution_method/plots/optimization/parameter{param_index}_comparison.png')
        plt.close()
    return trend_coefficients

######################### MAIN #################################
#to call function to find parameters of Moffat function

def find_parameters(image, verbose=False):
    """
    Find the Moffat PSF parameters for a given image.
    
    Input:
    - image (2Darray): Input blurred image to analyze
    
    Returns:
    - Coefficient (dict) of the differential evolution method
    """
    if verbose :
        print("Plotting cost function landscape")
        plot_landscape('structural_similarity')
    print(f"Estimated Moffat PSF Parameters for SSIM cost:")
    results_per_patch, _= comprehensive_psf_estimation_per_patch(image)
    coefficient = plot_parameter_vs_radial_distance_per_method(results_per_patch, image.shape)
    print("Keepig only results from differential evolution method")
    return coefficient['differential_evolution']