import numpy as np
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder
from scipy.optimize import minimize

def moffat_psf(image_shape, alpha=3, beta=2.5, normalize=True):
    """
    Compute Moffat PSF.
    """
    h, w = image_shape
    y, x = np.meshgrid(np.linspace(-h//2, h//2, h), np.linspace(-w//2, w//2, w), indexing='ij')
    r2 = x**2 + y**2
    psf = (1 + r2 / alpha**2) ** -beta
    if normalize:
        psf /= np.sum(psf)
    return psf

def detect_stars(image, threshold=0.99):
    """Detect star in an image using DAOStarFinder.
    
    Inputs :
    image : 2D array, the image.
    threshold : float, the threshold for detection.
    
    Returns :
    sources : Table, the detected stars
    """
    
    # Subtract the mean of the image
    mean = np.mean(image)
    image_subtracted = image - mean
    # Setup the DAOStarFinder with appropriate threshold and FWHM
    daofind = DAOStarFinder(threshold=threshold, fwhm=2) #3 because the image has been created with 3pixels FWHM
    sources = daofind.find_stars(image_subtracted)
    return sources

def recreate_star_image(target_size,sources, patch_size=3, brightness=255):
    """Recreate an image with stars at the detected positions. Use to check the star detection."""

    # Create an empty black image of the same size as the original image
    star_image = np.zeros(target_size)
    
    # For each detected star, create a bright point at its position
    for star in sources:
        x, y = int(star['xcentroid']), int(star['ycentroid'])
        
        # You can adjust the size of the points (using a small patch) and set brightness
        half_size = patch_size // 2
        if (x > half_size and y > half_size and 
            x < target_size[1] - half_size and 
            y < target_size[0] - half_size):
            star_image[y-half_size:y+half_size+1, x-half_size:x+half_size+1] = brightness
    
    return star_image

def plot_cost_landscape(cost, alpha_range=(0.01, 10), beta_range=(0.01, 10), grid_size=50):
    """Plot the cost function landscape for the given cost function."""
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], grid_size)
    beta_values = np.linspace(beta_range[0], beta_range[1], grid_size)
    
    # Create a meshgrid of alpha and beta values
    alpha_grid, beta_grid = np.meshgrid(alpha_values, beta_values)
    
    # Compute the cost function for each (alpha, beta) pair
    cost_values = np.zeros_like(alpha_grid)
    for i in range(grid_size):
        for j in range(grid_size):
            cost_values[i, j] = cost([alpha_grid[i, j], beta_grid[i, j]])
    
    # Plot the landscape
    print('Saving cost landscape plot')
    plt.figure(figsize=(10, 8))
    cp = plt.contourf(alpha_grid, beta_grid, cost_values, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Cost Function Landscape')
    plt.savefig('deconvolution_method/plots/optimization/cost_landscape.png')

def fit_moffat_from_image(blurred_image, sources, crop_size=15):
    """
    Extract individual PSFs and fit Moffat parameters for the average PSF.
    """
    star_positions = []
    
    # Extraction des régions d'intérêt (ROIs)
    psf_patches = []
    hsize = crop_size // 2
    for star in sources:
        x_star, y_star = int(star['xcentroid']), int(star['ycentroid'])
        if hsize < x_star < blurred_image.shape[1]-hsize and hsize < y_star < blurred_image.shape[0]-hsize:
            patch = blurred_image[y_star-hsize:y_star+hsize+1, x_star-hsize:x_star+hsize+1]
            patch[patch < np.max(patch)/2]=0
            patch_normalized = patch /np.sum(patch)
            psf_patches.append(patch_normalized)
        star_positions.append((x_star, y_star))
    
    # Moyenne des PSFs
    averaged_psf = np.mean(psf_patches, axis=0)
    averaged_psf[averaged_psf < np.max(averaged_psf)/4] = 0
    print('Saving averaged PSF')
    plt.figure()
    plt.imshow(averaged_psf, cmap='hot')
    plt.colorbar(label='Pixel Intensity')
    plt.axis('off')
    plt.savefig('deconvolution_method/plots/optimization/averaged_psf.png')

    #example_psf = moffat_psf(averaged_psf.shape, 2.5, 3)
    #print('Example PSF')
    #plt.figure()
    #plt.imshow(example_psf, cmap='hot')
    #plt.colorbar(label='Pixel Intensity')
    #plt.axis('off')
    #plt.savefig('deconvolution_method/example_psf.png')

    # Fonction de coût pour l'ajustement
    def cost(params):
        alpha, beta = params
        model_psf = moffat_psf(averaged_psf.shape, alpha, beta)
        weights = 1 / (averaged_psf + 1e-6)  # Avoid division by zero
        return np.sum(weights * (averaged_psf - model_psf)**2)  
    
    # Visualize the cost landscape before minimizing
    plot_cost_landscape(cost)
    
    # Ajustement non-linéaire
    initial_guess = [2.0, 2]  # Valeurs initiales pour alpha et beta
    bounds = [(0.1, 10), (0.1, 10)]  # Bornes pour alpha et beta
    result = minimize(cost, initial_guess, bounds=bounds, method='L-BFGS-B') 
    
    return result.x, averaged_psf, star_positions

def find_moffat_parameters(image):
    """Run the full pipeline to find the Moffat parameter from the image."""
    image_copy = np.copy(image)

    sources = detect_stars(image_copy, threshold=0.2)
    star_image  = recreate_star_image(image.shape,sources, patch_size=2, brightness=255)
    print('Saving recreated image')
    plt.figure(figsize=(10, 10))
    plt.imshow(star_image, cmap='gray')
    plt.axis('off')
    plt.savefig('deconvolution_method/plots/optimization/recreated_image.png')
 
    return fit_moffat_from_image(image_copy, sources, crop_size=2*3)


'''
def fit_psf_parameters(image, sources, cutout_size=15):
    star_positions = []
    alphas = []
    betas = []

    for star in sources:
        x_star, y_star = int(star['xcentroid']), int(star['ycentroid'])
    
        # Extract a 15x15 cutout around the star
        cutout = Cutout2D(image, (x_star, y_star), (cutout_size, cutout_size))
        cutout_data = cutout.data
        background = np.median(cutout_data)
        cutout_data -= background
        
        # Generate coordinate grid for fitting
        y, x = np.indices(cutout_data.shape)
        x = x - cutout_data.shape[1] // 2
        y = y - cutout_data.shape[0] // 2
        
        # Fit the Moffat function to this star
        try :
            popt, _ = curve_fit(moffat_2d, (x.ravel(), y.ravel()), cutout_data.ravel(), p0=[2, 2.5])
        except RuntimeError:
            psf = moffat_2d((x, y), 2, 2.5)
            plt.imshow(psf, cmap='hot')
            plt.savefig(f"deconvolution_method/fitting_{x_star}.png")
        
        print(popt)
        
        alphas.append(popt[0])
        betas.append(popt[1])
        star_positions.append((x_star, y_star))
    return alphas, betas, star_positions

def fit_moffat_from_image_2(blurred_image, sources, crop_size=15):
    """
    Extracts individual PSFs and fits Moffat parameters for each.
    """
    star_positions = []
    
    # Extraction of regions of interest (ROIs)
    psf_patches = []
    hsize = crop_size // 2
    alphas = []
    betas=[]

    for star in sources:
        x_star, y_star = int(star['xcentroid']), int(star['ycentroid'])
        
        # Ensure the patch is within bounds of the image
        if hsize < x_star < blurred_image.shape[1] - hsize and hsize < y_star < blurred_image.shape[0] - hsize:
            # Extract the patch around the star
            patch = blurred_image[y_star - hsize:y_star + hsize + 1, x_star - hsize:x_star + hsize + 1]
            
            # Thresholding: Set pixels below half of the max value to 0
            patch[patch < np.max(patch) / 2] = 0
            
            # Normalize the patch
            patch_normalized = patch / np.sum(patch)
            psf_patches.append(patch_normalized)
            
            # Fit Moffat parameters for the current patch
            star_positions.append((x_star, y_star))
            
            # Perform the fit for each PSF patch individually
            def cost(params):
                alpha, beta = params
                model_psf = moffat_psf(patch.shape, alpha, beta)
                return np.sum((patch_normalized - model_psf) ** 2)
            
            # Fit the Moffat PSF to the patch
            initial_guess = [2.0, 2.5]  # Initial guess for alpha and beta
            bounds = [(0.1, 10), (0.1, 10)]  # Bounds for alpha and beta
            result = minimize(cost, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            # Store the fitted parameters for this patch
            alphas.append(result.x[0])
            betas.append(result.x[1])
    plt.figure()
    #plt.subplot(1, 2, 1)
    plt.scatter(alphas, betas)
    #plt.subplot(1, 2, 2)
    #plt.scatter(len(betas), betas)
    plt.savefig('deconvolution_method/alphas_betas.png')
    exit()
    
    # Return the fitted parameters, the normalized PSFs, and the star positions
    return fitted_params, psf_patches, star_positions
'''