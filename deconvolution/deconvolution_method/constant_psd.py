import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.restoration import richardson_lucy
from skimage.transform import resize  # Pour redimensionner l'image

def resize_image(image, target_size=(255, 255)):
    """
    Redimensionne une image à la taille spécifiée.

    Parameters:
    - image: tableau 2D ou 3D représentant l'image (grayscale ou RGB).
    - target_size: tuple, nouvelle taille de l'image (hauteur, largeur).

    Returns:
    - resized_image: tableau redimensionné.
    """
    if image.shape[:2] != target_size:
        resized_image = resize(image, target_size, preserve_range=True, anti_aliasing=True)
        return resized_image.astype(image.dtype)  # Conserve le type de l'image originale
    return image

def renormalize_to_minus_one_to_one(image):
    """
    Renormalise les pixels d'une image pour qu'ils soient compris entre -1 et 1.
    
    Parameters:
    - image: numpy array, image à normaliser.
    
    Returns:
    - renormalized_image: numpy array, image avec des pixels entre -1 et 1.
    """
    # Trouver les valeurs minimale et maximale de l'image
    min_val = np.min(image)
    max_val = np.max(image)

    # Renormaliser entre -1 et 1
    renormalized_image = 2 * (image - min_val) / (max_val - min_val) - 1
    return renormalized_image

# Step 2 : create radial psf function with constant parameter
def generate_radial_psf(image_shape, radial_function, normalize=True):
    """
    Génère une PSF dépendant de r (distance radiale au centre de l'image).
    
    Parameters:
    - image_shape: tuple (hauteur, largeur) de l'image.
    - radial_function: fonction définissant la dépendance radiale PSF(r).
    - normalize: bool, pour normaliser la PSF afin que la somme = 1.

    Returns:
    - psf: tableau 2D représentant la PSF radiale.
    """
    h, w = image_shape
    y, x = np.meshgrid(np.linspace(-h//2, h//2, h), np.linspace(-w//2, w//2, w), indexing='ij')
    r = np.sqrt(x**2 + y**2)  # Calcul de la distance radiale r
    psf = radial_function(r)  # Applique la fonction radiale
    
    if normalize:
        psf /= np.sum(psf)  # Normalisation pour que la somme des valeurs = 1
    
    return psf

# Définir une fonction radiale (exemple : Gaussienne)

# Step 1 : image loading
image = plt.imread('./images/original/grid_image.jpg')
resized_image = resize_image(image, target_size=(128, 128))
if resized_image.ndim == 3:  # Si c'est une image RGB, la convertir en niveaux de gris
    resized_image = np.mean(resized_image, axis=2)
resized_image = renormalize_to_minus_one_to_one(resized_image)

sigma = 0.1  # Largeur du flou
radial_function = lambda r: np.exp(-r**2 / (2 * sigma**2))  # PSF(r) = exp(-r² / (2 sigma²))

# Générer la PSF pour la taille de l'image
image_shape = resized_image.shape
psf = generate_radial_psf(image_shape, radial_function)
print("ok1")

# Step 3 : convolution
blurred_image = convolve2d(resized_image, psf, mode='same', boundary='wrap')
print("ok2")

# Step 4 : deconvolution
deconvolved = richardson_lucy(blurred_image, psf, num_iter=20)
print("ok3")

# Affichage de la PSF
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("PSF radiale (2D)")
plt.imshow(psf, cmap='hot')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Profil radial (1D)")
r_values = np.linspace(0, 128, 128)  # Jusqu'au bord (distance max au centre)
psf_1d = radial_function(r_values)
plt.plot(r_values, psf_1d)
plt.xlabel("Distance radiale r")
plt.ylabel("PSF(r)")
plt.grid()
plt.savefig('./deconvolution_method/psf_radiale.png')

# Show the blurred image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Image originale")
plt.imshow(resized_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Image floue")
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')
plt.savefig('./images/blurred/blurred_image_1.png')

#Show the deconvolution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Image floue")
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Image défloutée")
plt.imshow(deconvolved, cmap='gray')
plt.axis('off')
plt.savefig('./images/deblurred/deblurred_image_1.png')

