import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def generate_star_field(image_shape=(256, 256), num_stars=25, star_intensity=(200, 255), star_radius=3):
    """
    Génère une image simulant un champ d'étoiles aléatoires.

    Parameters:
    - image_shape: tuple (hauteur, largeur), taille de l'image.
    - num_stars: int, nombre d'étoiles à générer.
    - star_intensity: tuple (min, max), plage d'intensités pour les étoiles (valeurs en niveaux de gris).
    - star_radius: int, rayon des étoiles (en pixels).

    Returns:
    - star_field: numpy array, image du champ d'étoiles.
    """
    # Crée une image noire
    star_field = np.zeros(image_shape, dtype=np.float32)

    # Génère des positions aléatoires pour les étoiles
    height, width = image_shape
    star_positions = np.random.randint(0, min(height, width), size=(num_stars, 2))

    # Ajoute des étoiles au champ
    for pos in star_positions:
        y, x = pos
        # Ajuste l'intensité et applique un rayon (disque 2D)
        intensity = np.random.uniform(star_intensity[0], star_intensity[1])
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((y_coords - y)**2 + (x_coords - x)**2)
        star_field[distance <= star_radius] += intensity

    # Clip pour rester dans des valeurs valides
    star_field = np.clip(star_field, 0, 255)

    return star_field

def generate_star_field_2(image_shape=(256, 256), num_stars=100, star_intensity=(100, 255)):
    """
    Génère une image simulant un champ d'étoiles où chaque étoile est un pixel unique
    avec une intensité lumineuse aléatoire.

    Parameters:
    - image_shape: tuple (hauteur, largeur), taille de l'image.
    - num_stars: int, nombre d'étoiles à générer.
    - star_intensity: tuple (min, max), plage d'intensités pour les étoiles (valeurs en niveaux de gris).

    Returns:
    - star_field: numpy array, image du champ d'étoiles.
    """
    # Crée une image noire
    star_field = np.zeros(image_shape, dtype=np.uint8)

    # Génère des positions aléatoires pour les étoiles
    height, width = image_shape
    star_positions = np.random.randint(0, [height, width], size=(num_stars, 2))

    # Ajoute des étoiles avec intensités aléatoires
    for y, x in star_positions:
        star_field[y, x] = np.random.randint(star_intensity[0], star_intensity[1] + 1)
    
    return star_field

def generate_star_cloud(image_shape=(256, 256), num_stars=25, star_intensity=(200, 255), star_radius=3):

    # Bruit aléatoire
    background = np.random.normal(loc=0.5, scale=0.1, size=image_shape)

    # Appliquer un flou gaussien pour créer un effet de "nuage"
    background = gaussian_filter(background, sigma=10)
    background *= 450

    background = np.clip(background, 0, 255)
    print(np.max(background))
    #plt.figure(figsize=(6, 6))
    #plt.imshow(background, cmap='gray', interpolation='nearest')
    #plt.axis('off')
    #plt.savefig('./images/original/star_cloud_image_25.jpg', bbox_inches='tight')
    #exit()

    # Nombre d'étoiles
    num_stars = 25

   # Génère des positions aléatoires pour les étoiles
    height, width = image_shape
    star_positions = np.random.randint(0, min(height, width), size=(num_stars, 2))

    # Ajoute des étoiles au champ
    for pos in star_positions:
        y, x = pos
        # Ajuste l'intensité et applique un rayon (disque 2D)
        intensity = np.random.uniform(star_intensity[0], star_intensity[1])
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((y_coords - y)**2 + (x_coords - x)**2)
        background[distance <= star_radius] += intensity
    # Clip pour rester dans des valeurs valides
    background = np.clip(background, 0, 255)

    return background

def generate_star_cloud_with_rectangle(image_shape=(256, 256), 
                                       num_stars=25, 
                                       star_intensity=(200, 255), 
                                       star_radius=3, 
                                       rectangle_intensity=100):
    height, width = image_shape

    # Créer un fond noir (ou gris neutre si tu veux mettre autre chose)
    background = np.zeros(image_shape, dtype=np.float32)
    
    # Ajouter un rectangle (nuage rectangulaire)
    rect_height, rect_width = height // 2, width // 2  # Taille du rectangle (la moitié de l'image)
    rect_top_left = (height // 4, width // 4)  # Position du coin supérieur gauche

    # Remplir le rectangle avec une intensité donnée
    background[rect_top_left[0]:rect_top_left[0]+rect_height, rect_top_left[1]:rect_top_left[1]+rect_width] = rectangle_intensity

    # 💡 (optionnel) Appliquer un flou léger au rectangle pour simuler un nuage diffus
    background = gaussian_filter(background, sigma=5)

    # Ajouter les étoiles
    star_positions = np.random.randint(0, min(height, width), size=(num_stars, 2))

    for pos in star_positions:
        y, x = pos
        intensity = np.random.uniform(star_intensity[0], star_intensity[1])
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((y_coords - y)**2 + (x_coords - x)**2)
        background[distance <= star_radius] += intensity  # Ajouter un disque autour de l'étoile

    # Clip final pour rester dans une image 8 bits (0-255)
    background = np.clip(background, 0, 255)

    return background

# Exemple d'utilisation
star_field = generate_star_field_2()

# Affichage de l'image
plt.figure(figsize=(8, 8))
plt.title("Champ d'étoiles simulé")
plt.imshow(star_field, cmap='gray')
plt.colorbar(label='Intensité (niveaux de gris)')
plt.axis('off')
plt.show()

# Affichage de l'image
plt.figure(figsize=(6, 6))
plt.imshow(star_field, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.savefig('./images/original/star_field_100.jpg', bbox_inches='tight')