import numpy as np
import matplotlib.pyplot as plt

def generate_grid_image(image_size=(256, 256), line_width=10, grid_spacing=20):
    """
    Génère une image de grille avec des lignes noires et blanches.
    
    Parameters:
    - image_size: tuple (hauteur, largeur) de l'image.
    - line_width: largeur des lignes noires.
    - grid_spacing: distance entre les lignes noires (inclut la largeur des lignes).

    Returns:
    - grid_image: tableau 2D représentant l'image de la grille.
    """
    height, width = image_size
    grid_image = np.ones((height, width))  # Initialise l'image avec du blanc (1)

    for y in range(0, height, grid_spacing):
        grid_image[y:y + line_width, :] = 0  # Lignes horizontales noires

    for x in range(0, width, grid_spacing):
        grid_image[:, x:x + line_width] = 0  # Lignes verticales noires

    return grid_image

def generate_sinusoidal(image_shape):
    # Crée une image sinusoïdale
    x = np.linspace(0, 2 * np.pi, image_shape[1])
    y = np.linspace(0, 2 * np.pi, image_shape[0])
    sinusoidal_image = 0.5 * (np.sin(5 * x) + np.sin(5 * y[:, None])) + 0.5
    return sinusoidal_image

def generate_star_field(image_shape=(256, 256), num_stars=10, star_intensity=(200, 255), star_radius=3):
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

# Exemple d'utilisation
star_field = generate_star_field()

# Affichage de l'image
plt.figure(figsize=(8, 8))
plt.title("Champ d'étoiles simulé")
plt.imshow(star_field, cmap='gray')
plt.colorbar(label='Intensité (niveaux de gris)')
plt.axis('off')
plt.show()

# Exemple d'utilisation
image_size = (256, 256)  # Taille de l'image
line_width = 5           # Largeur des lignes noires
grid_spacing = 20        # Espacement entre les lignes (pixels)

#grid_image = generate_grid_image(image_size, line_width, grid_spacing)
sin_image = generate_sinusoidal(image_size)
# Affichage de l'image
plt.figure(figsize=(6, 6))
plt.imshow(star_field, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.savefig('./images/original/star_image.jpg', bbox_inches='tight')