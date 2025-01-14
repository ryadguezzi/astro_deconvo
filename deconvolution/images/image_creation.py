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

# Exemple d'utilisation
image_size = (256, 256)  # Taille de l'image
line_width = 5           # Largeur des lignes noires
grid_spacing = 20        # Espacement entre les lignes (pixels)

grid_image = generate_grid_image(image_size, line_width, grid_spacing)

# Affichage de l'image
plt.figure(figsize=(6, 6))
plt.imshow(grid_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.savefig('./images/original/grid_image.jpg', bbox_inches='tight')