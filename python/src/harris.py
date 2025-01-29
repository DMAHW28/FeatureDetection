import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Création du handle pour la légende
red_circle = [mpatches.Circle((0, 0), radius=5, color='red', label='Coins Harris')]

def non_maxima_suppression(C, window_size=5):
    """
    Fait la suppression des non-maxima
    :parameters
    C: Scores d'Harris
    window_size: Taille de la fenêtre de voisin
    :returns
    local_max: Scores d'Harris après suppression
    """
    # On copie le score d'Harris
    local_max = np.copy(C)

    # On définit les limites de la fenêtre de voisinage
    half_window = window_size // 2

    for i in range(half_window, C.shape[0] - half_window):
        for j in range(half_window, C.shape[1] - half_window):
            # On définit la fenêtre de voisinage
            window = C[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]

            # On met à 0 tous les points qui n’ont pas la valeur maximale dans le voisinage
            if C[i, j] != np.max(window):
                local_max[i, j] = 0
    return local_max


def rotate_image(image, angle = 30):
    """
    Rotation de l'image
    :parameters
    image: l'image
    angle: angle de rotation
    :returns
    rotated: image après rotation
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def compute_gradients(image):
    """
    Calcul du gradient de I, Ix, Iy par difference finie
    :parameters
    image: l'image
    :returns
    Ix: derivée de l'image selon x
    Iy: derivée de l'image selon y
    """
    # dimension de l'image
    h, w = image.shape

    # Les derivées de l'images
    Ix = np.zeros_like(image, dtype=np.float64)
    Iy = np.zeros_like(image, dtype=np.float64)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # On met à jour les derivées par difference finie selon x et y
            Ix[y, x] = (image[y, x + 1] - image[y, x - 1]) / 2.0
            Iy[y, x] = (image[y + 1, x] - image[y - 1, x]) / 2.0
    return Ix, Iy


def rectangularFilter(Ix2, Iy2, Ixy, kernel_size):
    """
    Filtrage par filtre rectangulaire
    :parameters
    kernel_size: taille du noyau du filtre
    Ix2: dérivée selon Ix ** 2
    Iy2: dérivée selon Iy ** 2
    Ixy: produit des dérivées Ix * Iy
    :returns
    Sxx: filtrage rectangulaire de Ix2
    Syy: filtrage rectangualaire de Iy2
    Sxy: filtrage rectangualaire de Ixy
    """
    Sxx = cv2.boxFilter(Ix2, -1, kernel_size)
    Syy = cv2.boxFilter(Iy2, -1, kernel_size)
    Sxy = cv2.boxFilter(Ixy, -1, kernel_size)

    return Sxx, Syy, Sxy


def gaussianFilter(Ix2, Iy2, Ixy, kernel_size, sigma):
    """
    Filtrage par filtre gaussien
    :parameters
    kernel_size: taille du noyau du filtre
    sigma: écart-type du filtre gaussien
    Ix2: dérivée selon Ix ** 2
    Iy2: dérivée selon Iy ** 2
    Ixy: produit des dérivées Ix * Iy
    :returns
    Sxx: filtrage gaussien de Ix2
    Syy: filtrage gaussien de Iy2
    Sxy: filtrage gaussien de Ixy
    """
    Sxx = cv2.GaussianBlur(Ix2, kernel_size, sigmaX=sigma)
    Syy = cv2.GaussianBlur(Iy2, kernel_size, sigmaX=sigma)
    Sxy = cv2.GaussianBlur(Ixy, kernel_size, sigmaX=sigma)

    return Sxx, Syy, Sxy


def harris_detector_score(image_file, window_type="Gaussian", window_size=3, k=0.055, sigma=1, rotate=False, angle=30):
    """
    le calcul du score d'Harris
    :parameters
    image_file: nom du fichier image
    window_type: type de la fenêtre de pondération
    window_size: taille de la fenêtre de pondération
    k: critère d'Harris
    sigma: écart-type du filtre gaussien
    rotate: booléen pour faire la rotation de l'image True pour oui
    angle: angle de rotation
    :returns
    C: le score d'Harris
    """
    # Lit l'image en niveau de gris
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    # On définit la dimension du noyau pour le filtrage
    kernel_size = (window_size, window_size)

    # On teste la rotation
    if rotate:
        image = rotate_image(image, angle)

    # Calcul du gradient de l'image
    Ix, Iy = compute_gradients(image)

    # Calcul des produits des dérivées
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    # On applique la fenêtre de pondération sur les produits des dérivées en fonction du type de fenêtre utilisé
    if window_type == "Rectangular":
        Sxx, Syy, Sxy = rectangularFilter(Ix2, Iy2, Ixy, kernel_size)
    elif window_type == "Gaussian":
        Sxx, Syy, Sxy = gaussianFilter(Ix2, Iy2, Ixy, kernel_size, sigma)
    else:
        raise ValueError("Type 'Rectangular' or 'Gaussian' des choix.")

    # Déterminant de la matrice M
    det_M = Sxx * Syy - Sxy ** 2
    # Trace de la matrice M
    trace_M = Sxx + Syy

    # Calcul du score Harris
    C = det_M - k * (trace_M ** 2)
    return C


def harris_detector_image(image_file, corners, rotate = False, angle = 30, r = 2):
    """
    Affiche les points détectés sur l'image
    :parameters
    image_file: nom du fichier image
    corners: coins détectés
    rotate: booléen pour faire la rotation de l'image True pour oui
    angle: angle de rotation
    r: taille des points affichés
    :returns
    cv2.cvtColor(harris_image, cv2.COLOR_BGR2RGB): image avec les points détectés sur l'image
    corner_number: nombre de coins détectés
    points: liste des positions des points détectés
    """
    # Lit l'image
    harris_image = cv2.imread(image_file)

    # On teste la rotation
    if rotate:
        harris_image = rotate_image(harris_image, angle)

    # On compte le nombre de coins détectés
    corner_number = np.sum(corners.astype(int))

    # On cherche la position des coins
    X, Y = np.where(corners == True)
    points = []
    for i, j in zip(X, Y):

        # On ajoute les points dans une liste
        points.append((j, i))

        # On affiche les points sur l'image
        cv2.circle(harris_image, (j, i), r, (0, 0, 255), -1)

    return cv2.cvtColor(harris_image, cv2.COLOR_BGR2RGB), corner_number, points



def compute_harris_detector(image_file, window_type="Gaussian", window_size=3, k=0.055, sigma = 1, nms = True, nms_window = 5, rotate = False, angle = 30, r = 2, threshold = 0.01):
    """
    Retourne l'image avec les points détectés sur l'image à partir des différents paramètres
    :parameters
    image_file: nom du fichier image
    window_type: type de la fenêtre de pondération
    window_size: taille de la fenêtre de pondération
    k: critère d'Harris
    sigma: écart-type du filtre gaussien
    nms: booléen pour activer de la suppression des non-maxima
    rotate: booléen pour faire la rotation de l'image True pour oui
    angle: angle de rotation
    r: taille des points affichés
    :returns
    cv2.cvtColor(harris_image, cv2.COLOR_BGR2RGB): image avec les points détectés sur l'image
    corner_number: nombre de coins détectés
    points: liste des positions des points détectés
    """

    # Estimation du score d'Harris
    C = harris_detector_score(image_file, window_type, window_size, k, sigma, rotate, angle = angle)

    # On teste la suppression des non-maxima
    if nms:
        # On supprime les non-maxima
        C = non_maxima_suppression(C, window_size=nms_window)

    # On estime le seuil, threshold% du score maximal
    rate = C.max() * threshold
    # On estime les coins
    corners =  C > rate
    return  harris_detector_image(image_file, corners, rotate = rotate, angle = angle, r = r)


def display(images, titles):
    # Fonction d'affichage des images dans le notebook
    x, y = len(images), len(images[0])
    fig , axs = plt.subplots(x, y, figsize = (30, 10))
    for i in range(x):
        for j in range(y):
            if x == 1:
                axs[j].imshow(images[i][j])
                axs[j].set_title(f'{titles[i][j]}')
                axs[j].legend(handles=red_circle, loc='upper right')
                axs[j].axis("off")
            else:
                axs[i, j].imshow(images[i][j])
                axs[i, j].set_title(f'{titles[i][j]}')
                axs[i, j].legend(handles=red_circle, loc='upper right')
                axs[i, j].axis("off")
    plt.tight_layout()
    plt.show()

def display(images, titles):
    x, y = len(images), len(images[0])
    fig , axs = plt.subplots(x, y, figsize = (30, 8))
    for i in range(x):
        for j in range(y):
            if x == 1:
                axs[j].imshow(images[i][j])
                axs[j].set_title(f'{titles[i][j]}')
                axs[j].legend(handles=red_circle, loc='upper right')
                axs[j].axis("off")
            else:
                axs[i, j].imshow(images[i][j])
                axs[i, j].set_title(f'{titles[i][j]}')
                axs[i, j].legend(handles=red_circle, loc='upper right')
                axs[i, j].axis("off")
    plt.tight_layout()
    plt.show()

