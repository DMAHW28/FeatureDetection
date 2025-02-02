import numpy as np
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


def extract_descriptors(image, points, window_size=3):
    """
    Extrait un bloc d'intensité de pixels autour de chaque point d'intérêt
    :parameters
    image: l'image
    points: liste des positions des points détectés
    window_size: Taille de la fenêtre de voisin
    :returns
    descriptors: vecteur du bloc d'intensité de pixels autour de chaque point d'intérêt
    convert_points: liste la position des points d'intérêts qui ont été extrait
    """
    descriptors = []
    convert_points = []
    # On définit les limites de la fenêtre de voisinage
    half_window = window_size // 2

    for x, y in points:
        # On teste les limites de l'image
        if not (y - half_window < 0 or y + half_window >= image.shape[0] or x - half_window < 0 or x + half_window >=
                image.shape[1]):
            # On définit la fenêtre de voisinage
            window = image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            # On vectorise le bloc
            vect = window.flatten()
            # On ajoute le bloc et la position
            descriptors.append(vect)
            convert_points.append((x, y))
    return np.array(descriptors), convert_points


def distance_metric_eucludian(descriptor1, descriptor2):
    """
    Calcul de la distance Euclidienne entre deux descripteurs
    :parameters
    descriptor1: vecteur du bloc d'intensité de pixels autour d'un point d'intérêt de l'image1
    descriptor2: vecteur du bloc d'intensité de pixels autour d'un point d'intérêt de l'image2
    :returns
    la distance Euclidienne entre deux descripteurs
    """
    return np.sqrt(np.sum((descriptor1 - descriptor2) ** 2))


def distance_metric_correlation(descriptor1, descriptor2):
    """
    Calcul de la corrélation brute entre deux descripteurs
    :parameters
    descriptor1: vecteur du bloc d'intensité de pixels autour d'un point d'intérêt de l'image1
    descriptor2: vecteur du bloc d'intensité de pixels autour d'un point d'intérêt de l'image2
    :returns
    correlation: corrélation entre le descripteur1 et le descripteur2
    """
    desc1_norm = np.sqrt(np.sum(descriptor1 ** 2))
    desc2_norm = np.sqrt(np.sum(descriptor2 ** 2))
    correlation = np.sum(descriptor1 * descriptor2) / (desc1_norm * desc2_norm)
    return correlation


def distance_metric_correlation_norm(descriptor1, descriptor2):
    """
    Calcul de la corrélation normalisée entre deux descripteurs
    :parameters
    descriptor1: vecteur du bloc d'intensité de pixels autour d'un point d'intérêt de l'image1
    descriptor2: vecteur du bloc d'intensité de pixels autour d'un point d'intérêt de l'image2
    :returns
    correlation_norm: corrélation normalisée entre le descripteur1 et le descripteur2
    """
    desc1_norm_center = descriptor1 - np.mean(descriptor1)
    desc2_norm_center = descriptor2 - np.mean(descriptor2)
    correlation_norm = np.sum(desc1_norm_center * desc2_norm_center) / np.sqrt(
        np.sum(desc1_norm_center ** 2) * np.sum(desc2_norm_center ** 2))
    return correlation_norm


def compute_distance(descriptors1, descriptors2, metric=distance_metric_eucludian):
    """
    Calcul de la distance entre les descripteurs de 2 images
    :parameters
    descriptors1: descripteurs de l'image 1
    descriptors2: descripteurs de l'image 2
    metric: fonction pour calculer la distance (euclidienne, corrélation, etc.)
    :returns
    distances: la distance entre les descripteurs de 2 images
    """
    # Matrice qui empile dans les colonnes les distances entre chaque descripteur de l'image 1 et les descripteurs de l'image 2
    # Les distances de chaque descripteur de l'image 1 et tous les descripteurs de l'image 2  sont empilés dans les colonnes
    # Exemple la colonne 1 contient les distances entre le descripteur 1 de l'image 1 et tous les descripteurs de l'image 2
    distances = np.zeros((descriptors2.shape[0], descriptors1.shape[0]))
    # Pour chaque descripteur de l'image 1 on calcul la distance à partir d'une métrique
    for i, des1 in tqdm(enumerate(descriptors1), desc="Computing distance"):
        # On empile dans la colonne du descripteur
        distances[:, i] = np.array([metric(des1, des2) for des2 in descriptors2]).reshape((descriptors2.shape[0],))
    return distances


def matching_points_ratio(descriptors1, descriptors2, metric=distance_metric_eucludian, ratio_threshold=0.8):
    """
    Trouve les correspondances entre deux ensembles de descripteurs en appliquant le test de ratio
    :parameters
    descriptors1: descripteurs de l'image 1
    descriptors2: descripteurs de l'image 2
    metric: fonction pour calculer la distance (euclidienne, corrélation, etc.)
    ratio_threshold: seuil pour le test de ratio (d1/d2 < ratio_threshold)
    :returns:
    matches: liste des correspondances [(i, j)] où i est l'indice dans descriptors1 et j dans descriptors2
    """
    # On calcul la distance entre les descripteurs de l'image 1 et les descripteurs de l'image 2 à partir d'une métrique
    distances12 = compute_distance(descriptors1, descriptors2, metric)
    matches = []

    for i in range(distances12.shape[1]):
        # On trie les distances pour chaque descripteur de l'image 1
        sorted_indices = np.argsort(distances12[:, i])
        # On prend la distance de la meilleure correspondance
        d1 = distances12[sorted_indices[0], i]
        # On prend la distance de la deuxième meilleure correspondance
        d2 = distances12[sorted_indices[1], i]

        # On applique le test de ratioc
        if d1 / d2 < ratio_threshold:
            # On ajoute la correspondance
            matches.append((i, sorted_indices[0]))
    return matches


def matching_points_crosscheck(descriptors1, descriptors2, metric=distance_metric_eucludian):
    """
    Trouve les correspondances entre deux ensembles de descripteurs avec appariement croisé
    :parameters
    descriptors1: descripteurs de l'image 1
    descriptors2: descripteurs de l'image 2
    metric: fonction pour calculer la distance (euclidienne, corrélation, etc.)
    :returns:
    matches: liste des correspondances [(i, j)] où i est l'indice dans descriptors1 et j dans descriptors2
    """
    # Calcul des distances dans les deux sens
    distances12 = compute_distance(descriptors1, descriptors2, metric)
    distances21 = compute_distance(descriptors2, descriptors1, metric)

    matches = []
    for i in tqdm(range(distances12.shape[1]), desc='Matching'):
        # Correspondance dans un sens
        arg_min1 = np.argmin(distances12[:, i])
        # Correspondance dans l'autre sens
        arg_min2 = np.argmin(distances21[:, arg_min1])
        # On teste la réciprocité
        if arg_min2 == i:
            matches.append((i, arg_min1))

    return matches


def compute_matching_points(image1, image2, points1, points2, windows_size=3, metric=distance_metric_eucludian):
    # Extraction des descripteurs de chaque image
    descriptors1, convert_points1 = extract_descriptors(image1, points1, window_size=windows_size)
    descriptors2, convert_points2 = extract_descriptors(image2, points2, window_size=windows_size)

    # Calcul du matching des points entre les deux images
    matches = matching_points_ratio(descriptors1, descriptors2, metric)
    return matches, convert_points1, convert_points2


def display_matching(image1, image2, points1, points2, matches, nbp=30):
    """
    Dessine les correspondances entre deux images de manière plus efficace
    :parameters
    image1: image 1
    image2:  image 2
    points1: points d'intérêts 1
    points2: points d'intérêts 2
    matches: liste des correspondances des points d'intérêts [(i, j)]
    nbp: nombre de match à faire
    """
    # Convertir les images en couleur si nécessaires
    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR) if len(image1.shape) == 2 else image1
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR) if len(image2.shape) == 2 else image2

    # Concaténer les deux images
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = image1
    canvas[:h2, w1:] = image2

    # Dessiner les correspondances
    for i, j in tqdm(random.sample(matches, min(nbp, len(matches))), desc="Displaying"):
        x1, y1 = points1[i]
        x2, y2 = points2[j]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, (x1, y1), 5, color, -1)
        cv2.circle(canvas, (x2 + w1, y2), 5, color, -1)
        cv2.line(canvas, (x1, y1), (x2 + w1, y2), color, 1)
    return canvas


# Affichage de quelques patches extraits
def display_patches(image, keypoints, window_size, num_patches=5):
    half_window = window_size // 2
    plt.figure(figsize=(20, 8))
    for i in range(num_patches):
        x, y = keypoints[i]
        patch = image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
        plt.subplot(1, num_patches, i + 1)
        plt.imshow(patch, cmap='gray')
        plt.title(f'Patch autour du point ({x}, {y})')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


############################################################ LBP ###################################################################

__author__ = "EMADALY Khouzema"
__email__ = "khozemaemadaly360@gmail.com"

"""on doit calculer le Local Binary Pattern (LBP) pour un pixel donné. Cette fonction va nous servir à extraire une signature locale qui encode la texture autour d'un pixel central."""


def compute_lbp(image, x, y, radius=1, n_points=8):
    """
    Paramètres:
    - image: ndarray, image en niveaux de gris
    - x, y: coordonnées du pixel central
    - radius: rayon du cercle autour du pixel central
    - n_points: nombre de points échantillonnés sur le cercle

    Renvoie:
    - lbp_value: entier, code LBP du pixel central
    - lbp_value: entier, code LBP du pixel central
    """

    theta = 2 * np.pi / n_points
    lbp_value = 0
    center_intensity = image[y, x]

    for i in range(n_points):
        # les coordonnées du voisin
        dx = x + radius * np.cos(i * theta)
        dy = y - radius * np.sin(i * theta)

        # interpolation bilinéaire pour récupérer l'intensité du voisin
        x1, y1 = int(np.floor(dx)), int(np.floor(dy))
        x2, y2 = x1 + 1, y1 + 1
        if x1 < 0 or x2 >= image.shape[1] or y1 < 0 or y2 >= image.shape[0]:
            # mais si le voisin est en dehors de l'image, on continue
            continue

        Ia = image[y1, x1]
        Ib = image[y1, x2]
        Ic = image[y2, x1]
        Id = image[y2, x2]
        wa = (x2 - dx) * (y2 - dy)
        wb = (dx - x1) * (y2 - dy)
        wc = (x2 - dx) * (dy - y1)
        wd = (dx - x1) * (dy - y1)
        neighbor_intensity = wa * Ia + wb * Ib + wc * Ic + wd * Id
        # Comparaison avec le pixel central
        if neighbor_intensity >= center_intensity:
            lbp_value += 1 << i
    return lbp_value


"""Extrait les descripteurs LBP pour des points d'intérêt dans une image.
Les descripteurs LBP permettent de résumer la texture locale autour de chaque point."""


def extract_lbp_descriptors(image, points, radius=1, n_points=8):
    """
    Paramètres:
    - image: ndarray, image en niveaux de gris
    - points: list of tuples, coordonnées des points d'intérêt (x, y)
    - radius: int, rayon du voisinage
    - n_points: int, nombre de points échantillonnés sur le cercle

    Renvoie:
    - descriptors: ndarray, descripteurs LBP sous forme d'histogrammes
    - valid_points: list of tuples, points où les descripteurs ont été calculés
    """
    descriptors = []
    valid_points = []
    half_window = radius

    # on initialise une image LBP vide
    lbp_image = np.zeros_like(image, dtype=np.uint32)

    # on calcule les codes LBP pour chaque pixel de l'image
    for y in range(radius, image.shape[0] - radius):
        for x in range(radius, image.shape[1] - radius):
            lbp_image[y, x] = compute_lbp(image, x, y, radius, n_points)

    # pour chaque point d'intérêt, extraire un histogramme LBP
    for x, y in points:
        x, y = int(x), int(y)
        if y - half_window >= 0 and y + half_window < image.shape[0] and x - half_window >= 0 and x + half_window < \
                image.shape[1]:
            patch = lbp_image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            hist, _ = np.histogram(patch.ravel(), bins=2 ** n_points, range=(0, 2 ** n_points))
            hist = hist.astype(float)
            hist /= hist.sum()  # Normalisation
            descriptors.append(hist)
            valid_points.append((x, y))
    return np.array(descriptors), valid_points


"""on calcul la distance chi-carré entre deux histogrammes.
C'est utilisée pour comparer la similarité entre deux descripteurs LBP"""


def distance_metric_chi2(hist1, hist2):
    """
    Paramètres:
    - hist1, hist2: histogrammes des descripteurs à comparer

    Renvoie:
    - distance: float, distance chi-carré
    """
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-10))


"""rpour trouver les correspondances entre deux ensembles de descripteurs LBP.
    pour cela on utilise le test de ratio pour assurer la qualité des correspondances."""


def matching_points_lbp(descriptors1, descriptors2, ratio_threshold=0.8):
    """
    Paramètres:
    - descriptors1: ndarray, descripteurs de l'image 1
    - descriptors2: ndarray, descripteurs de l'image 2
    - ratio_threshold: float, seuil du test de ratio

    Renvoie:
    - matches: list of tuples, correspondances (indice_descripteur1, indice_descripteur2)
    """
    distances = compute_distance(descriptors1, descriptors2, metric=distance_metric_chi2)
    matches = []
    for i in range(distances.shape[1]):
        # trie les distances pour chaque descripteur
        sorted_indices = np.argsort(distances[:, i])
        d1 = distances[sorted_indices[0], i]
        d2 = distances[sorted_indices[1], i]
        if d1 / (d2 + 1e-10) < ratio_threshold:
            matches.append((i, sorted_indices[0]))
    return matches


# Fonction d'affichage des points d'intérêt
def display_keypoints(image, points, title):
    image_with_points = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(image_with_points, (int(x), int(y)), 3, (0, 0, 255), -1)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

