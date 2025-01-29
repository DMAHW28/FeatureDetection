import numpy as np
from skimage.color import rgb2gray


"""Le détecteur FAST va identifier les coins en vérifiant l'intensité des pixels autour d'un pixel central 
dans une région circulaire (généralement de 16 pixels). Pour qu'un pixel soit considéré comme un coin, il faut qu'au moins n pixel consécutifs doivent avoir une intensité supérieur à Ip + t (donc inversement plus sombre que Ip - t).""" 


"""On va donc créer une fonction qui va réaliser cette vérification."""


def seuil_ver(circle, Ip, seuil_t, n):
    """
    Paramètres:
    - circle: ndarray, intensités des 16 pixels autour du pixel central
    - Ip: int, intensité du pixel central
    - seuil_t: int, valeur de seuil
    - n: int, nombre minimum de pixels consécutifs

    Renvoie:
    - is_corner: bool, True si c'est un coin, False sinon
    - score: float, score du coin
    """

    #conversion en int16 car sinon il y avait problème de overflow
    Ip = np.array(Ip, dtype=np.int16)
    circle = np.array(circle, dtype=np.int16)
    
    #identification des pixels plus clairs ou plus sombres selon le seuil
    clair = circle > (Ip + seuil_t)
    sombre = circle < (Ip - seuil_t)

    #on construit un vecteur binaire qui va indiquer si les pixels sont plus clairs ou plus sombres que les seuils
    binary_vector = clair | sombre

    #pour gérer la circularité on devra dupliquer le vecteur
    bv = np.concatenate((binary_vector, binary_vector))

    #convolution avec un filtre de taille n rempli de 1
    filter_n = np.ones(n)
    conv_result = np.convolve(bv.astype(int), filter_n, mode='valid')
    max_conv = np.max(conv_result)
    if max_conv >= n:
        # Coin détecté
        #si une séquence suffisamment longue est détectée, on calcule le score
        score = np.sum(np.abs(circle - Ip))
        return True, score
    else:
        return False, 0



"""La fonction de detection FAST pour parcourir l'image afin de détecter les coins."""


def fast_detection(image, n=12, seuil_t=20):
    """
    Paramètres:
    - image: ndarray 2D ou 3D, image en niveaux de gris ou en couleur
    - n: int, nombre de pixels contigus
    - seuil_t: int, valeur de seuil

    Renvoie:
    - corners: liste de tuples (y, x), coordonnées des coins détectés
    - scores: liste de float, scores correspondants des coins
    """

    #on vérifier si l'image est en couleur
    if image.ndim == 3:
        image = rgb2gray(image)
        image = (image * 255).astype(np.uint8)
    elif image.ndim == 2:
        # Si l'image est déjà en niveaux de gris, s'assurer qu'elle est en uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

    #on initialise les coordonnées des 16 pixels sur le cercle de rayon 3
    circle_pixels = np.array([
        (0, 3), (1, 3), (2, 2), (3, 1),
        (3, 0), (3, -1), (2, -2), (1, -3),
        (0, -3), (-1, -3), (-2, -2), (-3, -1),
        (-3, 0), (-3, 1), (-2, 2), (-1, 3)
    ])

    h, w = image.shape

    corners = []
    scores = []

    #parcourir chaque pixel de l'image (en évitant les bords)
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            I_p = image[y, x]
            #intensités des 16 pixels du cercle
            circle_intensities = image[y + circle_pixels[:, 0], x + circle_pixels[:, 1]]

            #vérification si le pixel est un coin en utilisant seuil_ver
            is_corner, score = seuil_ver(circle_intensities, I_p, seuil_t, n)

            if is_corner:
                corners.append((y, x))
                scores.append(score)

    return corners, scores


"""Applique une suppression des non-maxima pour affiner les résultats et éviter des amas"""


def suppr_non_maxima(corners, scores, image_shape, window_size=3):
    """
    Paramètres:
    - corners: liste de tuples (y, x), coordonnées des coins détectés
    - scores: liste de float, scores correspondants
    - image_shape: tuple, dimensions de l'image
    - window_size: int, taille de la fenêtre pour la suppression

    Renvoie:
    - suppressed_corners: liste de tuples (y, x), coins après suppression
    """

    #on commence par initialiser une carte des réponses avec les scores des coins
    corner_response = np.zeros(image_shape)
    for (y, x), score in zip(corners, scores):
        corner_response[y, x] = score

    #et on les copie pour la suppression des non maxima
    suppressed_response = np.copy(corner_response)
    h, w = image_shape
    for y in range(h):
        for x in range(w):
            if corner_response[y, x] != 0:
                #on définit la région locale autour du pixel
                y_min = max(0, y - window_size)
                y_max = min(h, y + window_size + 1)
                x_min = max(0, x - window_size)
                x_max = min(w, x + window_size + 1)
                local_region = corner_response[y_min:y_max, x_min:x_max]
                if corner_response[y, x] < np.max(local_region):
                    suppressed_response[y, x] = 0

    #extraction des coins restants
    suppressed_corners = list(zip(*np.nonzero(suppressed_response)))
    suppressed_scores = [suppressed_response[y, x] for y, x in suppressed_corners]
    return suppressed_corners, suppressed_scores
