import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter


def Img_gradient(image):
    filter_x = np.array([[-1, 1], [-1, 1]]) / 4.0
    filter_y = filter_x.T
    I_x = cv2.filter2D(image, -1, filter_x)
    I_y = cv2.filter2D(image, -1, filter_y)
    return I_x, I_y


def suppression_maxima_locaux(C, window_size=3):
    # Initialiser une image de même taille que C pour stocker les maxima locaux
    maxima_locaux = np.zeros_like(C)
    half_window = window_size // 2

    # Parcourir chaque pixel (éviter les bords en fonction de la taille de la fenêtre)
    for i in range(half_window, C.shape[0] - half_window):
        for j in range(half_window, C.shape[1] - half_window):
            # Extraire une fenêtre autour du pixel (i, j)
            window = C[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]

            # Vérifier si le pixel courant est un maximum local
            if C[i, j] == np.max(window) and C[i, j] > 0:  # Supprimer les non-maxima et les valeurs nulles
                maxima_locaux[i, j] = C[i, j]

    return maxima_locaux


def non_maximum_suppression(C, corners, edges, flat):
    # Suppression of non-maximum pixels
    dilated = cv2.dilate(C, None)
    local_max = (C == dilated)

    corners_nms = local_max & corners
    edges_nms = local_max & edges
    flat_nms = local_max & flat
    return corners_nms, edges_nms, flat_nms


def det_corners_flat_edges(C, threshold=0.01):
    # Normalisation
    C /= C.max()
    # Trouver les coins
    corners = (C > threshold)  # Coins avec des valeurs de C > seuil
    # Trouver les bords
    edges = (C < -threshold)  # Bords avec des valeurs de C < -seuil
    # Trouver les régions plates
    flat = (abs(C) < threshold)  # Régions où |C| est inférieur au seuil
    return corners, edges, flat


def Img_gradient_with_Sobel(image, ksize = 3):
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    return Ix, Iy


def rectangularFilter(Ix2, Iy2, Ixy, kernel_size):
    Sxx = cv2.boxFilter(Ix2, -1, kernel_size)
    Syy = cv2.boxFilter(Iy2, -1, kernel_size)
    Sxy = cv2.boxFilter(Ixy, -1, kernel_size)
    return Sxx, Syy, Sxy


def gaussianFilter(Ix2, Iy2, Ixy, kernel_size, sigma):
    Sxx = cv2.GaussianBlur(Ix2, kernel_size, sigmaX=sigma)
    Syy = cv2.GaussianBlur(Iy2, kernel_size, sigmaX=sigma)
    Sxy = cv2.GaussianBlur(Ixy, kernel_size, sigmaX=sigma)
    return Sxx, Syy, Sxy


def harris_detector(image_file, window_type="gaussian", window_size=3, k=0.05, sigma = 1):
    # Load image and Convert image to grayscale
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    kernel_size = (window_size, window_size)
    # Calculate image gradients
    Ix, Iy = Img_gradient_with_Sobel(image)

    # Calculate products of gradients
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    # Apply weighting window
    if window_type == "Rectangular":
        Sxx, Syy, Sxy = rectangularFilter(Ix2, Iy2, Ixy, kernel_size)
    elif window_type == "Gaussian":
        Sxx, Syy, Sxy = gaussianFilter(Ix2, Iy2, Ixy, kernel_size, sigma)
    else:
        raise ValueError("Type 'Rectangular' or 'Gaussian' des choix.")

    # Compute the Harris response
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    C = det_M - k * (trace_M ** 2)

    return C


def display(image_file, corners, edges, flats):
    harris_image = cv2.imread(image_file)
    harris_image[corners == True] = [0, 0, 255]
    harris_image[edges == True] = [255, 0, 0]
    harris_image[flats == True] = [0, 0, 0]

    return cv2.cvtColor(harris_image, cv2.COLOR_BGR2RGB)
