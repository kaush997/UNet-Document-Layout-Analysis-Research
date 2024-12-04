import cv2
import numpy as np

def remove_noise(image_path):
    """
    Removes noise using Gaussian and Median filters.
    :param image_path: Path to the image.
    :return: Denoised image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    median_filtered = cv2.medianBlur(image, 5)
    gaussian_filtered = cv2.GaussianBlur(median_filtered, (5, 5), 0)
    return gaussian_filtered
