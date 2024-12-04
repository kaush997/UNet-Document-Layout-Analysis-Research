def enhance_contrast(image):
    """
    Enhances the contrast of the image using CLAHE.
    :param image: Grayscale image.
    :return: Enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image
