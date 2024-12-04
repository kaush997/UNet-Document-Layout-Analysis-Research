def binarize_image(image):
    """
    Applies adaptive thresholding to binarize the image.
    :param image: Grayscale image.
    :return: Binarized image.
    """
    binary_image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary_image
