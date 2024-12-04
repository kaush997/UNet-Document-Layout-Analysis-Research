def normalize_image(image, target_size=(512, 512)):
    """
    Normalizes the image to a consistent size.
    :param image: Grayscale image.
    :param target_size: Desired size (width, height).
    :return: Resized image.
    """
    normalized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return normalized_image
