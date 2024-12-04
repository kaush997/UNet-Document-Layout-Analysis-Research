import random

def augment_image(image):
    """
    Applies random augmentations (scaling, brightness adjustment).
    :param image: Grayscale image.
    :return: Augmented image.
    """
    # Random brightness adjustment
    factor = random.uniform(0.8, 1.2)
    augmented_image = np.clip(image * factor, 0, 255).astype(np.uint8)

    # Zoom effect
    scale_factor = random.uniform(0.9, 1.1)
    height, width = image.shape
    center = (width // 2, height // 2)
    zoomed_image = cv2.resize(
        image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR
    )
    return zoomed_image
