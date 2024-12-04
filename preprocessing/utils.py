import os

def save_image(image, output_path):
    """
    Saves the processed image to the specified path.
    :param image: Image to save.
    :param output_path: Path to save the image.
    """
    cv2.imwrite(output_path, image)
