import os
import random
import torch
import cv2
import matplotlib.pyplot as plt
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
import numpy as np

def visualize_augmentations(dataset, num_samples=5):
    """
    Visualize augmented images and masks from a dataset.

    Args:
        dataset (Dataset): Instance of a dataset class implementing __getitem__.
        num_samples (int): Number of samples to visualize.
    """
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, mask = dataset[idx]

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        # Normalize image for display if necessary
        if image.max() > 1:
            image = image / 255.0

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[0].set_title("Augmented Image")
        ax[0].axis("off")

        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("Augmented Mask")
        ax[1].axis("off")

        plt.show()

if __name__ == "__main__":
    from dataLoader import EnhancedDocumentLayoutDataset

    # Example usage
    dataset = EnhancedDocumentLayoutDataset(
        images_dir="data/images",
        labels_dir="data/combined_labels",
        csv_path="data/splits/train.csv",
        image_size=(512, 512),
        transform=True
    )

    visualize_augmentations(dataset, num_samples=5)
