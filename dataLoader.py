import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from sklearn.model_selection import train_test_split
import csv


class EnhancedDocumentLayoutDataset(Dataset):
    def __init__(self, images_dir, labels_dir, image_label_pairs, image_size, transforms=None):
        """
        Args:
            images_dir (str): Directory containing the raw images in .jpg format.
            labels_dir (str): Directory containing the label masks in .png format.
            image_label_pairs (list): List of tuples mapping images to labels.
            image_size (tuple): Tuple specifying the image size (height, width).
            transforms (callable, optional): Albumentations transformations for data augmentation.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.image_label_pairs = image_label_pairs
        self.transforms = transforms or A.Compose([
            A.Resize(*image_size),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.01, contrast_limit=0.01, p=0.5),
                A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=2, val_shift_limit=2, p=0.5),
            ], p=0.5),
            A.ToGray(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        image_name, label_name = self.image_label_pairs[idx]

        # Load raw image (.jpg format)
        image_path = os.path.join(self.images_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load label mask (.png format)
        label_path = os.path.join(self.labels_dir, label_name)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image is None or label is None:
            raise RuntimeError(f"Error loading image ({image_path}) or label ({label_path})")

        # Apply transformations
        augmented = self.transforms(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        return image, label


def split_data(images_dir, labels_dir, mapping_csv, test_size=0.2, val_size=0.1):
    image_label_pairs = []

    # Read mapping from CSV
    with open(mapping_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            image_label_pairs.append((row[0], row[1]))

    train_val_pairs, test_pairs = train_test_split(image_label_pairs, test_size=test_size, random_state=42)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=val_size, random_state=42)

    return train_pairs, val_pairs, test_pairs