### dataLoader.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class EnhancedDocumentLayoutDataset(Dataset):
    def __init__(self, images_dir, labels_dir, csv_path, image_size, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.transform = transform or A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        # Read CSV file to load image-label mappings
        with open(csv_path, 'r') as f:
            self.image_label_pairs = [line.strip().split(',') for line in f]

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        image_name, label_name = self.image_label_pairs[idx]
        image_path = os.path.join(self.images_dir, image_name)
        label_path = os.path.join(self.labels_dir, label_name)

        # Load image and mask
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = self.load_mask(label_path)

        # Apply augmentations
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']

    def load_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure mask values are within range [0, num_classes - 1]
        mask = np.clip(mask, 0, 25)  # Adjust based on output_channels (26 classes)

        return mask