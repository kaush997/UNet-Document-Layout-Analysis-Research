import os
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class DocumentLayoutDataset(Dataset):
    def __init__(self, images_dir, labels_dir, csv_path, augmentations=None):
        """
        Initializes the dataset for document layout analysis.

        :param images_dir: Path to the directory containing images.
        :param labels_dir: Path to the directory containing labels (multi-class masks).
        :param csv_path: Path to the CSV file mapping images to labels.
        :param augmentations: Albumentations augmentation pipeline.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.data = pd.read_csv(csv_path)
        self.augmentations = augmentations

        if self.data.empty:
            raise ValueError(f"CSV file at {csv_path} is empty or incorrectly formatted.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image and corresponding label
        image_name = self.data.iloc[idx, 0]  # Assumes image filename is in the first column
        label_name = self.data.iloc[idx, 1]  # Assumes label filename is in the second column

        image_path = os.path.join(self.images_dir, image_name)
        label_path = os.path.join(self.labels_dir, label_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found: {label_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image file: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)  # Loads as a single-channel mask
        if label is None:
            raise ValueError(f"Failed to load label file: {label_path}")

        # Ensure the label is a 2D numpy array
        if len(label.shape) == 3:
            label = label[:, :, 0]

        # Resize image and label to a fixed size
        target_size = (256, 256)  # Example target size; adjust as needed
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, target_size, interpolation=cv2.INTER_NEAREST)

        if self.augmentations:
            augmented = self.augmentations(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        return image, label

# Albumentations augmentation pipeline
augmentations = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.ToGray(p=0.3),  # Convert some images to grayscale
    A.Rotate(limit=3, p=0.5),
    ToTensorV2()
])

if __name__ == "__main__":
    # Define dataset paths
    images_dir = "data/images"
    labels_dir = "data/combined_labels"
    csv_path = "data/image_label_mapping.csv"

    # Load the CSV file
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading CSV file: {e}")

    # Split the data into training, validation, and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Save splits to CSV files
    train_data.to_csv("data/train_split.csv", index=False)
    val_data.to_csv("data/val_split.csv", index=False)
    test_data.to_csv("data/test_split.csv", index=False)

    print("Dataset splits created:")
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")

    # Initialize dataset and data loader for training set as an example
    dataset = DocumentLayoutDataset(images_dir, labels_dir, "data/train_split.csv", augmentations=augmentations)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Example: Iterate through one batch
    for images, labels in dataloader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break
