import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EnhancedDocumentLayoutDataset(Dataset):
    # Define default transforms
    DEFAULT_TRANSFORMS = A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.01, contrast_limit=0.01, p=0.5),
            A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=2, val_shift_limit=2, p=0.5),
        ], p=0.5),
        A.ToGray(p=0.5),
        ToTensorV2()
    ])

    def __init__(self,
                 images_dir: str,
                 labels_dir: str,
                 csv_path: str,
                 transform: bool = True,
                 custom_transforms: A.Compose = None,
                 image_size: tuple = (512, 512)):
        """
        Enhanced dataset for document layout analysis.

        Args:
            images_dir (str): Directory containing input images
            labels_dir (str): Directory containing label masks
            csv_path (str): Path to CSV file mapping images to labels
            transform (bool): Whether to apply transformations
            custom_transforms (A.Compose): Optional custom transformation pipeline
            image_size (tuple): Target size for images and masks
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.image_size = image_size
        self.transforms = custom_transforms if custom_transforms else self.DEFAULT_TRANSFORMS

        # Validate paths and data
        self._validate_setup()

    def _validate_setup(self):
        """Validate dataset setup and paths."""
        if self.data.empty:
            raise ValueError(f"CSV file is empty or incorrectly formatted")

        # Check if directories exist
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.labels_dir):
            raise ValueError(f"Labels directory not found: {self.labels_dir}")

        # Validate file existence (first few entries)
        for _, row in self.data.head().iterrows():
            img_path = os.path.join(self.images_dir, row.iloc[0])
            label_path = os.path.join(self.labels_dir, row.iloc[1])
            if not os.path.exists(img_path):
                raise ValueError(f"Image file not found: {img_path}")
            if not os.path.exists(label_path):
                raise ValueError(f"Label file not found: {label_path}")

    def __len__(self) -> int:
        return len(self.data)

    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess image."""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_mask(self, path: str) -> np.ndarray:
        """Load and preprocess mask."""
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to load mask: {path}")
        return mask if len(mask.shape) == 2 else mask[:, :, 0]

    def __getitem__(self, idx: int) -> tuple:
        # Get file paths
        img_name = self.data.iloc[idx, 0]
        label_name = self.data.iloc[idx, 1]

        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, label_name)

        # Load files
        try:
            image = self._load_image(img_path)
            mask = self._load_mask(label_path)
        except Exception as e:
            print(f"Error loading files for index {idx}: {str(e)}")
            # Skip to next item if available
            return self.__getitem__((idx + 1) % len(self)) if len(self) > 1 else None

        # Resize to target size
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # Apply transformations if enabled
        if self.transform:
            try:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            except Exception as e:
                print(f"Warning: Transform failed for index {idx}: {str(e)}")
                # Return non-transformed data if transform fails
                image = torch.from_numpy(image.transpose(2, 0, 1))
                mask = torch.from_numpy(mask)

        return image, mask.long()


if __name__ == "__main__":
    # Example usage
    dataset = EnhancedDocumentLayoutDataset(
        images_dir="data/images",
        labels_dir="data/combined_labels",
        csv_path="data/image_label_mapping.csv",
        image_size=(512, 512)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Test the dataloader
    for batch_idx, (images, masks) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
        break
