import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(csv_path, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """
    Split dataset into train, validation, and test sets based on given ratios.

    Args:
        csv_path (str): Path to the input CSV file mapping images to labels.
        output_dir (str): Directory where the split CSV files will be saved.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
        random_seed (int): Seed for reproducibility.
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file not found: {csv_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data from CSV
    data = pd.read_csv(csv_path)

    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, but got {total_ratio}")

    # Split into train, validation, and test sets
    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=random_seed)
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(temp_data, test_size=(1 - val_ratio_adjusted), random_state=random_seed)

    # Save splits to CSV files
    train_csv = os.path.join(output_dir, 'train.csv')
    val_csv = os.path.join(output_dir, 'val.csv')
    test_csv = os.path.join(output_dir, 'test.csv')

    train_data.to_csv(train_csv, index=False)
    val_data.to_csv(val_csv, index=False)
    test_data.to_csv(test_csv, index=False)

    print(f"Dataset split completed:\n  Train: {len(train_data)}\n  Validation: {len(val_data)}\n  Test: {len(test_data)}")
    print(f"CSV files saved to: {output_dir}")

if __name__ == "__main__":
    # Example usage
    split_dataset(
        csv_path="data/image_label_mapping.csv",
        output_dir="data/splits",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
