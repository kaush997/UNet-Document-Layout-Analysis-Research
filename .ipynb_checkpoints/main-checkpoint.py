import os
from src.data_preprocessing import DataPreprocessor
from src.data_augmentation import DataAugmentor
from src.data_splitting import DataSplitter
from src.data_loader import DocumentLayoutDataset, DataLoader
from torchvision import transforms

def main():
    # Preprocess data
    input_dir = "data/project-7-at-2024-11-28-20-06-811bd479"
    output_dir = "data/processed_data"
    preprocessor = DataPreprocessor(input_dir, output_dir)
    preprocessor.preprocess_all()

    # Augment data
    augmentor = DataAugmentor(os.path.join(output_dir, "train/images"), os.path.join(output_dir, "train/images"))
    augmentor.augment_all()

    # Split data
    splitter = DataSplitter(output_dir, output_dir)
    splitter.split_data()

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DocumentLayoutDataset(
        image_dir=os.path.join(output_dir, "train/images"),
        mask_dir=os.path.join(output_dir, "train/labels"),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    for images, masks in train_loader:
        print(images.shape, masks.shape)

if __name__ == "__main__":
    main()