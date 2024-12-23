import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import EnhancedUNet
from dataLoader import EnhancedDocumentLayoutDataset, split_data

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = EnhancedUNet(
            input_channels=config['input_channels'],
            output_channels=config['output_channels'],
            img_size=config['image_size'][0]
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])

        train_pairs, val_pairs, test_pairs = split_data(
            images_dir=config['images_dir'],
            labels_dir=config['labels_dir'],
            mapping_csv=config['mapping_csv']
        )

        self.train_dataset = EnhancedDocumentLayoutDataset(
            images_dir=config['images_dir'],
            labels_dir=config['labels_dir'],
            image_label_pairs=train_pairs,
            image_size=config['image_size']
        )

        self.val_dataset = EnhancedDocumentLayoutDataset(
            images_dir=config['images_dir'],
            labels_dir=config['labels_dir'],
            image_label_pairs=val_pairs,
            image_size=config['image_size']
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config['batch_size'], shuffle=False)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for images, masks in tqdm(self.train_loader, desc="Training"):
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

if __name__ == "__main__":
    config = {
        'input_channels': 3,
        'output_channels': 26,
        'image_size': (512, 512),
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'images_dir': 'data/images',
        'labels_dir': 'data/combined_labels',
        'mapping_csv': 'data/image_label_mapping.csv',
    }
    trainer = Trainer(config)
    trainer.train()
