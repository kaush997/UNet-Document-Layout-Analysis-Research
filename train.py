### train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import EnhancedUNet
from dataLoader import EnhancedDocumentLayoutDataset

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = EnhancedUNet(
            input_channels=config['input_channels'],
            output_channels=config['output_channels'],
            img_size=config['image_size'][0]  # Assuming square image
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.1, verbose=True
        )

        self.train_dataset = EnhancedDocumentLayoutDataset(
            images_dir=config['images_dir'],
            labels_dir=config['labels_dir'],
            csv_path=config['train_csv'],
            image_size=config['image_size']
        )

        self.val_dataset = EnhancedDocumentLayoutDataset(
            images_dir=config['images_dir'],
            labels_dir=config['labels_dir'],
            csv_path=config['val_csv'],
            image_size=config['image_size']
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        self.best_val_loss = float('inf')
        os.makedirs(config['checkpoint_dir'], exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        processed_batches = 0

        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                # Ensure target is in the range [0, num_classes - 1]
                loss = self.criterion(outputs, masks)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                processed_batches += 1

                pbar.set_postfix({'loss': loss.item()})

        return total_loss / processed_batches if processed_batches > 0 else float('inf')

    def validate(self):
        self.model.eval()
        total_loss = 0
        processed_batches = 0

        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation") as pbar:
                for images, masks in pbar:
                    images = images.to(self.device, dtype=torch.float32)
                    masks = masks.to(self.device, dtype=torch.long)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                    total_loss += loss.item()
                    processed_batches += 1

                    pbar.set_postfix({'loss': loss.item()})

        return total_loss / processed_batches if processed_batches > 0 else float('inf')

    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }

        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f"checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = os.path.join(
                self.config['checkpoint_dir'],
                "best_model.pth"
            )
            torch.save(checkpoint, best_model_path)

    def train(self):
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            self.save_checkpoint(epoch, val_loss)

if __name__ == "__main__":
    config = {
        'input_channels': 3,
        'output_channels': 26,
        'image_size': (512, 512),
        'batch_size': 8,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'images_dir': 'data/images',
        'labels_dir': 'data/combined_labels',
        'train_csv': 'data/train.csv',
        'val_csv': 'data/val.csv',
        'checkpoint_dir': 'checkpoints',
    }

    trainer = Trainer(config)
    trainer.train()
