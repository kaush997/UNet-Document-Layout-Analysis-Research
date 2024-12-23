import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import EnhancedUNet
from dataLoader import EnhancedDocumentLayoutDataset

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    def __init__(self, config):
        """
        Initialize the trainer with configuration parameters.

        Args:
            config (dict): Training configuration parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE

        # Initialize model
        self.model = EnhancedUNet(
            input_channels=config['input_channels'],
            output_channels=config['output_channels'],
            img_size=config['image_size'][0]  # Assuming square image
        ).to(self.device)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.1, verbose=True
        )

        # Initialize datasets
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

        # Initialize dataloaders
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

        # Initialize best metrics for model checkpointing
        self.best_val_loss = float('inf')

        # Create checkpoint directory if it doesn't exist
        os.makedirs(config['checkpoint_dir'], exist_ok=True)

    def validate_targets(self, masks):
        """Ensure that target values are within the valid range."""
        if torch.max(masks) >= self.config['output_channels']:
            raise ValueError(f"Target {torch.max(masks).item()} is out of bounds for output_channels={self.config['output_channels']}")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        processed_batches = 0

        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch_idx, (images, masks) in enumerate(pbar):
                try:
                    # Move data to device
                    images = images.to(self.device, dtype=torch.float32)
                    masks = masks.to(self.device, dtype=torch.long)

                    # Validate targets
                    self.validate_targets(masks)

                    # Zero gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()

                    # Update metrics
                    total_loss += loss.item()
                    processed_batches += 1

                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item()})

                    # Log to wandb if enabled
                    if self.use_wandb:
                        wandb.log({
                            'batch_loss': loss.item(),
                            'learning_rate': self.optimizer.param_groups[0]['lr']
                        })
                except RuntimeError as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue

        return total_loss / processed_batches if processed_batches > 0 else float('inf')

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        processed_batches = 0

        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation") as pbar:
                for images, masks in pbar:
                    try:
                        images = images.to(self.device, dtype=torch.float32)
                        masks = masks.to(self.device, dtype=torch.long)

                        # Validate targets
                        self.validate_targets(masks)

                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)

                        total_loss += loss.item()
                        processed_batches += 1
                        pbar.set_postfix({'loss': loss.item()})
                    except RuntimeError as e:
                        print(f"Error in validation: {e}")
                        continue

        return total_loss / processed_batches if processed_batches > 0 else float('inf')

    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint."""
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
        """Main training loop."""
        print(f"Training on device: {self.device}")
        print(f"Input channels: {self.config['input_channels']}")
        print(f"Output channels: {self.config['output_channels']}")
        print(f"Image size: {self.config['image_size']}")

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")

            try:
                # Training phase
                train_loss = self.train_epoch()

                # Validation phase
                val_loss = self.validate()

                # Update learning rate scheduler
                self.scheduler.step(val_loss)

                # Log metrics
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    })

                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")

                # Save checkpoint
                self.save_checkpoint(epoch, val_loss)

            except Exception as e:
                print(f"Error in epoch {epoch + 1}: {e}")
                continue


if __name__ == "__main__":
    # Training configuration
    config = {
        'input_channels': 3,  # RGB images
        'output_channels': 26,  # Number of layout classes
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
        'use_wandb': False,  # Set to False by default to avoid authentication issues
    }

    # Initialize wandb if enabled
    if config['use_wandb'] and WANDB_AVAILABLE:
        try:
            wandb.login()  # This will prompt for API key if not already logged in
            wandb.init(
                project="document-layout-analysis",
                config=config
            )
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            print("Training will continue without wandb logging")
            config['use_wandb'] = False

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()
