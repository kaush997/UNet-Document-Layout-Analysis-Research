# model.py
import torch
import torch.nn as nn
from transformers import ViTModel

class EnhancedUNet(nn.Module):
    def __init__(self, input_channels, output_channels, img_size, vit_embedding_dim=768):
        super(EnhancedUNet, self).__init__()

        # Load Vision Transformer (ViT) for global feature extraction
        try:
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        except Exception as e:
            print(f"Error loading ViT model: {e}")
            raise

        self.vit.conv_proj = nn.Conv2d(input_channels, self.vit.config.hidden_size, kernel_size=1)

        self.encoder1 = self._conv_block(input_channels, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.encoder4 = self._conv_block(256, 512)

        self.bottleneck = self._conv_block(512 + vit_embedding_dim, 1024)

        self.decoder4 = self._conv_block(1024, 512)
        self.decoder3 = self._conv_block(512, 256)
        self.decoder2 = self._conv_block(256, 128)
        self.decoder1 = self._conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Extract global features using ViT
        vit_features = self.vit(pixel_values=x)['last_hidden_state']
        batch_size, seq_len, embed_dim = vit_features.shape
        vit_features = vit_features.view(batch_size, 14, 14, embed_dim).permute(0, 3, 1, 2)

        # U-Net forward pass
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(2)(enc3))

        bottleneck = self.bottleneck(torch.cat([enc4, vit_features], dim=1))

        dec4 = self.decoder4(torch.cat([nn.Upsample(scale_factor=2)(bottleneck), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([nn.Upsample(scale_factor=2)(dec4), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([nn.Upsample(scale_factor=2)(dec3), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([nn.Upsample(scale_factor=2)(dec2), enc1], dim=1))

        return self.final_conv(dec1)
