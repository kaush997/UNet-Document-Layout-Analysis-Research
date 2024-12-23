import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from timm.models.vision_transformer import VisionTransformer


class EnhancedUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=26, embed_dim=64, img_size=512):
        super(EnhancedUNet, self).__init__()

        # Vision Transformer block - now accepts configurable image size
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=16,
            in_chans=input_channels,
            embed_dim=embed_dim,
            depth=6,
            num_heads=8,
            mlp_ratio=4.,
            num_classes=embed_dim
        )

        # U-Net Down path
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # U-Net Up path with adjusted dimensions
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(96, 64)

        # Final output
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Store input size for later use
        input_size = (x.size(2), x.size(3))

        # Apply Vision Transformer
        vit_out = self.vit(x)

        # Reshape and interpolate ViT output
        batch_size = x.size(0)
        vit_out = vit_out.view(batch_size, -1, 1, 1)
        vit_out = nn.functional.interpolate(vit_out, size=input_size, mode='bilinear', align_corners=True)

        # U-Net Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # U-Net Decoder with skip connections
        d4 = self.upconv4(e4)
        d4 = torch.cat((d4, e3), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)

        # Ensure ViT output has correct number of channels
        vit_feature = nn.functional.interpolate(vit_out, size=(d1.size(2), d1.size(3)),
                                              mode='bilinear', align_corners=True)

        d1 = torch.cat((d1, vit_feature), dim=1)
        d1 = self.dec1(d1)

        # Final layer
        out = self.final(d1)

        # Resize output to match input size if necessary
        if out.size(2) != input_size[0] or out.size(3) != input_size[1]:
            out = nn.functional.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        return out

    @staticmethod
    def pool(x):
        return nn.MaxPool2d(kernel_size=2, stride=2)(x)

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


if __name__ == "__main__":
    # Test the model
    model = EnhancedUNet(input_channels=3, img_size=512)
    sample_input = torch.rand((1, 3, 512, 512))
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
