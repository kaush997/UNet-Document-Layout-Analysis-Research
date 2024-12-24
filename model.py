### model.py
import torch
import torch.nn as nn
from torchvision.models import resnet34

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class EnhancedUNet(nn.Module):
    def __init__(self, input_channels, output_channels, img_size):
        super(EnhancedUNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.encoder = resnet34(pretrained=True)
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(input_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec1 = self.up1(bottleneck)
        dec1 = self.dec1(torch.cat((dec1, enc4), dim=1))
        dec2 = self.up2(dec1)
        dec2 = self.dec2(torch.cat((dec2, enc3), dim=1))
        dec3 = self.up3(dec2)
        dec3 = self.dec3(torch.cat((dec3, enc2), dim=1))
        dec4 = self.up4(dec3)
        dec4 = self.dec4(torch.cat((dec4, enc1), dim=1))

        return self.final_conv(dec4)