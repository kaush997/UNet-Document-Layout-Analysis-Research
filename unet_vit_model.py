import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(*[self.norm1(x)] * 3)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Add a reduction layer before concatenation
        self.reduce = nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=1)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust x1 size to match x2 if needed
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x2, x1], dim=1)
        x = self.reduce(x)
        return self.conv(x)


class UNetViT(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, img_size=256):
        super().__init__()

        # Calculate number of patches
        self.patch_size = 16
        self.num_patches = (img_size // 2 // self.patch_size) ** 2
        self.embed_dim = 768

        # Encoder
        self.inc = ConvBlock(in_channels, 64)
        self.down1 = ConvBlock(64, 128)

        # Patch Embedding and ViT
        self.patch_embed = PatchEmbedding(patch_size=self.patch_size, in_channels=128, embed_dim=self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        self.vit_blocks = nn.ModuleList([ViTBlock(self.embed_dim) for _ in range(6)])

        # Decoder
        self.up1 = DecoderBlock(self.embed_dim, 128, 256)  # Modified channels
        self.up2 = DecoderBlock(256, 64, 128)  # Modified channels
        self.outc = nn.Conv2d(128, out_channels, 1)

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        print(f"Input shape: {x.shape}")

        # Encoder path
        x1 = self.inc(x)
        print(f"After inc shape: {x1.shape}")

        x2 = self.down1(F.max_pool2d(x1, 2))
        print(f"After down1 shape: {x2.shape}")

        # ViT path
        vit_in = self.patch_embed(x2)
        print(f"After patch_embed shape: {vit_in.shape}")

        vit_in = vit_in + self.pos_embed
        vit_out = vit_in

        for block in self.vit_blocks:
            vit_out = block(vit_out)

        # Reshape back to spatial dimensions
        B, N, C = vit_out.shape
        H = W = int(N ** 0.5)
        vit_out = vit_out.transpose(1, 2).reshape(B, C, H, W)
        print(f"After ViT reshape shape: {vit_out.shape}")

        # Decoder path
        x = self.up1(vit_out, x2)
        print(f"After up1 shape: {x.shape}")

        x = self.up2(x, x1)
        print(f"After up2 shape: {x.shape}")

        x = self.outc(x)
        print(f"Output shape: {x.shape}")

        return x


# Example usage
if __name__ == "__main__":
    # Print dimensions for debugging
    img_size = 256
    patch_size = 16
    num_patches = (img_size // 2 // patch_size) ** 2
    print(f"Number of patches: {num_patches}")

    model = UNetViT(img_size=img_size)
    x = torch.randn(1, 3, img_size, img_size)
    output = model(x)