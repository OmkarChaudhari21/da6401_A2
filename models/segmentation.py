import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c + skip_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.size() != skip.size():
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class VGG11UNet(nn.Module):
    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels, dropout_p)

        self.up5 = UpBlock(512, 512, 512)
        self.up4 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64, 64, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, features = self.encoder(x, return_features=True)

        x = self.up5(bottleneck, features["skip5"])
        x = self.up4(x, features["skip4"])
        x = self.up3(x, features["skip3"])
        x = self.up2(x, features["skip2"])
        x = self.up1(x, features["skip1"])

        return self.final(x)