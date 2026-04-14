import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder

class VGG11Localizer(nn.Module):
    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels, dropout_p)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 4),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)