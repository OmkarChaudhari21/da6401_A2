"""VGG11 encoder"""
from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
from .layers import CustomDropout

class VGG11Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                CustomDropout(p=dropout_p)
            )

        self.block1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block3 = nn.Sequential(conv_block(128, 256), conv_block(256, 256))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block4 = nn.Sequential(conv_block(256, 512), conv_block(512, 512))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block5 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        f1 = self.block1(x)
        x = self.pool1(f1)
        
        f2 = self.block2(x)
        x = self.pool2(f2)
        
        f3 = self.block3(x)
        x = self.pool3(f3)
        
        f4 = self.block4(x)
        x = self.pool4(f4)
        
        f5 = self.block5(x)
        out = self.pool5(f5)

        if return_features:
            features = {"skip1": f1, "skip2": f2, "skip3": f3, "skip4": f4, "skip5": f5}
            return out, features
        return out