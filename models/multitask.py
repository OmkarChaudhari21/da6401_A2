import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, 
                 classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True),
            nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Linear(4096, num_breeds)
        )

        self.localizer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024), nn.ReLU(True),
            nn.Linear(1024, 4), nn.Sigmoid()
        )

        self.segmenter = VGG11UNet(seg_classes, in_channels)

        self.load_weights(classifier_path, localizer_path, unet_path)

    def load_weights(self, class_path, loc_path, unet_path):
        for path in [class_path, loc_path, unet_path]:
            try:
                state = torch.load(path, map_location="cpu")
                self.load_state_dict(state, strict=False)
                print(f"Loaded {path}")
            except:
                print(f"Failed to load {path}")

    def forward(self, x: torch.Tensor):
        bottleneck, _ = self.encoder(x, return_features=True)

        flat = torch.flatten(self.avgpool(bottleneck), 1)

        return {
            "classification": self.classifier(flat),
            "localization": self.localizer(flat),
            "segmentation": self.segmenter(x)
        }