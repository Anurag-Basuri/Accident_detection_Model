import torch
import torch.nn as nn
import torchvision


class VideoClassifier(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = torchvision.models.video.r3d_18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        # x: B, T, H, W, C -> convert to PyTorch video format: B, C, T, H, W
        x = x.permute(0, 4, 1, 2, 3)
        return self.backbone(x)
