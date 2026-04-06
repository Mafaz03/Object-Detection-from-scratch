import torch
from torch import nn

class VGG11Localizer(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.backbone(x)
        bbox = self.regressor(x)
        return bbox