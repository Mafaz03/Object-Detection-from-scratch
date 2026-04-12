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

    def forward(self, x, old = False):
        if old:
            x = self.backbone(x)
            bbox = self.regressor(x)
            return bbox
        else:
            x1 = bbox[:, 0]
            y1 = bbox[:, 1]
            x2 = bbox[:, 2]
            y2 = bbox[:, 3]
    
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w  = (x2 - x1).abs()
            h  = (y2 - y1).abs()
    
            bbox = torch.stack([cx, cy, w, h], dim=1)
            return bbox