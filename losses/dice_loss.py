import torch
from torch import nn

class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2. * intersection + 1e-6) / (union + 1e-6)

        return 1 - dice