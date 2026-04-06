import torch
from torch import nn
from models.vgg11 import VGG11Encoder

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # feature fusion
        x = self.conv(x)
        return x
    

class VGG11UNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.encoder = VGG11Encoder()

        self.up1 = UpBlock(512, 512, 512)  # 7   -> 14
        self.up2 = UpBlock(512, 512, 256)  # 14  -> 28
        self.up3 = UpBlock(256, 256, 128)  # 28  -> 56
        self.up4 = UpBlock(128, 128, 64)   # 56  -> 112
        self.up5 = UpBlock(64, 64, 64)     # 112 -> 224

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x6, feature_dict = self.encoder(x, return_features = True)
        x1 = feature_dict["x1"]
        x2 = feature_dict["x2"]
        x3 = feature_dict["x3"]
        x4 = feature_dict["x4"]
        x5 = feature_dict["x5"]

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        return self.final(x)
    