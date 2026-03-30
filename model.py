import torch
import torch.nn as nn
from torch.utils.data import Dataset



 
class CustomDropout_cnn_layers(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
 
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        N, C, H, W = x.shape
        mask = (torch.rand(N, C, 1, 1, device=x.device) > self.p).to(x.dtype)
        mask = mask / ((1 - self.p) + 1e-8)
        return x * mask
 
 
class CustomDropout_linear_layers(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
 
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand(x.shape, device=x.device) > self.p).to(x.dtype)
        mask = mask / ((1 - self.p) + 1e-8)
        return x * mask
 
class VGG11(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
 
        # FIX: Added BatchNorm after every Conv2d
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
 
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
 
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
 
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            CustomDropout_linear_layers(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            CustomDropout_linear_layers(p=0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )
 
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
 