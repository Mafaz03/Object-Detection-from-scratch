import torch
import torch.nn as nn
from torch.utils.data import Dataset
from models.layers import CustomDropout

class VGG11Classifier(nn.Module):
    def __init__(self, in_channels, num_classes = 37, dropoout = 0.5, use_batchnorm=True):
        super(VGG11Classifier, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropoout = dropoout
        self.use_batchnorm = use_batchnorm

        def BN(channels):
            return nn.BatchNorm2d(channels) if self.use_batchnorm else nn.Identity()
 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            BN(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            BN(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
 
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            BN(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            BN(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
 
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            BN(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            BN(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            BN(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            BN(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
 
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            CustomDropout(p=self.dropoout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            CustomDropout(p=self.dropoout),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )
 
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x