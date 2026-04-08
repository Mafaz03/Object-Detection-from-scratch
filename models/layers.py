import torch.nn as nn
import torch

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        # p is probl of weights getting 0-ed
        # p = 1, all 0
        # p = 0, all non 0
        
        super().__init__()
        self.p = p
 
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand(x.shape, device=x.device) > self.p).to(x.dtype)
        mask = mask / ((1 - self.p) + 1e-8)
        return x * mask
 