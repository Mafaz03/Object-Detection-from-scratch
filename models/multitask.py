import torch
import torch.nn as nn
from classification import VGG11Classifier
from localization import VGG11Localizer
from segmentation import VGG11UNet
from vgg11 import VGG11Encoder

import copy

def copy_weights(old_seq, new_blocks):
    old_layers = list(old_seq.children())

    idx = 0
    for block in new_blocks:
        for layer in block:
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                layer.weight.data = old_layers[idx].weight.data.clone()
                layer.bias.data   = old_layers[idx].bias.data.clone()
                idx += 1
            elif isinstance(layer, nn.ReLU):
                idx += 1  # skip ReLU
        idx += 1  # skip MaxPool


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # classifier
        model_classifier = VGG11Classifier(num_classes=37, in_channels=3)
        checkpoint = torch.load(classifier_path, map_location=device)
        model_classifier.load_state_dict(checkpoint['state_dict'])
        model_classifier.to(device)
        print("classifier loaded!")

        # localizer
        model_localizer = VGG11Localizer(copy.deepcopy(model_classifier.conv_layers))
        checkpoint = torch.load(localizer_path, map_location=device)
        model_localizer.load_state_dict(checkpoint['state_dict'])
        model_localizer.to(device)
        print("localizer loaded!")

        # unet
        encoder = VGG11Encoder()
        copy_weights(model_classifier.conv_layers, [
            encoder.block1,
            encoder.block2,
            encoder.block3,
            encoder.block4,
            encoder.block5
        ])
        unet = VGG11UNet(num_classes = 1)
        checkpoint = torch.load(unet_path, map_location=device)
        unet.load_state_dict(checkpoint['state_dict'])
        unet.to(device)
        print("unet loaded!")

        self.model_classifier = model_classifier
        self.model_localizer  = model_localizer
        self.unet             = unet
    
    def forward(self, x: torch.tensor):
        pred_logits = self.model_classifier(x)
        preds = torch.softmax(pred_logits, dim = 1)
        class_id_pred = torch.argmax(preds) # return 

        bbox_pred = self.model_localizer(x) # return 
        mask_pred = self.unet(x)            # return 

        return class_id_pred, bbox_pred, mask_pred