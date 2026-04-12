import torch
import torch.nn as nn
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.vgg11 import VGG11Encoder
import os
import gdown
import copy

def copy_weights(old_seq, new_blocks):
    old_layers = list(old_seq.children())

    idx = 0
    for block in new_blocks:
        for layer in block:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data = old_layers[idx].weight.data.clone()
                if layer.bias is not None:
                    layer.bias.data = old_layers[idx].bias.data.clone()
                idx += 1

            elif isinstance(layer, nn.BatchNorm2d):
                # Only copy if old layer is also BatchNorm
                if isinstance(old_layers[idx], nn.BatchNorm2d):
                    layer.weight.data = old_layers[idx].weight.data.clone()
                    layer.bias.data   = old_layers[idx].bias.data.clone()
                idx += 1

            elif isinstance(layer, nn.ReLU):
                idx += 1  # skip ReLU

            elif isinstance(layer, nn.Identity):
                continue
        idx += 1  # skip MaxPool


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds : int   = 37, 
                 seg_classes      : int   = 3, 
                 in_channels      : int   = 3, 
                 classifier_path  : str   = "checkpoints/classifier.pth", 
                 localizer_path   : str   = "checkpoints/localizer.pth",
                 unet_path        : str   = "checkpoints/unet.pth",
                 transfer_learning: str   = "freeze all",
                 use_batchnorm    : bool  = True,
                 dropout          : float = 0.5,
                 train_classifier : float = True,
                 train_localizer  : float = True,
                 train_unet       : float = True,
                 download         : bool  = True 
                 ):
        super().__init__()
        if download:
            os.makedirs("checkpoints", exist_ok=True)
            # gdown.download(id="1gMd4dAtubHVmPUBYVmMm11U1y76GvJWO", output=classifier_path, quiet=False) # will still work
            gdown.download(id="1bxaHWrB6liDdKEpXcUPQfk1wwUX6N_v0", output=classifier_path, quiet=False) # retrained
            # gdown.download(id="10UGWUOCADt1c1pKAnTonsnB9VPehIhB0", output=localizer_path, quiet=False) # will still work
            gdown.download(id="10mUDbxmT3yluEjjeLfUUefCTlQdsLk8H", output=localizer_path, quiet=False) # retrain
            gdown.download(id="1WOTClHYU8N2lHaTWeYoKtyYwOrWRXts3", output=unet_path, quiet=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # classifier
        model_classifier = VGG11Classifier(num_classes = num_breeds, in_channels = in_channels, use_batchnorm = use_batchnorm, dropoout = dropout)
        if classifier_path and (train_classifier == True): 
            checkpoint = torch.load(classifier_path, map_location=device)
            model_classifier.load_state_dict(checkpoint['state_dict'])
            model_classifier.to(device)

            if transfer_learning == "freeze all":
                for param in model_classifier.parameters():
                    param.requires_grad = False

            elif transfer_learning == "partial unfreeze":
                conv_layers = [m for m in model_classifier.conv_layers if isinstance(m, nn.Conv2d)]
                for param in model_classifier.parameters():
                    param.requires_grad = False
                k = 2  
                for layer in conv_layers[-k:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                
            print("classifier loaded!")
        else:
            print("classifer not initialised, random weights assigned")

        # localizer
        model_localizer = VGG11Localizer(copy.deepcopy(model_classifier.conv_layers))
        if localizer_path and (train_localizer == True): 
            checkpoint = torch.load(localizer_path, map_location=device)
            model_localizer.load_state_dict(checkpoint['state_dict'])
            model_localizer.to(device)
            print("localizer loaded!")
        else:
            print("localizer not initialised, random weights assigned")

        # unet
        encoder = VGG11Encoder()
        copy_weights(model_classifier.conv_layers, [
            encoder.block1,
            encoder.block2,
            encoder.block3,
            encoder.block4,
            encoder.block5
        ])
        
        unet = VGG11UNet(num_classes = seg_classes)
        if unet_path and (train_unet == True): 
            checkpoint = torch.load(unet_path, map_location=device)
            unet.load_state_dict(checkpoint['state_dict'])
            unet.to(device)
            print("unet loaded!")
        else:
            print("unet not initialised, random weights assigned")

        if transfer_learning == "freeze all":
            for name, param in unet.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False

        elif transfer_learning == "partial unfreeze":
            for name, param in unet.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False
            for name, param in unet.named_parameters():
                if "encoder.block4" in name or "encoder.block5" in name:
                    param.requires_grad = True

        elif transfer_learning == "unfreeze all":
            for param in unet.parameters():
                param.requires_grad = True

        self.model_classifier = model_classifier
        self.model_localizer  = model_localizer
        self.unet             = unet
    
    def forward(self, x: torch.tensor, conf: bool = False):
        pred_logits = self.model_classifier(x)
        preds = torch.softmax(pred_logits, dim = 1)
        conf_cls, class_id = torch.max(preds, dim=1)
        class_id_pred = torch.argmax(preds) # return 

        bbox_pred = self.model_localizer(x) # return 
        mask_pred = self.unet(x)            # return 
        if not conf:
            return {"classification": pred_logits, "localization": bbox_pred, "segmentation": mask_pred}
        else:
            return {"classification": pred_logits, "localization": bbox_pred, "segmentation": mask_pred, "confidence": conf_cls}