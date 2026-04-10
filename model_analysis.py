import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import Dataset, random_split, DataLoader, dataset
import pathlib
import matplotlib.patches as patches
from data.pets_dataset import OxfordIIITPetDataset, get_class_map
from models import MultiTaskPerceptionModel
from losses import DiceLoss, IoULoss
from models import VGG11Classifier
import wandb
wandb.init(project="Multitask-Pet-Detection")


#### Task 2.1 ####
mappings = get_class_map(pathlib.Path("oxford-iiit-pet"))
dataset = OxfordIIITPetDataset(root_dir = "oxford-iiit-pet")
train_ds, test_ds = random_split(dataset, [int(0.8 * len(dataset)), len(dataset)-int(0.8 * len(dataset))])
train_dl = DataLoader(train_ds, batch_size = 1, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size = 1, shuffle=True)


model_classifier = VGG11Classifier(in_channels = 3, num_classes = 37, use_batchnorm = True)
checkpoint = torch.load("/Users/mohamedmafaz/Downloads/2_1_batchnorm.pth", map_location=torch.device('cpu'))
model_classifier.load_state_dict(checkpoint['state_dict'])
model_classifier.eval()

sample = next(iter(train_dl))
sample_image = sample["image"]

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

hooks = []

for name, layer in model_classifier.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        h = layer.register_forward_hook(get_activation(name))
        hooks.append(h)


with torch.no_grad():
    _ = model_classifier(sample_image)

for h in hooks:
    h.remove()


act = activations["conv_layers.8"]
vals = act.detach().flatten().cpu().numpy()


hist, bin_edges = np.histogram(vals)#, bins=50)
bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
counts = hist.tolist()

table = wandb.Table(data=[[x, y] for x, y in zip(bin_centers, counts)],
                    columns=["activation_value", "count"])

wandb.log({
    "conv_layers_8_activations": wandb.plot.bar(
        table,
        "activation_value",
        "count",
        title="3rd Conv Activation Distribution - BATCHNORM "
    )
})

wandb.finish()