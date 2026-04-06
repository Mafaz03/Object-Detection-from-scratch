import argparse

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, random_split, DataLoader, dataset

import pathlib
import matplotlib.patches as patches
from data.pets_dataset import OxfordIIITPetDataset, get_class_map

from models import MultiTaskPerceptionModel
from losses import DiceLoss, IoULoss

bce = nn.BCEWithLogitsLoss()

def unet_loss_fn(pred, target):
    return bce(pred, target) + DiceLoss()(pred, target)

def bbox_loss_fn(pred, target):
    iou_loss = IoULoss()
    return iou_loss(pred, target)

parser = argparse.ArgumentParser(description="Inferencing Object detection", conflict_handler="resolve")


parser.add_argument('-n_breeds', "--num_breeds", type=int, default=37, help="Total number of breeds to classify")
parser.add_argument('-s_class', "--seg_classes", type=int, default=3, help="Total number of classes to segment")
parser.add_argument('-in_c', "--in_channels", type=int, default=3, help="In channels of images")
parser.add_argument('-c_path', "--classifier_path", type=str, default="checkpoints/classifier.pth", help="Path for classifier model (.pth)")
parser.add_argument('-l_path', "--localizer_path", type=str, default="checkpoints/localizer.pth", help="Path for localizer model (.pth)")
parser.add_argument('-u_path', "--unet_path", type=str, default="checkpoints/unet.pth", help="Path for unet model (.pth)")
parser.add_argument('-d_path', "--dataset_path", type=str, default="oxford-iiit-pet", help="Path for dataset model")

args = parser.parse_args()

multitask_model = MultiTaskPerceptionModel(num_breeds      = args.num_breeds, 
                                           seg_classes     = args.seg_classes, 
                                           in_channels     = args.in_channels, 
                                           classifier_path = args.classifier_path,
                                           localizer_path  = args.localizer_path,
                                           unet_path       = args.unet_path)
multitask_model.eval()
print("read to go")

train_ratio = 0.8
BATCH_SIZE = 4
mappings = get_class_map(pathlib.Path(args.dataset_path))

dataset = OxfordIIITPetDataset(root_dir = args.dataset_path)

train_ds, test_ds = random_split(dataset, [int(train_ratio * len(dataset)), len(dataset)-int(train_ratio * len(dataset))])

train_dl = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size = BATCH_SIZE, shuffle=False)


sample = next(iter(train_dl))

sample_image   = sample['image']
sample_classid = sample['class_id']
sample_bbox    = sample['bbox']
sample_mask    = sample['mask']


predictions = multitask_model(sample_image)

class_pred = torch.argmax((torch.softmax(predictions["classification"], dim = 1)), dim = 1)
bbox_pred  = predictions['localization']
mask_pred  = predictions['segmentation']

print(f"Mask generation / Unet loss:         {unet_loss_fn(mask_pred, sample_mask).item():.3f}")
print(f"Bbox generation / localization loss: {bbox_loss_fn(predictions['localization'], sample_bbox).item():.3f}")

# processing for plotting
mean = torch.tensor([0.485, 0.456, 0.406], device=sample_image.device).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225], device=sample_image.device).view(3, 1, 1)
sample_image = sample_image * std + mean

fig, ax = plt.subplots(3, BATCH_SIZE, figsize = (8*1.3, 7*1.3))
for idx in range(BATCH_SIZE):
    ax[0][idx].imshow(sample_image[idx].permute(1,2,0).detach().cpu().numpy()) 
    ax[0][idx].set_title("")  

    ax[0][idx].text(0.5, 1.08, f"Actual: {mappings[sample_classid[idx].item()]}",
                transform=ax[0][idx].transAxes,
                ha='center', va='bottom',
                color='green', fontsize=10, fontweight='bold')

    ax[0][idx].text(0.5, 1.01, f"Pred: {mappings[class_pred[idx].item()]}",
                transform=ax[0][idx].transAxes,
                ha='center', va='bottom',
                color='red', fontsize=10, fontweight='bold')

    # Predicted bbox (RED) - convert center -> top-left
    xc, yc, w, h = bbox_pred[idx].squeeze().detach().cpu().numpy()
    _, _, H, W = sample_image.shape

    x = (xc - w / 2) * W
    y = (yc - h / 2) * H
    w = w * W
    h = h * H

    rect_pred = patches.Rectangle(
        (x, y), w, h,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax[0][idx].add_patch(rect_pred)

    # Actual bbox (YELLOW)
    xc, yc, w_gt, h_gt = sample_bbox[idx].cpu().numpy()
    x_gt = (xc - w_gt / 2) * W
    y_gt = (yc - h_gt / 2) * H
    w_gt = w_gt * W
    h_gt = h_gt * H

    rect_gt = patches.Rectangle((x_gt, y_gt), w_gt, h_gt,
                                  linewidth=2, edgecolor='Yellow', facecolor='none')
    ax[0][idx].add_patch(rect_gt)  

    ax[1][idx].imshow(sample_mask[idx].permute(1,2,0).detach().cpu().numpy(), cmap = "gray") 
    ax[1][idx].imshow(sample_mask[idx].permute(1,2,0).detach().cpu().numpy(), cmap = "gray") 
    ax[1][idx].set_title("Actual Mask")

    ax[2][idx].imshow(mask_pred[idx].permute(1,2,0).detach().cpu().numpy(), cmap = "gray") 
    ax[2][idx].imshow(mask_pred[idx].permute(1,2,0).detach().cpu().numpy(), cmap = "gray") 
    ax[2][idx].set_title("Pred Mask")

    ax[0][idx].axis("off")
    ax[1][idx].axis("off")
    ax[2][idx].axis("off")

plt.tight_layout()
plt.show()