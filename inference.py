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

import wandb

import pandas as pd

bce = nn.BCEWithLogitsLoss()

def unet_loss_fn(pred, target):
    return bce(pred, target) + DiceLoss()(pred, target)

def bbox_loss_fn(pred, target):
    iou_loss = IoULoss()
    return iou_loss(pred, target)

def compute_macro_f1(pred_classes, true_classes, num_classes):
    f1_scores = []
    for c in range(num_classes):
        tp = ((pred_classes == c) & (true_classes == c)).sum().item()
        fp = ((pred_classes == c) & (true_classes != c)).sum().item()
        fn = ((pred_classes != c) & (true_classes == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def box_iou(pred_boxes, gt_boxes):
    def to_xyxy(boxes):
        xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return torch.stack([xc - w/2, yc - h/2, xc + w/2, yc + h/2], dim=1)

    pred_xyxy = to_xyxy(pred_boxes)
    gt_xyxy   = to_xyxy(gt_boxes)

    inter_x1 = torch.max(pred_xyxy[:, 0], gt_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], gt_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], gt_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], gt_xyxy[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    pred_area  = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    gt_area    = (gt_xyxy[:, 2]   - gt_xyxy[:, 0])   * (gt_xyxy[:, 3]   - gt_xyxy[:, 1])
    union_area = pred_area + gt_area - inter_area

    return inter_area / union_area.clamp(min=1e-6)


def compute_map(pred_boxes, gt_boxes, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 1.00, 0.05)

    ious = box_iou(pred_boxes.detach(), gt_boxes.detach()).cpu().numpy()
    ap_per_thresh = [float(np.mean(ious >= t)) for t in iou_thresholds]
    return float(np.mean(ap_per_thresh))


def compute_dice(pred_logits, gt_masks, threshold=0.5):
    pred_probs = torch.sigmoid(pred_logits)
    pred_bin   = (pred_probs >= threshold).float()
    gt_bin     = (gt_masks   >= threshold).float()

    dims = (1, 2, 3)   # reduce over C, H, W per sample
    intersection    = (pred_bin * gt_bin).sum(dim=dims)
    dice_per_sample = (2 * intersection) / (pred_bin.sum(dim=dims) + gt_bin.sum(dim=dims) + 1e-6)
    return float(dice_per_sample.mean().item())


parser = argparse.ArgumentParser(description="multi task pet inference", conflict_handler="resolve")

parser.add_argument('-n_breeds', "--num_breeds",       type=int, default=37,                           help="Total number of breeds to classify")
parser.add_argument('-s_class',  "--seg_classes",      type=int, default=3,                            help="Total number of classes to segment")
parser.add_argument('-in_c',     "--in_channels",      type=int, default=3,                            help="In channels of images")
parser.add_argument('-c_path',   "--classifier_path",  type=str, default="checkpoints/classifier.pth", help="Path for classifier model (.pth)")
parser.add_argument('-l_path',   "--localizer_path",   type=str, default="checkpoints/localizer.pth",  help="Path for localizer model (.pth)")
parser.add_argument('-u_path',   "--unet_path",        type=str, default="checkpoints/unet.pth",       help="Path for unet model (.pth)")
parser.add_argument('-d_path',   "--dataset_path",     type=str, default="oxford-iiit-pet",            help="Path for dataset model")

args = parser.parse_args()

wandb.init(project="Multitask-Pet-Detection", config=vars(args))

multitask_model = MultiTaskPerceptionModel(num_breeds      = args.num_breeds,
                                           seg_classes     = args.seg_classes,
                                           in_channels     = args.in_channels,
                                           classifier_path = args.classifier_path,
                                           localizer_path  = args.localizer_path,
                                           unet_path       = args.unet_path,
                                           train_classifier= True,
                                           train_localizer = True,
                                           train_unet      = True,
                                           download        = False)
multitask_model.eval()
print("read to go")

train_ratio = 0.8
BATCH_SIZE  = 4
mappings    = get_class_map(pathlib.Path(args.dataset_path))

dataset = OxfordIIITPetDataset(root_dir=args.dataset_path)

train_ds, test_ds = random_split(dataset, [int(train_ratio * len(dataset)),
                                            len(dataset) - int(train_ratio * len(dataset))])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

sample = next(iter(train_dl))

sample_image   = sample['image']
sample_classid = sample['class_id']
sample_bbox    = sample['bbox']
sample_mask    = sample['mask']

predictions = multitask_model(sample_image)

class_pred = torch.argmax(torch.softmax(predictions["classification"], dim=1), dim=1)
bbox_pred  = predictions['localization']
mask_pred  = predictions['segmentation']

classifier_loss_fn = nn.CrossEntropyLoss()

cls_loss  = classifier_loss_fn(predictions['classification'], sample_classid).item()
bbox_loss = bbox_loss_fn(predictions['localization'], sample_bbox).item()
unet_loss = unet_loss_fn(mask_pred, sample_mask).item()

# Compute metrics 
macro_f1  = compute_macro_f1(class_pred, sample_classid, num_classes=args.num_breeds)
map_score = compute_map(bbox_pred, sample_bbox)
dice      = compute_dice(mask_pred, sample_mask)

print(f"classification loss:                 {cls_loss:.3f}")
print(f"Bbox generation / localization loss: {bbox_loss:.3f}")
print(f"Mask generation / Unet loss:         {unet_loss:.3f}")
print(f"Macro F1-Score  (classification):    {macro_f1:.3f}")
print(f"mAP@0.50:0.95   (localization):      {map_score:.3f}")
print(f"Dice Score      (segmentation):      {dice:.3f}")

# wandb: scalars
wandb.log({
    "classification_loss": cls_loss,
    "bbox_loss":           bbox_loss,
    "unet_loss":           unet_loss,
    "macro_f1":            macro_f1,
    "mAP_50_95":           map_score,
    "dice_score":          dice,
})

# wandb: loss bar chart (must be its own log call)
loss_table = wandb.Table(data=[
    ["classification", cls_loss],
    ["bbox",           bbox_loss],
    ["unet",           unet_loss],
], columns=["loss_type", "value"])
wandb.log({"loss_bar": wandb.plot.bar(loss_table, "loss_type", "value", title="Loss Comparison")})

# wandb: metrics bar chart
metrics_table = wandb.Table(data=[
    ["Macro F1",   macro_f1],
    ["mAP@50:95",  map_score],
    ["Dice Score", dice],
], columns=["metric", "value"])
wandb.log({"metrics_bar": wandb.plot.bar(metrics_table, "metric", "value", title="Metrics")})

# matplotlib grid
mean = torch.tensor([0.485, 0.456, 0.406], device=sample_image.device).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225], device=sample_image.device).view(3, 1, 1)
sample_image = sample_image * std + mean
sample_image = torch.clamp(sample_image, 0.0, 1.0)

fig, ax = plt.subplots(3, BATCH_SIZE, figsize=(8*1.3, 7*1.3))
for idx in range(BATCH_SIZE):
    img_np = sample_image[idx].permute(1,2,0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    ax[0][idx].imshow(img_np)
    ax[0][idx].set_title("")

    ax[0][idx].text(0.5, 1.08, f"Actual: {mappings[sample_classid[idx].item()]}",
                    transform=ax[0][idx].transAxes, ha='center', va='bottom',
                    color='green', fontsize=10, fontweight='bold')
    ax[0][idx].text(0.5, 1.01, f"Pred: {mappings[class_pred[idx].item()]}",
                    transform=ax[0][idx].transAxes, ha='center', va='bottom',
                    color='red', fontsize=10, fontweight='bold')

    # Predicted bbox (RED)
    xc, yc, w, h = bbox_pred[idx].squeeze().detach().cpu().numpy()
    _, _, H, W = sample_image.shape
    rect_pred = patches.Rectangle(
        ((xc - w/2)*W, (yc - h/2)*H), w*W, h*H,
        linewidth=2, edgecolor='r', facecolor='none')
    ax[0][idx].add_patch(rect_pred)

    # Actual bbox (YELLOW)
    xc, yc, w_gt, h_gt = sample_bbox[idx].cpu().numpy()
    rect_gt = patches.Rectangle(
        ((xc - w_gt/2)*W, (yc - h_gt/2)*H), w_gt*W, h_gt*H,
        linewidth=2, edgecolor='Yellow', facecolor='none')
    ax[0][idx].add_patch(rect_gt)

    ax[1][idx].imshow(sample_mask[idx].permute(1,2,0).detach().cpu().numpy(), cmap="gray")
    ax[1][idx].set_title("Actual Mask")

    pred_mask_vis = torch.sigmoid(mask_pred[idx]).permute(1,2,0).detach().cpu().numpy()
    ax[2][idx].imshow(pred_mask_vis, cmap="gray")
    ax[2][idx].set_title("Pred Mask")

    ax[0][idx].axis("off")
    ax[1][idx].axis("off")
    ax[2][idx].axis("off")

plt.tight_layout()

# wandb: per-sample combined images
from PIL import Image
for idx in range(BATCH_SIZE):
    img = sample_image[idx].permute(1,2,0).detach().cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    pred_cls = int(class_pred[idx])
    gt_cls   = int(sample_classid[idx])

    xc, yc, w, h = bbox_pred[idx].detach().cpu().numpy()
    x1, y1, x2, y2 = xc-w/2, yc-h/2, xc+w/2, yc+h/2

    xc_gt, yc_gt, w_gt, h_gt = sample_bbox[idx].cpu().numpy()
    x1_gt, y1_gt, x2_gt, y2_gt = xc_gt-w_gt/2, yc_gt-h_gt/2, xc_gt+w_gt/2, yc_gt+h_gt/2

    clip = lambda v: float(max(0, min(1, v)))
    x1, y1, x2, y2 = clip(x1), clip(y1), clip(x2), clip(y2)
    x1_gt, y1_gt, x2_gt, y2_gt = clip(x1_gt), clip(y1_gt), clip(x2_gt), clip(y2_gt)

    normal_img = Image.fromarray(img).resize((512, 512))
    bbox_np    = np.array(normal_img).copy()
    H, W, _    = bbox_np.shape

    # Pred box (RED)
    x1p, y1p, x2p, y2p = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
    bbox_np[y1p:y1p+3, x1p:x2p] = [255,0,0];  bbox_np[y2p:y2p+3, x1p:x2p] = [255,0,0]
    bbox_np[y1p:y2p, x1p:x1p+3] = [255,0,0];  bbox_np[y1p:y2p, x2p:x2p+3] = [255,0,0]

    # GT box (GREEN)
    x1g, y1g, x2g, y2g = int(x1_gt*W), int(y1_gt*H), int(x2_gt*W), int(y2_gt*H)
    bbox_np[y1g:y1g+3, x1g:x2g] = [0,255,0];  bbox_np[y2g:y2g+3, x1g:x2g] = [0,255,0]
    bbox_np[y1g:y2g, x1g:x1g+3] = [0,255,0];  bbox_np[y1g:y2g, x2g:x2g+3] = [0,255,0]

    pred_mask_np = torch.sigmoid(mask_pred[idx]).argmax(0).cpu().numpy()
    gt_mask_np   = sample_mask[idx].argmax(0).cpu().numpy()
    mask_vis = np.stack([pred_mask_np*85, gt_mask_np*85,
                         np.zeros_like(pred_mask_np)], axis=-1).astype(np.uint8)
    mask_vis = Image.fromarray(mask_vis).resize((512, 512))

    combined = np.concatenate([np.array(normal_img), bbox_np, np.array(mask_vis)], axis=1)

    wandb.log({
        f"sample_{idx}_combined": wandb.Image(
            combined,
            caption=f"GT: {mappings[gt_cls]} | Pred: {mappings[pred_cls]}"
        )
    }, commit=False)

wandb.log({}, commit=True)

plt.show()

wandb.finish()