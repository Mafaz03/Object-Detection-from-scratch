import argparse

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
import wandb

bce = nn.BCEWithLogitsLoss()

def unet_loss_fn(pred, target):
    return bce(pred, target) + DiceLoss()(pred, target)

def bbox_loss_fn(pred, target):
    iou_loss = IoULoss()
    return iou_loss(pred, target)


parser = argparse.ArgumentParser(description="Inferencing Object detection", conflict_handler="resolve")

parser.add_argument('-n_breeds', "--num_breeds",           type=int,           default=37,                            help="Total number of breeds to classify")
parser.add_argument('-s_class',  "--seg_classes",          type=int,           default=3,                             help="Total number of classes to segment")
parser.add_argument('-in_c',     "--in_channels",          type=int,           default=3,                             help="In channels of images")
parser.add_argument('-c_path',   "--classifier_path",      type=str,           default="",                            help="Path for classifier model (.pth)")
parser.add_argument('-l_path',   "--localizer_path",       type=str,           default="",                            help="Path for localizer model (.pth)")
parser.add_argument('-u_path',   "--unet_path",            type=str,           default="",                            help="Path for unet model (.pth)")
parser.add_argument('-d_path',   "--dataset_path",         type=str,           default="oxford-iiit-pet",             help="Path for dataset model")
parser.add_argument('-t_ratio',  "--train_ratio",          type=float,         default=0.8,                           help="train ratio between 0 and 1")
parser.add_argument('-bs',       "--batch_size",           type=int,           default=3,                             help="batch size")
parser.add_argument('-ep',       "--epochs",               type=int,           default=10,                            help="epochs")
parser.add_argument('-sp',       "--classifier_save_path", type=str,           default="checkpoints_/classifier.pth", help="where to save the trained classifier model")
parser.add_argument('-sl',       "--localizer_save_path",  type=str,           default="checkpoints_/localizer.pth",  help="where to save the trained localizer model")
parser.add_argument('-su',       "--unet_save_path",       type=str,           default="checkpoints_/unet.pth",       help="where to save the trained unet model")
parser.add_argument('-t_c',      "--train_classifier",     action="store_true",                                       help="want to train classifier?")
parser.add_argument('-t_l',      "--train_localizer",      action="store_true",                                       help="want to train localizer?")
parser.add_argument('-t_u',      "--train_unet",           action="store_true",                                       help="want to train unet?")
parser.add_argument('-reuse',    "--reuse_classifer",      action="store_true",                                       help="Reuse classifier saved from this training loop?")
parser.add_argument('-save',     "--save_every",           type=int,           default=5,                             help="After how many epochs to save?")
parser.add_argument('-bn',       "--use_batchnorm",        action="store_true",                                       help="Use batch norm in vgg11 classifier or not?")
parser.add_argument('-do',       "--dropout",              type=float,         default=0.5,                           help="Dropout")
parser.add_argument('-tl',       "--transfer_learning",    type=str,           default="freeze all",                  help="`freeze all` or `partial unfreeze` or `unfreeze all`")
parser.add_argument('-lr',       "--learning_rate",        type=float,         default=1e-4,                          help="learning rate for all")
parser.add_argument('-lc',       "--load_classifier",      action="store_true",                                       help="load classifier?")
parser.add_argument('-ll',       "--load_localizer",       action="store_true",                                       help="load localizer?")
parser.add_argument('-lu',       "--load_unet",            action="store_true",                                       help="load unet?")
parser.add_argument('-lu',       "--learning_rate",        type=float,         default=1e-4,                          help="learning rate for all")

args = parser.parse_args()

wandb.init(project="Multitask-Pet-Detection", config=vars(args))

# Force train and test lines onto the same plot for each metric
wandb.define_metric("epoch")
wandb.define_metric("classifier_loss/train", step_metric="epoch")
wandb.define_metric("classifier_loss/test",  step_metric="epoch")
wandb.define_metric("classifier_acc/test",   step_metric="epoch")
wandb.define_metric("localizer_loss/train",  step_metric="epoch")
wandb.define_metric("localizer_loss/test",   step_metric="epoch")
wandb.define_metric("localizer_conf/train",  step_metric="epoch")
wandb.define_metric("localizer_conf/test",   step_metric="epoch")
wandb.define_metric("unet_loss/train",       step_metric="epoch")
wandb.define_metric("unet_loss/test",        step_metric="epoch")
wandb.define_metric("unet_dice/train",       step_metric="epoch")
wandb.define_metric("unet_dice/test",        step_metric="epoch")
wandb.define_metric("unet_acc/train",        step_metric="epoch")
wandb.define_metric("unet_acc/test",         step_metric="epoch")

multitask_model = MultiTaskPerceptionModel(num_breeds        = args.num_breeds,
                                           seg_classes       = args.seg_classes,
                                           in_channels       = args.in_channels,
                                           classifier_path   = args.classifier_path,
                                           localizer_path    = args.localizer_path,
                                           unet_path         = args.unet_path,
                                           use_batchnorm     = args.use_batchnorm,
                                           transfer_learning = args.transfer_learning,
                                           train_classifier  = args.load_classifier,
                                           train_localizer   = args.load_localizer,
                                           train_unet        = args.load_unet
                                           )

device = "cuda" if torch.cuda.is_available() else "cpu"

classifier = multitask_model.model_classifier.to(device)
localizer  = multitask_model.model_localizer.to(device)
unet       = multitask_model.unet.to(device)

classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)
classifier_loss_fn   = nn.CrossEntropyLoss()

localizer_optimizer = torch.optim.Adam(localizer.parameters(), lr=args.learning_rate, weight_decay=1e-4)
localizer_loss_fn   = IoULoss()

unet_optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate, weight_decay=1e-4)
bce = nn.BCEWithLogitsLoss()

def loss_fn(pred, target):
    return bce(pred, target) + DiceLoss()(pred, target)


mappings = get_class_map(pathlib.Path(args.dataset_path))

dataset = OxfordIIITPetDataset(root_dir=args.dataset_path)

train_ds, test_ds = random_split(dataset, [int(args.train_ratio * len(dataset)),
                                            len(dataset) - int(args.train_ratio * len(dataset))])

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=True)


# ── Classifier ────────────────────────────────────────────────────────────────
if args.train_classifier:
    print("TRAINING CLASSIFIER")

    for epoch in range(args.epochs):
        # Train
        classifier.train()
        train_loss = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            images    = batch['image'].to(device)
            class_ids = batch['class_id'].to(device)

            logits = classifier(images)
            loss   = classifier_loss_fn(logits, class_ids)

            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)

        # Test
        classifier.eval()
        test_loss = 0.0
        correct   = 0
        total     = 0

        with torch.no_grad():
            for batch in tqdm(test_dl, desc=f"Epoch {epoch+1}/{args.epochs} [test] "):
                images    = batch['image'].to(device)
                class_ids = batch['class_id'].to(device)

                logits = classifier(images)
                loss   = classifier_loss_fn(logits, class_ids)
                test_loss += loss.item()

                preds    = logits.argmax(dim=1)
                correct += (preds == class_ids).sum().item()
                total   += class_ids.size(0)

        test_loss /= len(test_dl)
        test_acc   = correct / total * 100

        # FIX: single log call so train and test share the same x-axis step
        wandb.log({
            "classifier_loss/train": train_loss,
            "classifier_loss/test":  test_loss,
            "classifier_acc/test":   test_acc,
            "epoch": epoch,
        })

        if epoch % args.save_every == 0:
            if "/" in args.classifier_save_path:
                os.makedirs("/".join(args.classifier_save_path.split("/")[:-1]), exist_ok=True)
            torch.save({
                'state_dict':     classifier.state_dict(),
                'opt_state_dict': classifier_optimizer.state_dict(),
                'epoch': epoch, 'loss': loss},
                args.classifier_save_path.replace(".pth", f"_epoch-{epoch}.pth"))

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    if "/" in args.classifier_save_path:
        os.makedirs("/".join(args.classifier_save_path.split("/")[:-1]), exist_ok=True)
    torch.save({
        'state_dict':     classifier.state_dict(),
        'opt_state_dict': classifier_optimizer.state_dict(),
        'epoch': epoch, 'loss': loss},
        args.classifier_save_path)

    print("Classifier file saved")
    print("\n" + "="*20 + "\n")


# ── Localizer ─────────────────────────────────────────────────────────────────
if args.train_localizer:
    print("TRAINING LOCALIZER")

    if (args.train_classifier and args.classifier_save_path) or args.classifier_path:
        multitask_model = MultiTaskPerceptionModel(
            num_breeds      = args.num_breeds,
            seg_classes     = args.seg_classes,
            in_channels     = args.in_channels,
            classifier_path = args.classifier_save_path if args.reuse_classifer else args.classifier_path,
            localizer_path  = args.localizer_path,
            unet_path       = args.unet_path)

    localizer = multitask_model.model_localizer.to(device)

    for epoch in range(args.epochs):
        # Train
        localizer.train()
        train_loss = 0.0
        train_conf = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            images    = batch['image'].to(device)
            bbox      = batch['bbox'].to(device)

            pred_bbox  = localizer(images)
            iou        = localizer_loss_fn(pred_bbox, bbox)
            confidence = 1 - iou
            l1         = torch.abs(pred_bbox - bbox).mean()
            loss       = iou + 1.0 * l1

            localizer_optimizer.zero_grad()
            loss.backward()
            localizer_optimizer.step()

            train_loss += loss.item()
            train_conf += confidence.item()

        train_loss /= len(train_dl)
        train_conf /= len(train_dl)

        # Test
        localizer.eval()
        test_loss = 0.0
        test_conf = 0.0

        with torch.no_grad():
            for batch in tqdm(test_dl, desc=f"Epoch {epoch+1}/{args.epochs} [test] "):
                images = batch['image'].to(device)
                bbox   = batch['bbox'].to(device)

                pred_bbox  = localizer(images)
                iou        = localizer_loss_fn(pred_bbox, bbox)
                confidence = 1 - iou
                l1         = torch.abs(pred_bbox - bbox).mean()
                loss       = iou + 1.0 * l1

                test_loss += loss.item()
                test_conf += confidence.item()

        test_loss /= len(test_dl)
        test_conf /= len(test_dl)

        wandb.log({
            "localizer_loss/train": train_loss,
            "localizer_loss/test":  test_loss,
            "localizer_conf/train": train_conf,
            "localizer_conf/test":  test_conf,
            "epoch": epoch,
        })

        if epoch % args.save_every == 0:
            if "/" in args.localizer_save_path:
                os.makedirs("/".join(args.localizer_save_path.split("/")[:-1]), exist_ok=True)
            torch.save({
                'state_dict':     localizer.state_dict(),
                'opt_state_dict': localizer_optimizer.state_dict(),
                'epoch': epoch, 'loss': loss},
                args.localizer_save_path.replace(".pth", f"_epoch-{epoch}.pth"))

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    if "/" in args.localizer_save_path:
        os.makedirs("/".join(args.localizer_save_path.split("/")[:-1]), exist_ok=True)
    torch.save({
        'state_dict':     localizer.state_dict(),
        'opt_state_dict': localizer_optimizer.state_dict(),
        'epoch': epoch, 'loss': loss},
        args.localizer_save_path)

    print("Localizer file saved")
    print("\n" + "="*20 + "\n")


# ── UNet ──────────────────────────────────────────────────────────────────────
if args.train_unet:
    print("TRAINING UNET")

    def dice_score(pred, target, threshold=0.5):
        pred   = (torch.sigmoid(pred) > threshold).float()
        inter  = (pred * target).sum(dim=(1, 2, 3))
        union  = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice   = (2 * inter / (union + 1e-6)).mean()
        return dice.item()

    def pixel_accuracy(pred, target, threshold=0.5):
        pred    = (torch.sigmoid(pred) > threshold).float()
        correct = (pred == target).float().sum()
        total   = torch.numel(pred)
        return (correct / total).item()

    for epoch in range(args.epochs):
        # Train
        unet.train()
        train_loss = 0.0
        train_dice = 0.0
        train_acc  = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)

            pred_mask = unet(images)
            loss      = unet_loss_fn(pred_mask, masks)

            unet_optimizer.zero_grad()
            loss.backward()
            unet_optimizer.step()

            train_loss += loss.item()
            train_dice += dice_score(pred_mask, masks)
            train_acc  += pixel_accuracy(pred_mask, masks)

        train_loss /= len(train_dl)
        train_dice /= len(train_dl)
        train_acc  /= len(train_dl)

        # Test
        unet.eval()
        test_loss = 0.0
        test_dice = 0.0
        test_acc  = 0.0

        with torch.no_grad():
            for batch in tqdm(test_dl, desc=f"Epoch {epoch+1}/{args.epochs} [test] "):
                images = batch['image'].to(device)
                masks  = batch['mask'].to(device)

                pred_mask = unet(images)
                loss      = loss_fn(pred_mask, masks)

                test_loss += loss.item()
                test_dice += dice_score(pred_mask, masks)
                test_acc  += pixel_accuracy(pred_mask, masks)

        test_loss /= len(test_dl)
        test_dice /= len(test_dl)
        test_acc  /= len(test_dl)

        wandb.log({
            "unet_loss/train": train_loss,
            "unet_loss/test":  test_loss,
            "unet_dice/train": train_dice,
            "unet_dice/test":  test_dice,
            "unet_acc/train":  train_acc,
            "unet_acc/test":   test_acc,
            "epoch": epoch,
        })

        if epoch % args.save_every == 0:
            if "/" in args.unet_save_path:
                os.makedirs("/".join(args.unet_save_path.split("/")[:-1]), exist_ok=True)
            torch.save({
                'state_dict':     unet.state_dict(),
                'opt_state_dict': unet_optimizer.state_dict(),
                'epoch': epoch, 'loss': loss},
                args.unet_save_path.replace(".pth", f"_epoch-{epoch}.pth"))

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
              f"Train Dice: {train_dice:.4f} | Test Dice: {test_dice:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
    trainable     = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in unet.parameters() if not p.requires_grad)
    wandb.config.update({
        "trainable_params":     trainable,
        "non_trainable_params": non_trainable,
        "transfer_learning":    args.transfer_learning
    })
    print(f"Trainable: {trainable:,} | Frozen: {non_trainable:,}")

    if "/" in args.unet_save_path:
        os.makedirs("/".join(args.unet_save_path.split("/")[:-1]), exist_ok=True)
    torch.save({
        'state_dict':     unet.state_dict(),
        'opt_state_dict': unet_optimizer.state_dict(),
        'epoch': epoch, 'loss': loss},
        args.unet_save_path)

    print("Unet file saved")
    print("\n" + "="*20 + "\n")