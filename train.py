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


parser.add_argument('-n_breeds', "--num_breeds",           type   = int,          default = 37,                                      help="Total number of breeds to classify")
parser.add_argument('-s_class',  "--seg_classes",          type   = int,          default = 3,                                       help="Total number of classes to segment")
parser.add_argument('-in_c',     "--in_channels",          type   = int,          default = 3,                                       help="In channels of images")
parser.add_argument('-c_path',   "--classifier_path",      type   = str,          default = "",                                      help="Path for classifier model (.pth)")
parser.add_argument('-l_path',   "--localizer_path",       type   = str,          default = "",                                      help="Path for localizer model (.pth)")
parser.add_argument('-u_path',   "--unet_path",            type   = str,          default = "",                                      help="Path for unet model (.pth)")
parser.add_argument('-d_path',   "--dataset_path",         type   = str,          default = "oxford-iiit-pet",                       help="Path for dataset model")
parser.add_argument('-t_ratio',  "--train_ratio",          type   = float,        default = 0.8,                                     help="train ration between 0 and 1")
parser.add_argument('-bs',       "--batch_size",           type   = int,          default = 3,                                       help="batch size")
parser.add_argument('-ep',       "--epochs",               type   = int,          default = 10,                                      help="epochs")
parser.add_argument('-sp',       "--classifier_save_path", type   = str,          default = "checkpoint_2/classifier_save_path.pth", help="where to save the trained classifier model")
parser.add_argument('-sl',       "--localizer_save_path",  type   = str,          default = "checkpoint_2/localizer_save_path.pth",  help="where to save the trained localizer model")
parser.add_argument('-su',       "--unet_save_path",       type   = str,          default = "checkpoint_2/unet_save_path.pth",       help="where to save the trained unet model")
parser.add_argument('-t_c',      "--train_classifier",     action = "store_true", default = True,                                    help="want to train classifier?")
parser.add_argument('-t_l',      "--train_localizer",      action = "store_true", default = True,                                    help="want to train localizer?")
parser.add_argument('-t_u',      "--train_unet",           action = "store_true", default = True,                                    help="want to train unet?")
parser.add_argument('-reuse',    "--reuse_classifer",      action = "store_true", default = True,                                    help="Reuse classifier saved from this training loop?")
parser.add_argument('-save',     "--save_every",           type   = int         , default = 5,                                       help="After how many epochs to save?")


args = parser.parse_args()

wandb.init(project="Multitask-Pet-Detection", config=vars(args))

multitask_model = MultiTaskPerceptionModel(num_breeds      = args.num_breeds, 
                                           seg_classes     = args.seg_classes, 
                                           in_channels     = args.in_channels, 
                                           classifier_path = args.classifier_path,
                                           localizer_path  = args.localizer_path,
                                           unet_path       = args.dataset_path)


device = "cuda" if torch.cuda.is_available() else "cpu"

classifier = multitask_model.model_classifier.to(device)
localizer  = multitask_model.model_localizer.to(device)
unet       = multitask_model.unet.to(device)


classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-4)
classifier_loss_fn   = nn.CrossEntropyLoss()

localizer_optimizer = torch.optim.Adam(localizer.parameters(), lr=1e-4, weight_decay=1e-4)
localizer_loss_fn = IoULoss()

unet_optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4, weight_decay=1e-4)
bce = nn.BCEWithLogitsLoss()
def loss_fn(pred, target):
    return bce(pred, target) + DiceLoss()(pred, target)


mappings = get_class_map(pathlib.Path(args.dataset_path))

dataset = OxfordIIITPetDataset(root_dir = args.dataset_path)

train_ds, test_ds = random_split(dataset, [int(args.train_ratio * len(dataset)), len(dataset)-int(args.train_ratio * len(dataset))])

train_dl = DataLoader(train_ds, batch_size = args.batch_size, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size = args.batch_size, shuffle=True)


if args.train_classifier:
    print("TRAINING CLASSIFIER")
    train_loss_tracker = []
    test_loss_tracker  = []
    for epoch in range(args.epochs):
        # Train
        classifier.train()
        train_loss = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            images = batch['image'].to(device)
            class_ids = batch['class_id'].to(device)

            logits = classifier(images)
            loss = classifier_loss_fn(logits, class_ids)

            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)
        train_loss_tracker.append(train_loss)

        # Test
        classifier.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(test_dl, desc=f"Epoch {epoch+1}/{args.epochs} [test] "):
                images = batch['image'].to(device)
                class_ids = batch['class_id'].to(device)

                logits = classifier(images)
                loss = classifier_loss_fn(logits, class_ids)
                test_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == class_ids).sum().item()
                total += class_ids.size(0)

        test_loss /= len(test_dl)
        test_loss_tracker.append(test_loss)
        test_acc = correct / total * 100

        wandb.log({
            "classifier/train_loss": train_loss,
            "classifier/test_loss": test_loss,
            "classifier/test_acc": test_acc,
            "epoch": epoch
        })

        if epoch % args.save_every == 0:
            if "/" in args.classifier_save_path:
                p = "/".join(args.classifier_save_path.split("/")[:-1])
                os.makedirs(p, exist_ok=True)
            torch.save({
                'state_dict': classifier.state_dict(),
                'opt_state_dict': classifier_optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss}
                    , args.classifier_save_path.replace(".pth", f"epoch-{epoch}.pth"))

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")


    if "/" in args.classifier_save_path:
        p = "/".join(args.classifier_save_path.split("/")[:-1])
    os.makedirs(p, exist_ok=True)

    torch.save({
        'state_dict': classifier.state_dict(),
        'opt_state_dict': classifier_optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss}
            , args.classifier_save_path)

    print("Classiifer file saved")
    print("\n" + "="*20 + "\n")









if args.train_localizer:
    print("LOCALIZER CLASSIFIER")

    train_loss_tracker = []
    train_loss_tracker = []
    train_conf_tracker = []
    test_conf_tracker  = []

    if (args.train_classifier and args.classifier_save_path) or (args.classifier_path):
        classifier_pth = args.classifier_save_path if args.reuse_classifer else args.classifier_path
        multitask_model = MultiTaskPerceptionModel(num_breeds   = args.num_breeds, 
                                                seg_classes     = args.seg_classes, 
                                                in_channels     = args.in_channels, 
                                                classifier_path = args.classifier_save_path, # I might need vgg11 pretrained for localizer
                                                localizer_path  = args.localizer_path,
                                                unet_path       = args.unet_path)
        
    localizer  = multitask_model.model_localizer.to(device)

    for epoch in range(args.epochs):
        # Train
        localizer.train()
        train_loss = 0.0
        train_conf = 0.0
    
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            images = batch['image'].to(device)
            class_ids = batch['class_id'].to(device)
            bbox = batch['bbox'].to(device)
    
            pred_bbox = localizer(images)
            iou = localizer_loss_fn(pred_bbox, bbox) # iou loss actually
            confidence = 1 - iou
            l1  = torch.abs(pred_bbox - bbox).mean()
            loss = iou + 1.0 * l1 # l1

            localizer_optimizer.zero_grad()
            loss.backward()
            localizer_optimizer.step()
    
            train_loss += loss.item()
            train_conf += confidence.item()
    
        train_loss /= len(train_dl)
        train_conf /= len(train_dl)
        train_loss_tracker.append(train_loss)
        train_conf_tracker.append(train_conf)
    
        # Test
        localizer.eval()
        test_loss = 0.0
        test_conf = 0.0

        with torch.no_grad():
            for batch in tqdm(test_dl, desc=f"Epoch {epoch+1}/{args.epochs} [test] "):
                images = batch['image'].to(device)
                class_ids = batch['class_id'].to(device)
                bbox = batch['bbox'].to(device)
    
                pred_bbox = localizer(images)
                iou = localizer_loss_fn(pred_bbox, bbox)
                confidence = 1 - iou
                l1  = torch.abs(pred_bbox - bbox).mean()
                loss = iou + 1.0 * l1 # l1

                test_loss += loss.item()
                test_conf += confidence.item()
    
        test_loss /= len(test_dl)
        test_conf /= len(test_dl)
        test_loss_tracker.append(test_loss)
        test_conf_tracker.append(test_conf)

        wandb.log({
            "localizer/train_loss": train_loss,
            "localizer/test_loss":  test_loss,
            "localizer/train_conf": train_conf,
            "localizer/test_conf":  test_conf,
            "epoch": epoch
        })

        if epoch % args.save_every == 0:
            if "/" in args.localizer_save_path:
                p = "/".join(args.localizer_save_path.split("/")[:-1])
                os.makedirs(p, exist_ok=True)
            torch.save({
                'state_dict': localizer.state_dict(),
                'opt_state_dict': localizer_optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss}
                    , args.localizer_save_path.replace(".pth", f"epoch-{epoch}.pth"))

    
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} ")


    if "/" in args.localizer_save_path:
        p = "/".join(args.localizer_save_path.split("/")[:-1])
    os.makedirs(p, exist_ok=True)


    torch.save({
        'state_dict': localizer.state_dict(),
        'opt_state_dict': localizer_optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss}
            , args.localizer_save_path)

    print("Localizer file saved")
    print("\n" + "="*20 + "\n")







if args.train_unet:
    print("UNET CLASSIFIER")
    train_loss_tracker = []
    test_loss_tracker = []
    for epoch in range(args.epochs):
        # Train
        unet.train()
        train_loss = 0.0
    
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
    
            pred_mask = unet(images)
            loss = unet_loss_fn(pred_mask, masks)

            unet_optimizer.zero_grad()
            loss.backward()
            unet_optimizer.step()
    
            train_loss += loss.item()
    
        train_loss /= len(train_dl)
        train_loss_tracker.append(train_loss)
    
        unet.eval()
        test_loss = 0.0
    
        with torch.no_grad():
            for batch in tqdm(test_dl, desc=f"Epoch {epoch+1}/{args.epochs} [test] "):
                masks = batch['mask'].to(device)
                images = batch['image'].to(device)
                pred_mask = unet(images)
                loss = loss_fn(pred_mask, masks)

                test_loss += loss.item()
    
    
        test_loss /= len(test_dl)
        test_loss_tracker.append(test_loss)

        wandb.log({
            "unet/train_loss": train_loss,
            "unet/test_loss": test_loss,
            "epoch": epoch
        })

        if epoch % args.save_every == 0:
            if "/" in args.unet_save_path:
                p = "/".join(args.unet_save_path.split("/")[:-1])
                os.makedirs(p, exist_ok=True)
            torch.save({
                'state_dict': unet.state_dict(),
                'opt_state_dict': unet_optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss}
                    , args.unet_save_path.replace(".pth", f"epoch-{epoch}.pth"))
        

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} ")

    if "/" in args.unet_save_path:
        p = "/".join(args.unet_save_path.split("/")[:-1])
    os.makedirs(p, exist_ok=True)

    torch.save({
        'state_dict': unet.state_dict(),
        'opt_state_dict': unet_optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss}
            , args.unet_save_path)

    print("Unet file saved")
    print("\n" + "="*20 + "\n")