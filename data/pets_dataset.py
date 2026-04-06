
from PIL import Image
import pathlib
from torch.utils.data import Dataset, random_split, DataLoader, dataset
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import xml.etree.ElementTree as ET



def get_class_map(root):
    class_map = {}
    with open(root / "annotations" / "list.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            name, class_id, species, breed_id = line.strip().split()
            class_id = int(class_id) - 1

            # Store mapping (only once per class_id)
            breed_name = "_".join(name.split("_")[:-1])

            if class_id not in class_map:
                class_map[class_id] = breed_name
    return class_map


class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root = pathlib.Path(root_dir)
        self.images_dir = self.root / "images"
        self.xml_dir = self.root / "annotations" / "xmls"
        self.trimap_dir = self.root / "annotations" / "trimaps"
        self.transforms = transforms

        self.class_map = {}

        # Read list.txt
        self.samples = []
        with open(self.root / "annotations" / "list.txt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                name, class_id, species, breed_id = line.strip().split()
                class_id = int(class_id) - 1

                # Store mapping (only once per class_id)
                breed_name = "_".join(name.split("_")[:-1])

                if class_id not in self.class_map:
                    self.class_map[class_id] = breed_name

                img_path = self.images_dir / f"{name}.jpg"
                xml_path = self.xml_dir / f"{name}.xml"
                trimap_path = self.trimap_dir / f"{name}.png"

                if img_path.exists() and xml_path.exists() and trimap_path.exists():
                        self.samples.append({
                            "name": name,
                            "class_id": class_id,
                            "species": int(species),
                            "breed_id": int(breed_id)
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        name = sample["name"]

        # Paths
        img_path = self.images_dir / f"{name}.jpg"
        xml_path = self.xml_dir / f"{name}.xml"
        trimap_path = self.trimap_dir / f"{name}.png"

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)/255.0
        image = F.interpolate(image.unsqueeze(0), [224, 224], mode='bilinear')
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        image = (image - mean) / std

        # Load mask
        mask = Image.open(trimap_path)
        mask = np.array(mask)
        mask_1 = (mask == 1).astype(np.float32)
        mask_2 = (mask == 2).astype(np.float32)
        mask_3 = (mask == 3).astype(np.float32)
        mask = np.stack((mask_1, mask_2, mask_3))
        mask = torch.from_numpy(mask)
        mask = F.interpolate(mask.unsqueeze(0), [224,224], mode='bilinear', align_corners=False)
        # mask = mask.unsqueeze(1)

        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        orig_w, orig_h = Image.open(img_path).size
        H, W = 224, 224

        xc = (xmin + xmax) / 2 / orig_w
        yc = (ymin + ymax) / 2 / orig_h
        w  = (xmax - xmin) / W
        h  = (ymax - ymin) / H

        bbox = torch.tensor([xc, yc, w, h], dtype=torch.float32)

        scale_x = 224 / orig_w
        scale_y = 224 / orig_h

        xmin_224 = xmin * scale_x
        xmax_224 = xmax * scale_x
        ymin_224 = ymin * scale_y
        ymax_224 = ymax * scale_y

        xc_224 = (xmin_224 + xmax_224) / 2
        yc_224 = (ymin_224 + ymax_224) / 2
        w_224  = (xmax_224 - xmin_224)
        h_224  = (ymax_224 - ymin_224)

        bbox_224 = torch.tensor([xc_224, yc_224, w_224, h_224], dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image)

        return {
            "image": image[0],
            "name": name,
            "bbox": bbox,
            "bbox_224": bbox_224,
            "mask": mask[0],
            "species": sample["species"],   # 1=cat, 2=dog
            "breed": sample["breed_id"],
            "class_id": sample["class_id"],
            "breed_name": self.class_map[sample["class_id"]]
        }