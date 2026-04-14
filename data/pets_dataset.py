import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.images_dir = os.path.join(root_dir, "images")
        self.annotations_dir = os.path.join(root_dir, "annotations")

        if split == "train":
            split_file = os.path.join(self.annotations_dir, "trainval.txt")
        else:
            split_file = os.path.join(self.annotations_dir, "test.txt")

        self.samples = []

        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                label = int(parts[1]) - 1
                self.samples.append((img_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]

        img_path = os.path.join(self.images_dir, img_name + ".jpg")
        image = Image.open(img_path).convert("RGB")

        mask_path = os.path.join(self.annotations_dir, "trimaps", img_name + ".png")
        mask = Image.open(mask_path)

        mask = mask.resize((224, 224), resample=Image.NEAREST)

        mask = np.array(mask) - 1
        mask = torch.tensor(mask, dtype=torch.long)

        bbox_path = os.path.join(self.annotations_dir, "xmls", img_name + ".xml")
        bbox = self.parse_bbox(bbox_path, image.size)

        if self.transform:
            image = self.transform(image)

        return image, label, bbox, mask

    def parse_bbox(self, xml_path, img_size):
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml_path)
        root = tree.getroot()

        bndbox = root.find("object").find("bndbox")

        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        W, H = img_size

        x_center = ((xmin + xmax) / 2) / W
        y_center = ((ymin + ymax) / 2) / H
        w = (xmax - xmin) / W
        h = (ymax - ymin) / H

        return torch.tensor([x_center, y_center, w, h], dtype=torch.float32)