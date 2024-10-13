import torch
import os
import csv
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets

class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_file, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.pairs = []
        if csv_file.endswith('.csv'):
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    else:
                        self.pairs.append(row)
        else:
            with open(csv_file, 'r') as f:
                for line in f.readlines():
                    self.pairs.append(line.strip().split(' '))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path1, img_path2, match = self.pairs[idx]
        img1 = Image.open(os.path.join(self.data_dir, img_path1)).convert('RGB')
        img2 = Image.open(os.path.join(self.data_dir, img_path2)).convert('RGB')
        
        img1 = np.array(img1)
        img2 = np.array(img2)

        if self.transform:
            transformed1 = self.transform(image=img1)
            transformed2 = self.transform(image=img2)
            img1 = transformed1['image']
            img2 = transformed2['image']

        return img1, img2, int(match)

class TestImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_file, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.pairs = []
        if csv_file.endswith('.csv'):
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    else:
                        self.pairs.append(row)
        else:
            with open(csv_file, 'r') as f:
                for line in f.readlines():
                    self.pairs.append(line.strip().split(' '))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path1, img_path2 = self.pairs[idx]
        img1 = Image.open(os.path.join(self.data_dir, img_path1)).convert('RGB')
        img2 = Image.open(os.path.join(self.data_dir, img_path2)).convert('RGB')

        img1 = np.array(img1)
        img2 = np.array(img2)

        if self.transform:
            transformed1 = self.transform(image=img1)
            transformed2 = self.transform(image=img2)
            img1 = transformed1['image']
            img2 = transformed2['image']

        return img1, img2

class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            augmented = self.transform(image=np.array(img))
            img = augmented['image']
        return img, label

    def __len__(self):
        return len(self.dataset)

# Define transforms
train_transforms = A.Compose([
    A.RandomResizedCrop(height=112, width=112, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.RandomGamma(gamma_limit=(80, 120), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(height=128, width=128),
    A.CenterCrop(height=112, width=112),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])