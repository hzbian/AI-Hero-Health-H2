import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import pandas as pd

class CovidImageDataset(data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.info_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.df_contains_label = True if 'label' in self.info_df.columns else False

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.info_df.image.iloc[idx])
        image = Image.open(img_name).convert('RGB')
        if self.df_contains_label:
            label = 0 if self.info_df.label.iloc[idx] == 'negative' else 1

        if self.transform == 'resize_rotate_crop':
            image = transforms.Compose([
                transforms.Resize((240, 240)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.Pad((10, 10)),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(image)

        else:
            # for validation
            image = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(image)

        sample = [image, label] if self.df_contains_label else [image, img_name]

        return sample

def loader(path, batch_size=64, num_workers=10, pin_memory=True):
    train_set = CovidImageDataset(
        os.path.join(path, "train.csv"),
        os.path.join(path, 'imgs'),
        transform='resize_rotate_crop')
    return data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

def test_loader(path, batch_size=64, num_workers=10, pin_memory=True):
    val_set = CovidImageDataset(
        os.path.join(path, 'valid.csv'),
        os.path.join(path, 'imgs'),
        transform=None)
    return data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
