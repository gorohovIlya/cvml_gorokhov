import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision.transforms import v2 as transforms
from pathlib import Path
import random
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


path = Path('./roads')

class RoadsDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.images_path = path / 'images'
        self.masks_path = path / 'masks'
        self.images = list(self.images_path.glob('*.png'))
        self.masks = list(self.masks_path.glob('*.png'))
        self.length = len(self.images)
        # for i, m in zip(self.images.glob("*.png"),
        #                 self.masks.glob("*.png")):
        #     print(i, m)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        image = np.array(image, dtype='f4') / 255.
        mask = Image.open(self.masks[index]).convert('L')
        mask = np.array(mask, dtype='f4')
        mask = (mask == 82).astype('f4')
        mask = np.expand_dims(mask, axis=0)
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=2).copy()
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = torch.from_numpy(mask)
        return image, mask

ds = RoadsDataset(path)
print(len(ds))

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, 
                 features=[64, 128, 256, 512]):
        super().__init__()
        self.downscale = nn.ModuleList()
        self.upscale = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for n in features:
            self.downscale.append(DoubleConv(in_channels, n))
            in_channels = n

        for n in features[::-1]:
            self.upscale.append(nn.ConvTranspose2d(n * 2, n, 2, 2))
            self.upscale.append(DoubleConv(n*2, n))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.result = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []

        for ds in self.downscale:
            x = ds(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        for idx in range(0, len(self.upscale), 2):
            x = self.upscale[idx](x)
            skip = skips[idx // 2]
            cx = torch.cat((skip, x), dim=1)
            x = self.upscale[idx + 1](cx)
        return self.result(x)
        

model = UNet()

print(sum([p.numel() for p in model.parameters()]))
