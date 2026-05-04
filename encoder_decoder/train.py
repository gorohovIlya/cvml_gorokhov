import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
import torch.optim as optim
import string
import random

letters = string.ascii_letters + string.digits

class ImageDataset(Dataset):

    def __init__(self, n=200, size=128, state=1):
        super().__init__()
        self.n = n
        self.size = size
        self.state = state
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        image = Image.new('L', (self.size, self.size), color=255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        x = None
        y = None
        text = None
        if self.state == 1:
            text = "ABCDE"    # Фиксированный текст
            x = np.random.randint(10, self.size - 40)
            y = np.random.randint(10, self.size - 40)
        elif self.state == 2:
            textlen = 5    # Фиксированная длина текста
            text = ''.join(random.choice(letters) for _ in range(textlen))
            x = 30
            y = 30
        elif self.state == 3:
            x = 30
            y = 30
            textlen = random.randint(3, 12)
            text = ''.join(random.choice(letters) for _ in range(textlen))
        elif self.state == 4:
            textlen = random.randint(3, 12)
            text = ''.join(random.choice(letters) for _ in range(textlen))
            x = np.random.randint(10, self.size - 40)
            y = np.random.randint(10, self.size - 40)
        draw.text((x, y), text, fill=0, font=font)
        tensor = self.transform(image)
        return tensor, tensor
    
# ds = ImageDataset(2000, 256, state=2)
# dataloader = DataLoader(ds, batch_size=32, shuffle=True)
# print(ds[0][0].shape)
# plt.imshow(ds[0][0][0])
# plt.show()


class Encoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.bottleneck = nn.Linear(256 * 16 * 16, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x
    

class Decoder(nn.Module):
    
    def __init__(self, latent_size=512):
        super().__init__()
        self.bottleneck = nn.Linear(latent_size, 256 * 16 * 16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x
    
if __name__ == "__main__":
    for st in range(1, 5):
        encoder = Encoder()
        decoder = Decoder()

        ds = ImageDataset(2000, 256, state=st)
        dataloader = DataLoader(ds, batch_size=32, shuffle=True)
        
        ep =sum(p.numel() for p in encoder.parameters())
        dp = sum(p.numel() for p in decoder.parameters())

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(ep, dp, device)

        encoder.to(device)
        decoder.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(encoder.parameters()) + 
                            list(decoder.parameters()))

        encoder.train()
        decoder.train()
        epochs = 10
        for epoch in range(epochs):
            epoch_loss = 0.0
            for imgs, _ in dataloader:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                latent = encoder(imgs)
                output = decoder(latent)
                loss = criterion(imgs, output)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            print(f"{epoch=}, {avg_loss=:.2f}")

        torch.save(encoder.state_dict(), f"encoder_{ds.state}.pth")
        torch.save(decoder.state_dict(), f"decoder_{ds.state}.pth")