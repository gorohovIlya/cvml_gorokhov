import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision.transforms import v2 as transforms
from pathlib import Path
import zipfile as zpf
import random
from PIL import Image

class Preprocessor:

    def __init__(self, train_rate=0.7):
        self.orig_path = Path("./cyrillic/Cyrillic")
        self.test_path = Path("./test")
        self.train_path = Path("./train")
        self.train_rate = train_rate

    def extract_files_from_zip(self):
        if 'cyrillic' not in os.listdir('./'):
            with zpf.ZipFile('cyrillic.zip', 'r') as zip_ref:
                zip_ref.extractall('cyrillic')
        else:
            print("Folder 'cyrillic' already exists!")
            pass

    def create_letter_directories(self, letter):
        train_path = self.train_path / letter
        test_path = self.test_path / letter
        if not train_path.exists():
            os.mkdir(self.train_path / letter)
        if not test_path.exists():
            os.mkdir(self.test_path / letter)

    def my_train_test_split(self):
        if "train" not in os.listdir("./"):
            os.mkdir("train")
        if "test" not in os.listdir("./"):
            os.mkdir("test")
        letters = os.listdir(self.orig_path)
        for letter in letters:
            self.create_letter_directories(letter)
            print("Current letter: ", letter)
            files = os.listdir(str(self.orig_path / letter))
            random.shuffle(files)
            idx = int(len(files) * self.train_rate)
            for_train = files[:idx]
            for_test = files[idx:]
            for ftr in for_train:
                os.rename(str(self.orig_path / letter / ftr), str(self.train_path / letter / ftr))
            for fts in for_test:
                os.rename(str(self.orig_path / letter / fts), str(self.test_path / letter / fts))
    
    def preprocess(self):
        if len(os.listdir(self.test_path)) == 34:
            print("Data is already preprocessed!")
        else:
            self.extract_files_from_zip()
            self.my_train_test_split()

prep = Preprocessor()
prep.preprocess()

class CyrillicMNISTDataset(Dataset):

    def __init__(self, is_train=True, transforms=None):
        self.path = Path("./" + ("train" if is_train else "test"))
        self.length = 0
        self.files = []
        self.targets = torch.eye(len(os.listdir(self.path)))
        self.classes = os.listdir(self.path)
        self.transforms = transforms
        for label in self.classes:
            path_to_letter_dir = self.path / label
            list_files = os.listdir(path_to_letter_dir)
            self.length += len(list_files)
            for fname in list_files:
                self.files.append((path_to_letter_dir / fname, label))

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        path_file, target = self.files[idx]
        img = Image.open(path_file).split()[-1]
        label = self.classes.index(target)
        return self.transforms(img), label
    
tfs_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomAffine(8, (0.1, 0.1), (0.5, 1)),
            # transforms.RandomResizedCrop(size=(8, 8), scale=(0.8, 1.0), antialias=True),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])

tfs_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
    
cmd_train = CyrillicMNISTDataset(transforms=tfs_train)
cmd_test = CyrillicMNISTDataset(is_train=False, transforms=tfs_test)

BATCH_SIZE = 16
train_loader = DataLoader(cmd_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(cmd_test, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

class CyrillicCNN(nn.Module):

    def __init__(self):
        super(CyrillicCNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) # 96,96 -> 48,48 | 64,64 -> 32,32
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # 48,48 -> 24,24 | 32,32 -> 16,16
        # Block 3
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2) # 24,24 -> 12,12 | 16,16 -> 8,8
        # Block 4
        self.conv4 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2) # 12,12 -> 6,6 | 8,8 -> 4,4
        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 34)


    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        # # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
# img, label = cmd_train[500]
# img = img.numpy().transpose(1, 2, 0)
# plt.imshow(img)
# plt.show()
    
model = CyrillicCNN().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(device)
print(f"{total_params=}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 40
train_loss = []
train_acc = []

save_path = Path(__file__).parent
model_path = save_path / "cyrillicmnist_model.pth"

if not model_path.exists():
    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        total = 0
        correct = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = (images.to(device), labels.to(device))
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        scheduler.step()
        epoch_loss = run_loss / len(train_loader)
        epoch_acc = 100 * (correct / total)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(f"Epoch {epoch}, {epoch_loss:=.3f}, {epoch_acc:=.3f}")
    torch.save(model.state_dict(), model_path)
    plt.figure()
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(train_loss)
    plt.subplot(122)
    plt.title("Acc")
    plt.plot(train_acc)
    plt.show()
else:
    model.load_state_dict(torch.load(model_path))

model.eval()
it = iter(test_loader)  
images, labels = next(it)
image = images[13].unsqueeze(0)
image = image.to(device)

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

classes = cmd_test.classes
print(f"True - {classes[labels[14]]}")
print(f"Pred - {classes[predicted.cpu().item()]}")