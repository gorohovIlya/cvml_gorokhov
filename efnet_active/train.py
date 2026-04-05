import torch
import torch.nn as nn
from pathlib import Path
import time
from collections import deque
import torchvision
import cv2
from torchvision import transforms

save_path = Path(__file__).parent
model_path = save_path / 'model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights)
    for param in model.features.parameters():
        param.requires_grad = False
    features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(features, 1)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, 
                                         map_location=torch.device(device),
                                         weights_only=True))
    return model.to(device)

model = build_model()
print(model)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

def train(buffer):
    if len(buffer) < 10:
        return None
    model.train()
    images, labels = buffer.get_batch()
    optimizer.zero_grad()
    predictions = model(images).squeeze(1)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

class Buffer:
    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)
    
    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)

    def __len__(self):
        return len(self.frames)
    
    def get_batch(self):
        images = torch.stack(list(self.frames)).to(device)
        labels = torch.tensor(list(self.labels), dtype=torch.float32).to(device)
        return images, labels

if __name__ == "__main__":
    model = build_model()
    print(model)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
    buffer = Buffer()
    count_labeled = 0
    
    while True:
        _, frame = cap.read()
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if key == ord('q'):
            break
        elif key == ord('1'):
            tensor = transform(image)
            buffer.append(tensor, 1.0)
            count_labeled += 1
        elif key == ord('2'):
            tensor = transform(image)
            buffer.append(tensor, 0.0)
            count_labeled += 1
        elif key == ord('s'):
            torch.save(model.state_dict(), model_path)
            print(f"Модель сохранена в {model_path}")
        
        if count_labeled >= buffer.frames.maxlen:
            loss = train(buffer)
            if loss:
                print(f'Loss = {loss}')
            count_labeled = 0
    
    cap.release()
    cv2.destroyAllWindows()