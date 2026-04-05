import time
import cv2
import torch
import sys
from train import build_model, transform, device, model_path

if not model_path.exists():
    raise RuntimeError(f"Модель не найдена в {model_path}. Сначала запустите train.py для обучения.")

model = build_model()
model.eval()

def predict(frame):
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()
    label = 'person' if prob > 0.5 else 'no_person'
    return label, prob

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

while True:
    _, frame = cap.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('p'):
        t = time.perf_counter()
        label, conf = predict(frame)
        print(f"Elapsed time: {time.perf_counter() - t}")
        print(label, conf)

cap.release()
cv2.destroyAllWindows()