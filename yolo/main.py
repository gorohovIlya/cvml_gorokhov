import matplotlib
matplotlib.use('TkAgg')
from ultralytics import YOLO
from PIL import Image
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

classes = {0: 'cube', 1: 'neither', 2: 'sphere'}

image_path = './for_yolo/ds/images/val/cube/IMG_20260330_191703_143.jpg'

model = YOLO('./runs/detect/figures/yolo6/weights/best.pt')

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break
    prediction = model.predict(source=frame,
                               conf=0.25, iou=0.1, imgsz=640)[0]
    boxes = prediction.boxes.xyxy.cpu().numpy()
    cls = prediction.boxes.cls.cpu().numpy()
    scores = prediction.boxes.conf.cpu().numpy()

    for box, label, score, in zip(boxes, cls, scores):
        x1, y1, x2, y2 = box
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        text_pt = (int(x2), int(y1))
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0))
        cv2.putText(frame, f"{classes[int(label)]}: {score:.2f}", pt1,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

