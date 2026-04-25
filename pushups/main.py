import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from playsound3 import playsound
import numpy as np

def get_angle(a, b, c):
    cb = np.atan2(c[1] - b[1], c[0] - b[0])
    ab = np.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle

def detect_push_up(keypoints):
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left = get_angle(left_shoulder, left_elbow, left_wrist)
    right = get_angle(right_shoulder, right_elbow, right_wrist)
    is_correct_seq = (left_shoulder[1] <= 
                      left_elbow[1] < 
                      left_wrist[1]) and (right_shoulder[1] <=
                                          right_elbow[1] <
                                          right_wrist[1])
    avg_angle = (left + right) / 2
    if (left <= 111
        and right <= 111 and is_correct_seq):
        return True, avg_angle
    else:
        return False, avg_angle

ps = None
cnt = 0
begin_push_up = False

model = YOLO("yolo26n-pose.pt")

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    results = model(frame)
    person_detected = False
    result = results[0]
    keypoints = result.keypoints.xy.tolist()
    if len(keypoints) > 0:
        if any(result.keypoints.conf[0] > 0.5): 
            person_detected = True

    if person_detected:
        last_seen_time = time.time()
    else:
        if time.time() - last_seen_time > 10:
            if cnt > 0:
                print("Человек исчез надолго. Сброс счетчика.")
                cnt = 0
    if not person_detected:
        cv2.putText(frame, "NO PERSON DETECTED", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow('POSE', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue
    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0],
                   result.orig_shape, 5, True)
    annotated = annotator.result()
    detect, avg_angle = detect_push_up(keypoints[0])
    if detect and not begin_push_up:
        begin_push_up = True
    if begin_push_up and not detect and avg_angle >= 150:
        begin_push_up = False
        cnt += 1
        if ps is None:
            ps = playsound('sound.mp3', block=False, backend='ffplay')
        else:
            if not ps.is_alive():
                ps = None
    cv2.putText(annotated, f"Push-ups: {cnt}, average angle: {int(avg_angle)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, 2)
    cv2.imshow('POSE', annotated)

camera.release()
cv2.destroyAllWindows()