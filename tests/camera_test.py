import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from definitions import ROOT_DIR
from method.method import Method
from method.method_payload import MethodPayload
from scripts.gestures import Gesture10

MODELS_DIR = os.path.join(ROOT_DIR, 'runs', 'classify')

YOLO_MODELS = {
    "YOLO 3 epochs":  os.path.join(MODELS_DIR, 'train3epochs',  'weights', 'best.pt'),
    "YOLO 10 epochs": os.path.join(MODELS_DIR, 'train10epochs', 'weights', 'best.pt'),
    "YOLO 20 epochs": os.path.join(MODELS_DIR, 'train20epochs', 'weights', 'best.pt'),
}

THRESHOLD = 60


def detect_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 5000:
            x, y, w, h = cv2.boundingRect(largest)
            return (x, y, x + w, y + h)
    return None


def _put_label(image, title, prediction, certainty):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, title, (8, 22), font, 0.6, (200, 200, 200), 2)
    if certainty >= THRESHOLD:
        text = f"{prediction} ({certainty}%)"
        (tw, _), _ = cv2.getTextSize(text, font, 0.8, 2)
        x = image.shape[1] - tw - 10
        y = image.shape[0] - 10
        cv2.putText(image, text, (x, y), font, 0.8, (255, 255, 255), 2)


def _method_panel(frame):
    payload = MethodPayload(image=frame)
    prediction, certainty = Method.classify(payload, custom_model_path=ROOT_DIR)

    processed = Method.process_image(payload)
    if processed.dtype != np.uint8:
        processed = np.clip(processed, 0, 255).astype(np.uint8)
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    panel = frame.copy()

    thumb_h = 120
    thumb_w = int(processed.shape[1] * (thumb_h / processed.shape[0]))
    thumbnail = cv2.resize(processed, (thumb_w, thumb_h))
    if len(thumbnail.shape) == 2:
        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_GRAY2BGR)
    panel[30:30 + thumbnail.shape[0], 0:thumbnail.shape[1]] = thumbnail

    _put_label(panel, "Method (SVM)", prediction.name, certainty)
    return panel


def _yolo_panel(model, title, frame):
    results = model.predict(frame, verbose=False)
    result = results[0]
    top_id = result.probs.top1
    gesture = Gesture10(int(result.names[top_id]))
    certainty = int(result.probs.top1conf.item() * 100)

    panel = frame.copy()
    _put_label(panel, title, gesture.name, certainty)
    return panel


def camera_test():
    yolo_models = {name: YOLO(path) for name, path in YOLO_MODELS.items()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    time.sleep(0.5)

    frame_count = 0
    current_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        frame_count += 1
        if frame_count % 2 == 0:
            current_frame = frame

        if current_frame is None:
            continue

        bbox = detect_hand(current_frame)
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        panel_method = _method_panel(current_frame)

        yolo_names = list(yolo_models.keys())
        panel_yolo3  = _yolo_panel(yolo_models[yolo_names[0]], yolo_names[0], current_frame)
        panel_yolo10 = _yolo_panel(yolo_models[yolo_names[1]], yolo_names[1], current_frame)
        panel_yolo20 = _yolo_panel(yolo_models[yolo_names[2]], yolo_names[2], current_frame)

        top_row    = np.hstack([panel_method, panel_yolo3])
        bottom_row = np.hstack([panel_yolo10, panel_yolo20])
        grid = np.vstack([top_row, bottom_row])

        cv2.imshow("Gesture Classifier Comparison", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_test()
