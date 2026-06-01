import shutil
import os
from ultralytics import YOLO

folder_wynikow = 'runs/classify/'

if os.path.exists(folder_wynikow):
    shutil.rmtree(folder_wynikow)

model = YOLO('yolov8n-cls.pt')
results1 = model.train(
    data='yolo_gestures_cls',
    epochs=3,
    imgsz=128,
    name='train3epochs'
)
model = YOLO('yolov8n-cls.pt')
results2 = model.train(
    data='yolo_gestures_cls',
    epochs=10,
    imgsz=128,
    name='train10epochs'
)
model = YOLO('yolov8n-cls.pt')
results3 = model.train(
    data='yolo_gestures_cls',
    epochs=20,
    imgsz=128,
    name='train20epochs'
)