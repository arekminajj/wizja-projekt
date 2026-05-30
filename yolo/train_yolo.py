from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
results = model.train(
    data='yolo_gestures_cls',
    epochs=3,
    imgsz=224,
    batch=16
)
