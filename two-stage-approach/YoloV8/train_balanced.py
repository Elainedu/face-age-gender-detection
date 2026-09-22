from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # nano 版，比 yolov8m 快 3 倍

results = model.train(
    data='person_balanced.yaml',
    epochs=10,
    batch=16,
    imgsz=640,
    patience=5,
)
