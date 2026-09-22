from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8m.pt')

# Train the model
results = model.train(data='person.yaml', epochs=20, batch=8, imgsz=640)
 
