from ultralytics import YOLO
 

 
# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8m.pt')
 
# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='person.yaml', epochs=3,batch=8)
 
