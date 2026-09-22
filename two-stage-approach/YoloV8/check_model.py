from ultralytics import YOLO
import os

model = YOLO('runs/detect/train6/weights/best.pt')
print("類別:", model.names)

test_imgs = [f for f in os.listdir('test_image') if f.endswith(('.jpg', '.png'))]

for img in test_imgs[:3]:
    path = f'test_image/{img}'
    results = model(path, conf=0.01, verbose=False)
    boxes = results[0].boxes
    print(f"\n{img}: 偵測到 {len(boxes)} 個")
    for box in boxes:
        name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"  {name}: {conf:.2%}")
