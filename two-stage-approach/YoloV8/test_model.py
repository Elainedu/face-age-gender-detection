"""測試模型是否能正常推理"""
from ultralytics import YOLO
import os

# 載入模型
model_path = 'runs/detect/train/weights/best.pt'
print(f"[INFO] 載入模型: {model_path}")
model = YOLO(model_path)

# 測試圖片
test_images = []

# 找測試圖片
if os.path.exists('test_image'):
    for f in os.listdir('test_image'):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(os.path.join('test_image', f))

if not test_images:
    # 用訓練圖片測試
    for f in os.listdir('images/train')[:3]:
        if f.endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(os.path.join('images/train', f))

print(f"[INFO] 找到 {len(test_images)} 張測試圖片")

for img_path in test_images[:3]:  # 只測試前3張
    print(f"\n[TEST] 測試圖片: {img_path}")

    # 執行推理
    results = model(img_path, conf=0.1, verbose=False)

    # 顯示結果
    boxes = results[0].boxes
    print(f"  偵測到 {len(boxes)} 個物件")

    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]
        print(f"    物件 {i+1}: {class_name} (信心度: {conf:.2%})")

print("\n[DONE] 測試完成")
