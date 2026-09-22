"""快速測試 YOLOv8 是否正常運作"""
from ultralytics import YOLO
import os

print("[INFO] 測試 YOLOv8 預訓練模型...")

# 載入預訓練模型
model = YOLO('yolov8n.pt')
print("[OK] 模型載入成功")

# 找測試圖片
test_images = []
if os.path.exists('test_image'):
    for f in os.listdir('test_image'):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(os.path.join('test_image', f))

if not test_images:
    test_images = [os.path.join('images/train', f)
                   for f in os.listdir('images/train')[:2]
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"[INFO] 測試圖片: {len(test_images)} 張")

for img_path in test_images[:2]:
    print(f"\n[TEST] {img_path}")

    # 基本偵測（所有類別）
    results = model(img_path, verbose=False)

    print(f"  總共偵測到: {len(results[0].boxes)} 個物件")

    # 顯示所有偵測到的物件
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = results[0].names[cls]
        print(f"    - {class_name} (信心度: {conf:.2%})")

    # 只看 person (class 0)
    person_boxes = [b for b in results[0].boxes if int(b.cls[0]) == 0]
    print(f"  其中 person: {len(person_boxes)} 個")

print("\n[INFO] 請把上面的結果貼給我看")
