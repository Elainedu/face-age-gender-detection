"""
Face Age-Gender Detection - Inference Script
使用訓練好的 YOLOv8 模型進行推理

用法:
    python inference.py --image test.jpg
    python inference.py --image test.jpg --model results/train/weights/best.pt
"""
import argparse
from ultralytics import YOLO
import cv2
import os


# 類別名稱對應
CLASS_NAMES = {
    0: 'Male 0-20',
    1: 'Male 21-40',
    2: 'Male 41-60',
    3: 'Male 61-80',
    4: 'Male 81+',
    5: 'Female 0-20',
    6: 'Female 21-40',
    7: 'Female 41-60',
    8: 'Female 61-80',
    9: 'Female 81+'
}


def predict_image(model_path, image_path, conf=0.25, save_dir='results'):
    """
    對單張圖片進行預測

    Args:
        model_path: 模型路徑
        image_path: 圖片路徑
        conf: 信心度閾值
        save_dir: 結果儲存目錄
    """
    print(f"📦 載入模型: {model_path}")
    model = YOLO(model_path)

    print(f"🔍 推理圖片: {image_path}")

    # 預測
    results = model.predict(
        source=image_path,
        conf=conf,
        save=True,
        project=save_dir,
        name='inference',
        exist_ok=True
    )

    # 顯示結果
    result = results[0]
    boxes = result.boxes

    print(f"\n📊 偵測結果:")
    print(f"   偵測到 {len(boxes)} 張人臉")

    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = CLASS_NAMES.get(cls, f'Class {cls}')

        print(f"   人臉 {i+1}: {label} (信心度: {conf:.2f})")

    # 輸出路徑
    output_path = os.path.join(save_dir, 'inference', os.path.basename(image_path))
    print(f"\n💾 結果已儲存至: {output_path}")

    return results


def predict_batch(model_path, image_dir, conf=0.25, save_dir='results'):
    """
    批次預測多張圖片

    Args:
        model_path: 模型路徑
        image_dir: 圖片資料夾路徑
        conf: 信心度閾值
        save_dir: 結果儲存目錄
    """
    print(f"📦 載入模型: {model_path}")
    model = YOLO(model_path)

    print(f"🔍 批次推理: {image_dir}")

    # 批次預測
    results = model.predict(
        source=image_dir,
        conf=conf,
        save=True,
        project=save_dir,
        name='batch_inference',
        exist_ok=True
    )

    print(f"\n✅ 批次推理完成！")
    print(f"   處理 {len(results)} 張圖片")
    print(f"💾 結果已儲存至: {save_dir}/batch_inference/")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8 人臉年齡性別偵測推理'
    )
    parser.add_argument('--image', type=str, default=None,
                       help='單張圖片路徑')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='圖片資料夾路徑 (批次處理)')
    parser.add_argument('--model', type=str,
                       default='results/train/weights/best.pt',
                       help='模型路徑')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='信心度閾值')
    parser.add_argument('--save-dir', type=str, default='results',
                       help='結果儲存目錄')

    args = parser.parse_args()

    # 檢查模型
    if not os.path.exists(args.model):
        print(f"❌ 找不到模型: {args.model}")
        print(f"請先執行訓練: python train.py")
        return

    # 單張圖片或批次處理
    if args.image:
        predict_image(args.model, args.image, args.conf, args.save_dir)
    elif args.image_dir:
        predict_batch(args.model, args.image_dir, args.conf, args.save_dir)
    else:
        print("❌ 請指定 --image 或 --image-dir")
        return

    print("🎉 完成！")


if __name__ == '__main__':
    main()
