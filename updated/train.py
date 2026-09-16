"""
Face Age-Gender Detection - YOLOv8 Training Script
訓練 YOLOv8m 進行人臉偵測 + 年齡性別分類 (10 類別)

用法:
    python train.py
    python train.py --epochs 10 --batch 16
"""
import argparse
from ultralytics import YOLO
import os


def train_yolov8(
    model_name='yolov8m.pt',
    data_yaml='configs/person.yaml',
    epochs=3,
    batch=8,
    imgsz=640,
    device=0
):
    """
    訓練 YOLOv8 模型

    Args:
        model_name: 預訓練模型名稱 (yolov8n/s/m/l/x)
        data_yaml: 資料集設定檔路徑
        epochs: 訓練輪數
        batch: Batch size
        imgsz: 圖片大小
        device: GPU 裝置編號 (0, 1, ...) 或 'cpu'
    """
    print("🚀 開始訓練 YOLOv8 人臉年齡性別偵測模型...")
    print(f"📊 設定:")
    print(f"   模型: {model_name}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch}")
    print(f"   Image Size: {imgsz}")
    print(f"   Device: {device}")

    # 載入模型
    model = YOLO(model_name)

    # 訓練
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project='results',
        name='train',
        exist_ok=True,
        patience=50,
        save=True,
        plots=True,
        verbose=True
    )

    print("✅ 訓練完成！")
    print(f"📁 結果儲存於: results/train/")
    print(f"🏆 最佳模型: results/train/weights/best.pt")

    return results


def validate_model(model_path, data_yaml='configs/person.yaml'):
    """驗證模型"""
    print(f"🔍 驗證模型: {model_path}")

    model = YOLO(model_path)
    results = model.val(data=data_yaml)

    print(f"✅ 驗證完成！")
    print(f"📊 mAP50: {results.box.map50:.4f}")
    print(f"📊 mAP50-95: {results.box.map:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='訓練 YOLOv8 人臉年齡性別偵測模型'
    )
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt',
                               'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 模型大小')
    parser.add_argument('--data', type=str, default='configs/person.yaml',
                       help='資料集 YAML 設定檔')
    parser.add_argument('--epochs', type=int, default=3,
                       help='訓練 epochs')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='圖片大小')
    parser.add_argument('--device', type=str, default='0',
                       help='裝置 (0, 1, ... 或 cpu)')
    parser.add_argument('--validate', action='store_true',
                       help='訓練後進行驗證')

    args = parser.parse_args()

    # 訓練
    results = train_yolov8(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )

    # 驗證
    if args.validate:
        best_model = 'results/train/weights/best.pt'
        if os.path.exists(best_model):
            validate_model(best_model, args.data)

    print("🎉 完成！")


if __name__ == '__main__':
    main()
