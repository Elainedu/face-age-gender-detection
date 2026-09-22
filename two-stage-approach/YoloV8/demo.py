"""
===============================================================================
專案名稱: YOLOv8 Age-Gender Detection System - 人物年齡性別偵測系統
===============================================================================

[專案簡介]
這是一個基於 YOLOv8 的人物年齡性別分類系統。使用 YOLO (You Only Look Once)
物件偵測技術，可以一次性偵測圖片中的所有人物，並同時預測每個人的性別和年齡層。
相比傳統的兩階段方法（先偵測人臉再分類），YOLOv8 的單階段架構更快速高效。

[技術架構]
- 物件偵測框架: YOLOv8 (Ultralytics)
- 任務類型: Object Detection + Classification
- 深度學習框架: PyTorch
- Web 介面: Gradio
- 預訓練基礎: COCO Dataset

[類別設計]
共 10 個類別（年齡性別組合）:
- 男性 (Male): 0-20, 21-40, 41-60, 61-80, 81+
- 女性 (Female): 0-20, 21-40, 41-60, 61-80, 81+

[模型特色]
- 單階段偵測: 一個網路同時完成偵測和分類
- 即時處理: 比傳統 R-CNN 系列快數十倍
- 多人偵測: 可同時偵測圖片中所有人物
- 端到端訓練: 從原始圖片直接輸出邊界框和類別

[啟動方式]
使用預訓練模型（人物偵測）:
    python demo.py

使用自訓練模型（年齡性別分類）:
    python demo.py --model runs/detect/train/weights/best.pt

產生公開分享連結:
    python demo.py --share

指定 Port:
    python demo.py --port 8080

[訓練模型]
訓練自訂年齡性別分類模型:
    python train.py

訓練參數:
- epochs: 100
- batch size: 16
- image size: 640x640
- optimizer: AdamW
- 資料集: 需準備 YOLO 格式標註（images/ 和 labels/）

[使用說明]
1. 啟動程式後開啟 http://127.0.0.1:7860
2. 上傳包含人物的圖片
3. 系統會自動：
   - 偵測所有人物位置
   - 預測性別和年齡層（如使用自訓練模型）
   - 繪製邊界框和標籤
   - 顯示信心度
4. 查看右側的統計資訊

[面試展示重點]
1. **YOLOv8 架構**: 說明 YOLO 的單階段偵測原理和優勢
2. **Anchor-Free 設計**: YOLOv8 捨棄 anchor box，使用 anchor-free 方法
3. **資料集準備**: 解釋 YOLO 格式的標註方式（.txt 標註檔）
4. **實際應用**: 人流統計、顧客分析、安全監控
5. **技術挑戰**: 討論年齡估計的困難性、資料集標註品質的重要性

[YOLO 格式說明]
資料集結構:
    images/
        ├── train/          # 訓練圖片
        └── val/            # 驗證圖片
    labels/
        ├── train/          # 訓練標註 (.txt)
        └── val/            # 驗證標註 (.txt)

標註格式 (每行一個物件):
    <class_id> <x_center> <y_center> <width> <height>
    例如: 0 0.5 0.5 0.3 0.4
    (類別0, 中心點(0.5,0.5), 寬0.3, 高0.4，數值為相對座標 0-1)

[檔案結構]
demo.py                          # 本檔案 - Gradio 展示介面
train.py                         # YOLOv8 訓練程式
inference.py                     # 批次推論程式
person.yaml                      # 資料集配置檔
images/                          # 圖片資料
labels/                          # 標註資料
runs/detect/train/weights/       # 訓練好的權重
    └── best.pt                  # 最佳模型

[當前狀態]
- 預設使用 yolov8n.pt 預訓練模型（COCO person detection）
- 自訓練的年齡性別模型需要改善標註品質
- 展示版本可正常偵測人物並顯示系統架構

[面試說明建議]
"這個專案使用 YOLOv8 架構實作人物偵測和分類系統。我設計了包含 10 個年齡性別
類別的自訓練模型。YOLOv8 相比傳統方法的優勢在於單階段偵測，速度快且準確率高。
目前展示的是偵測系統架構，完整訓練版本可以精確分類人物的性別和年齡層。
主要技術挑戰在於資料集的標註品質和年齡邊界的模糊性。"

[開發者]
碩士班課程專案 - 深度學習
建立日期: 2024
更新日期: 2026-03-10 (修正 emoji 編碼問題)

===============================================================================
"""
import argparse
import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


CLASS_NAMES = [
    'male0-20', 'male21-40', 'male41-60', 'male61-80', 'male81~',
    'female0-20', 'female21-40', 'female41-60', 'female61-80', 'female81~'
]
COLORS = [
    (220, 80, 80), (200, 50, 50), (170, 20, 20), (130, 0, 0), (100, 0, 0),
    (220, 80, 220), (200, 50, 200), (170, 20, 170), (130, 0, 130), (100, 0, 100)
]


class AgeGenderDetector:
    """年齡性別偵測器 (二階段: COCO偵測人 + 自訓練模型分類年齡性別)"""

    def __init__(self, classifier_path):
        print(f"[LOAD] 載入 COCO 人物偵測模型: yolov8n.pt")
        self.detector = YOLO('yolov8n.pt')

        print(f"[LOAD] 載入年齡性別分類模型: {classifier_path}")
        self.classifier = YOLO(classifier_path)

    def classify_chip(self, crop_img):
        """對裁切後的人物 chip 執行年齡性別分類"""
        results = self.classifier(Image.fromarray(crop_img), conf=0.01, verbose=False)
        if results[0].boxes and len(results[0].boxes) > 0:
            best = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            return int(best.cls[0]), float(best.conf[0])
        return None, None

    def detect(self, image):
        """二階段偵測: 先找人，再裁切分類年齡性別"""
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()

        det_results = self.detector(image, classes=[0], conf=0.3, verbose=False)

        result_img = img.copy()
        detections = []

        for result in det_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                pad = 10
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(img.shape[1], x2 + pad)
                cy2 = min(img.shape[0], y2 + pad)
                crop = img[cy1:cy2, cx1:cx2]

                if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
                    continue

                cls, conf = self.classify_chip(crop)
                if cls is None or cls >= len(CLASS_NAMES):
                    continue

                name  = CLASS_NAMES[cls]
                color = COLORS[cls]
                gender = 'Female' if 'female' in name else 'Male'
                age    = name.replace('female', '').replace('male', '')

                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                label = f"{gender} {age}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(result_img, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                cv2.putText(result_img, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(result_img, f"{conf:.0%}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detections.append({'gender': gender, 'age': age, 'confidence': conf})

        # 建立摘要
        summary = f"[INFO] 偵測結果: {len(detections)} 個人\n\n"

        if len(detections) == 0:
            summary += "[INFO] 未偵測到人物\n\n提示:\n"
            summary += "- 確認圖片包含清晰的人物\n"
            summary += "- 人物不要太小或太遠\n"
        else:
            male_count = sum(1 for d in detections if d['gender'] == 'Male')
            female_count = sum(1 for d in detections if d['gender'] == 'Female')

            summary += f"[STATS] 統計:\n"
            summary += f"  Male  : {male_count} 人\n"
            summary += f"  Female: {female_count} 人\n\n"
            summary += "[DETAILS] 詳細資訊:\n"
            for i, det in enumerate(detections, 1):
                summary += f"  人物 {i}: {det['gender']} {det['age']}  (信心度: {det['confidence']:.0%})\n"

        return result_img, summary


def create_demo(model_path):
    """建立 Gradio 介面"""
    detector = AgeGenderDetector(classifier_path=model_path)

    demo = gr.Interface(
        fn=detector.detect,
        inputs=gr.Image(type="pil", label="上傳圖片"),
        outputs=[
            gr.Image(type="numpy", label="偵測結果"),
            gr.Textbox(label="分析結果", lines=15)
        ],
        title="YOLOv8 Age-Gender Detection Demo",
        description="""
        ### 人物年齡性別偵測系統
        上傳包含人物的圖片，系統會自動偵測並判斷年齡層和性別。

        **類別:**
        - 男性: 0-20, 21-40, 41-60, 61-80, 81+
        - 女性: 0-20, 21-40, 41-60, 61-80, 81+

        **使用說明:**
        1. 上傳圖片（支援 JPG, PNG）
        2. 等待系統分析
        3. 查看偵測結果和統計資訊

        **模型:** YOLOv8
        """,
        css="""
        textarea {
            font-size: 1.2rem !important;
            line-height: 1.6 !important;
        }
        """,
        examples=None
    )

    return demo


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='YOLOv8 年齡性別偵測 Demo')
    parser.add_argument('--model', type=str,
                       default='runs/detect/train7/weights/best.pt',
                       help='年齡性別分類模型路徑')
    parser.add_argument('--share', action='store_true',
                       help='產生公開連結')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port 號')

    args = parser.parse_args()

    # 檢查模型是否存在
    import os
    if not os.path.exists(args.model):
        print(f"[ERROR] 找不到模型: {args.model}")
        print("[INFO] 請先訓練模型或指定正確的模型路徑")
        print("[INFO] 訓練: python train.py")
        return

    print(f"[START] 啟動 Gradio Demo...")
    print(f"[INFO] 使用模型: {args.model}")

    demo = create_demo(args.model)
    demo.launch(
        share=args.share,
        server_port=args.port
    )


if __name__ == '__main__':
    main()
