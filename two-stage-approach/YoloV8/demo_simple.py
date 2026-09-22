"""
===============================================================================
專案名稱: YOLOv8 Person Detection Demo - 人物偵測展示（簡化版）
===============================================================================

[專案簡介]
這是 YOLOv8 人物偵測的簡化展示版本。使用預訓練的 COCO 模型進行人物偵測，
無需自行訓練，開箱即用。適合快速展示 YOLOv8 的偵測能力和系統架構。

[與完整版的差異]
demo_simple.py (本檔案):
- 使用 COCO 預訓練模型（yolov8n.pt）
- 只偵測人物（person class）
- 無需訓練資料集
- 快速啟動展示

demo.py (完整版):
- 支援自訓練的年齡性別分類模型
- 10 個年齡性別類別
- 需要準備訓練資料
- 完整功能展示

[適用場景]
簡化版適合:
1. 面試快速展示 YOLOv8 架構理解
2. 展示物件偵測的基本流程
3. 驗證系統環境配置
4. 無訓練資料時的替代方案

[技術架構]
- 物件偵測: YOLOv8n (nano 版本，最輕量)
- 預訓練資料: COCO Dataset (80 個類別)
- 使用類別: Person (class_id = 0)
- 深度學習框架: PyTorch (via Ultralytics)
- Web 介面: Gradio

[YOLOv8 系列模型]
模型      | 參數量   | 速度    | 準確率  | 適用場景
---------|---------|---------|---------|-------------
YOLOv8n  | 3.2M    | 最快     | 中等    | 邊緣裝置、即時應用
YOLOv8s  | 11.2M   | 快      | 良好    | 平衡選擇
YOLOv8m  | 25.9M   | 中等    | 很好    | 一般應用
YOLOv8l  | 43.7M   | 慢      | 優秀    | 高精度需求
YOLOv8x  | 68.2M   | 最慢    | 最佳    | 離線處理

本專案使用 YOLOv8n（nano）：最快速，適合展示

[COCO Dataset]
COCO (Common Objects in Context) 是最常用的物件偵測資料集:
- 80 個物件類別
- class 0 = person（人）
- 包含：動物、交通工具、家具、食物等
- YOLOv8 預訓練模型基於 COCO

[啟動方式]
基本啟動（自動下載 yolov8n.pt）:
    python demo_simple.py

指定 Port:
    python demo_simple.py  # 預設 7860

產生公開連結:
    需修改程式碼中的 share=True

[使用說明]
1. 執行程式（首次會自動下載模型 ~6MB）
2. 開啟 http://127.0.0.1:7860
3. 上傳包含人物的圖片
4. 系統會：
   - 偵測所有人物
   - 繪製綠色邊界框
   - 顯示信心度
   - 統計人數
5. 查看分析結果

[偵測參數]
```python
results = self.model(image, classes=[0], conf=0.25, verbose=False)
```
- classes=[0]: 只偵測 person 類別
- conf=0.25: 信心度閾值 25%（低於此值的偵測會被過濾）
- verbose=False: 不顯示詳細日誌

[面試展示技巧]
使用本檔案展示時的說明重點:

1. **快速展示架構**:
   "這是使用 YOLOv8 預訓練模型的人物偵測展示。YOLOv8 是目前最先進的
   單階段物件偵測架構之一，相比 R-CNN 系列速度快數十倍。"

2. **說明設計思路**:
   "我基於這個架構設計了年齡性別分類系統，定義了 10 個類別。
   目前展示的是偵測基礎功能，完整版本可以進行年齡性別分類。"

3. **技術理解**:
   "YOLOv8 採用 anchor-free 設計，使用 C2f 模組和 PAN-FPN 架構。
   預訓練模型在 COCO 資料集上達到 mAP 50-95: 37.3%。"

4. **實際應用**:
   "人物偵測可應用在人流統計、安全監控、人數計算等場景。
   如果加上年齡性別分類，可用於顧客分析、行銷研究。"

[優勢說明]
為何使用簡化版展示:
1. 無需準備訓練資料 - 節省時間
2. 預訓練模型穩定可靠 - 確保展示成功
3. 聚焦架構理解 - 而非資料處理
4. 快速啟動 - 適合時間有限的展示

[與自訓練模型的對比]
面試時可以說明:

預訓練模型展示（本檔案）:
- 優點: 快速、穩定、展示架構
- 缺點: 無法展示完整訓練流程

自訓練模型展示（demo.py）:
- 優點: 展示完整能力、自訂類別
- 缺點: 需要高品質訓練資料

[實際專案經驗分享]
"在實作過程中，我發現年齡分類的主要挑戰在於：
1. 年齡邊界模糊（40歲和41歲視覺差異極小）
2. 資料集標註品質影響很大
3. 需要大量不同年齡層的訓練樣本
因此預訓練模型可作為初步驗證，確認系統架構後再進行完整訓練。"

[檔案結構]
demo_simple.py                   # 本檔案 - 簡化展示版
demo.py                          # 完整版（支援自訓練模型）
train.py                         # 訓練程式
person.yaml                      # 資料集配置

[技術細節]
偵測流程:
1. 圖片輸入 → YOLOv8 主幹網路 (backbone)
2. 特徵提取 → C2f 模組處理
3. 特徵融合 → PAN-FPN 多尺度融合
4. 偵測頭 → 輸出邊界框和類別機率
5. NMS 後處理 → 移除重複偵測

信心度計算:
- 綜合考慮：物件存在機率 × 類別機率
- 預設閾值 0.25 = 25% 信心度
- 可調整以平衡準確率和召回率

[開發者]
碩士班課程專案 - 深度學習
建立日期: 2024
用途: 快速展示和架構驗證

===============================================================================
"""
import argparse
import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

class PersonDetector:
    """人物偵測器（使用預訓練 COCO 模型）"""

    def __init__(self, model_path='yolov8n.pt'):
        """初始化偵測器"""
        print(f"[LOAD] 載入模型: {model_path}")
        self.model = YOLO(model_path)

    def detect(self, image):
        """偵測圖片中的人物"""
        # Gradio 傳入的是 PIL Image，直接用就好
        # 執行偵測（不用轉換格式，YOLOv8 可以直接處理 PIL Image）
        results = self.model(image, classes=[0], conf=0.25, verbose=False)  # class 0 = person

        # 取得原始圖片的 numpy array 來畫框
        img = np.array(image)

        # 繪製結果
        result_img = img.copy()
        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # 畫框
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # 畫標籤
                label = f"Person"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(result_img, (x1, y1-text_h-10), (x1+text_w, y1), (0, 255, 0), -1)
                cv2.putText(result_img, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # 信心度
                conf_text = f"{conf:.2%}"
                cv2.putText(result_img, conf_text, (x1, y2+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                detections.append({'confidence': conf})

        # 文字摘要
        summary = f"[INFO] 偵測結果: {len(detections)} 個人\n\n"

        if len(detections) == 0:
            summary += "[INFO] 未偵測到人物\n"
            summary += "\n提示:\n"
            summary += "- 確認圖片中有清晰可見的人物\n"
            summary += "- 人物不要太小或太遠\n"
            summary += "- 嘗試上傳其他圖片測試\n"
        else:
            summary += "[DETAILS] 詳細資訊:\n"
            for i, det in enumerate(detections, 1):
                summary += f"  人物 {i}: 信心度 {det['confidence']:.2%}\n"

            summary += f"\n[STATS] 統計: 共 {len(detections)} 人"

        return result_img, summary


def main():
    """主程式"""
    detector = PersonDetector('yolov8n.pt')

    demo = gr.Interface(
        fn=detector.detect,
        inputs=gr.Image(type="pil", label="上傳圖片"),
        outputs=[
            gr.Image(type="numpy", label="偵測結果"),
            gr.Textbox(label="分析結果", lines=12)
        ],
        title="YOLOv8 Person Detection Demo",
        description="""
        ### 人物偵測系統展示
        使用 YOLOv8 模型進行人物偵測

        **功能:**
        - 自動偵測圖片中的人物
        - 標示人物位置
        - 顯示偵測信心度

        **使用說明:**
        1. 上傳包含人物的圖片
        2. 系統自動分析
        3. 查看偵測結果

        **技術:** YOLOv8 (COCO pre-trained)
        **應用場景:** 人流統計、安全監控、人數計算
        """,
        css="""
        textarea {
            font-size: 1.2rem !important;
            line-height: 1.6 !important;
        }
        """,
    )

    print("[START] 啟動展示介面...")
    demo.launch(share=False, server_port=7860)


if __name__ == '__main__':
    main()
