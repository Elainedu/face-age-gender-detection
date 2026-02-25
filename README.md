# Face Age-Gender Detection

> 使用 YOLOv8m 進行人臉偵測與年齡性別分類 (10 類別)

## 📌 專案概述

本專案訓練 YOLOv8m 模型同時完成人臉偵測和年齡性別分類任務。將5個年齡組 × 2個性別組合成 10 個類別，可在單次推理中同時得到人臉位置與屬性資訊。

## 🎯 專案目標

- 多類別物件偵測 (10 classes)
- 自動標註工具開發
- 端到端的人臉屬性分析系統

## 🛠️ 技術棧

- **Framework**: Ultralytics YOLOv8
- **模型**: YOLOv8m (medium variant)
- **Dataset**: IMDB-WIKI Face Chips (自動標註)
- **類別**: 10 classes (5 age groups × 2 genders)

## 📊 類別定義

| Class ID | 類別 | Class ID | 類別 |
|----------|------|----------|------|
| 0 | 👨 男性 0-20歲 | 5 | 👩 女性 0-20歲 |
| 1 | 👨 男性 21-40歲 | 6 | 👩 女性 21-40歲 |
| 2 | 👨 男性 41-60歲 | 7 | 👩 女性 41-60歲 |
| 3 | 👨 男性 61-80歲 | 8 | 👩 女性 61-80歲 |
| 4 | 👨 男性 81歲以上 | 9 | 👩 女性 81歲以上 |

## 🚀 快速開始

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 自動標註資料

```bash
# 從 IMDB-WIKI 人臉檔名自動產生 YOLO 標註
python auto_label.py --input data/raw_images --output data/labels

# 測試少量資料
python auto_label.py --input data/raw_images --output data/labels --limit 100
```

### 訓練模型

```bash
# 基本訓練
python train.py

# 自訂參數
python train.py --model yolov8m.pt --epochs 10 --batch 16

# 訓練並驗證
python train.py --epochs 5 --validate
```

### 推理

```bash
# 單張圖片
python inference.py --image test.jpg

# 批次處理
python inference.py --image-dir test_images/

# 調整信心度
python inference.py --image test.jpg --conf 0.5
```

### 啟動 Demo

```bash
python demo.py

# 或指定模型
python demo.py --model results/train/weights/best.pt
```

## 📁 專案結構

```
face-age-gender-detection/
├── train.py              # 訓練腳本
├── inference.py          # 推理腳本
├── demo.py              # Gradio 互動介面
├── auto_label.py        # 自動標註工具 ⭐
├── requirements.txt     # 依賴套件
├── notebooks/           # Jupyter notebooks
├── configs/
│   └── person.yaml      # YOLOv8 資料集設定 (10 classes)
├── data/
│   ├── raw_images/      # 原始 IMDB-WIKI face chips
│   └── labels/          # YOLO 格式標註檔 (.txt)
└── results/
    └── train/
        └── weights/
            └── best.pt  # 最佳模型
```

## 🔬 方法論

### 資料準備流程

1. **資料來源**: IMDB-WIKI Face Chips
   - 檔名格式: `{age}_{gender}_{other}_{timestamp}.jpg.chip.jpg`
   - 範例: `60_1_0_20170110141405608.jpg.chip.jpg` → 60歲女性

2. **自動標註**: `auto_label.py`
   ```python
   # 從檔名解析
   age = 60, gender = 1 (female)

   # 判斷年齡組
   age_group = 2  # 41-60歲

   # 計算類別 ID
   class_id = age_group + 5 = 7  # 女性 41-60歲
   ```

3. **YOLO 格式輸出**
   ```
   7 0.5 0.5 0.9 0.9
   # class_id x_center y_center width height (normalized)
   ```

### 訓練配置

```yaml
# configs/person.yaml
train: 'path/to/images/train'
val: 'path/to/images/val'
nc: 10
names: ['male0-20','male21-40','male41-60','male61-80','male81~',
        'female0-20','female21-40','female41-60','female61-80','female81~']
```

**訓練參數:**
- Base Model: yolov8m.pt (COCO pretrained)
- Epochs: 3 (概念驗證)
- Batch Size: 8
- Image Size: 640×640

## 📈 實驗結果

**訓練成果:**
- 3 epochs 為快速概念驗證
- 建議完整訓練: 10-20 epochs
- mAP 指標視資料集品質而定

**主要挑戰:**
- 類別不平衡 (81+ 歲樣本較少)
- 年齡標註來自檔名，可能有誤差

## 💡 關鍵洞察

1. **自動標註工具**: 省去手動標註的大量時間，從檔名即可生成標籤
2. **端到端流程**: 從資料處理、訓練到部署的完整pipeline
3. **YOLOv8 優勢**: 快速收斂，易於部署
4. **實際應用**: 可用於人群分析、商業智能等場景

## 🎨 Demo 功能

Gradio 介面提供:
- 📤 圖片上傳
- 👥 多人臉同時偵測
- 🎯 年齡性別分類
- 📊 統計資訊（性別比例、年齡分佈）
- 🎨 彩色標註框（不同類別不同顏色）
- ⚙️ 可調整信心度閾值

## 📦 資料集說明

**IMDB-WIKI Face Chips Dataset**
- 公開人臉資料集
- 檔名包含年齡和性別資訊
- 已裁切的人臉圖片（face chips）

**資料集準備步驟:**
1. 下載 IMDB-WIKI face chips
2. 執行自動標註工具
3. 將圖片和標註分別放入 `images/` 和 `labels/`
4. 確保檔名對應（image.jpg → image.txt）

## 🔗 參考資料

- YOLOv8: https://github.com/ultralytics/ultralytics
- IMDB-WIKI Dataset: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

## 📝 環境需求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (GPU 訓練，推薦)
- 16GB+ RAM (建議)

## 🏆 專案亮點

✅ 自製自動標註工具
✅ 10 類別多任務學習
✅ 完整的訓練推理流程
✅ 互動式 Gradio Demo
✅ 統計分析功能

---

*專案建立於 2023-2024 年度深度學習課程*
