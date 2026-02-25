# Assignment 02 — YOLOv8 Face Detection with Age & Gender Classification

## Task Overview

Train YOLOv8m to detect faces and simultaneously classify them into **10 age-gender categories** (5 age groups × 2 genders). This builds on the standard YOLO object detection pipeline, using custom-labeled face chip images as training data.

**Task type**: Multi-class object detection
**Model**: YOLOv8m (medium variant)
**Platform**: Local GPU + Google Colab

---

## Class Definitions

Defined in `YoloV8/person.yaml`:

| Class ID | Label | Class ID | Label |
|---|---|---|---|
| 0 | male, age 0–20 | 5 | female, age 0–20 |
| 1 | male, age 21–40 | 6 | female, age 21–40 |
| 2 | male, age 41–60 | 7 | female, age 41–60 |
| 3 | male, age 61–80 | 8 | female, age 61–80 |
| 4 | male, age 81+ | 9 | female, age 81+ |

---

## Dataset

- **Source images**: IMDB-WIKI face chip dataset — cropped face images with filenames encoding age, gender, and timestamp
  - Filename format: `{age}_{gender}_{other}_{timestamp}.jpg.chip.jpg`
  - Example: `60_1_0_20170110141405608.jpg.chip.jpg` → age=60, gender=1(female)
- **Labels folder**: `YoloV8/labels/` (YOLO format `.txt` files)
- **YAML config**: `YoloV8/person.yaml`
  ```yaml
  train: 'G:/YoloV8/images'
  val:   'G:/YoloV8/images/val'
  nc: 10
  names: ['male0-20','male21-40','male41-60','male61-80','male81~',
          'female0-20','female21-40','female41-60','female61-80','female81~']
  ```

---

## Training

`YoloV8/train-yolov8m-face-detection.ipynb` / `train-yolov8m-face-detection.py`:

```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')          # load pretrained YOLOv8m
results = model.train(
    data='person.yaml',
    epochs=3,
    batch=8
)
```

- Base model: `yolov8m.pt` (pretrained on COCO)
- Fine-tuned for 3 epochs with custom 10-class face labels
- Batch size 8 (memory constrained)

---

## Data Labeling Pipeline

A separate labeling tool was developed in the `yolov8-face-age-gender-labeling/` folder (see its own README).
Labels in `pre/` folder contain age and gender extracted from filenames alongside YOLO bounding boxes.

---

## File Structure

```
assignment-02-yolov8-face-detection/
├── YoloV8/
│   ├── train-yolov8m-face-detection.ipynb   # training notebook
│   ├── train-yolov8m-face-detection.py      # equivalent Python script
│   ├── person.yaml                          # dataset config (10 classes)
│   └── labels/
│       ├── train/   # YOLO-format label files for training images
│       └── val/     # YOLO-format label files for validation images
├── yolov8-face-age-gender-labeling/
│   └── 123/
│       └── t2.ipynb                         # batch labeling notebook
├── yolov8-face-dataset-raw/
│   └── 123/pre/    # auto-labeled face chips (age_gender_bbox per file)
├── assignment-02.zip                        # submission archive
└── YoloV8.zip                               # dataset backup
```

---

## How to Run

1. Install dependencies:
   ```bash
   pip install ultralytics
   ```
2. Place images into `YoloV8/images/train/` and `YoloV8/images/val/`
3. Update paths in `person.yaml` to match your local drive
4. Run training:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8m.pt')
   model.train(data='person.yaml', epochs=3, batch=8)
   ```
5. Results saved under `runs/detect/train*/`

---

## Key Observations & Issues

- **Dataset path**: YAML pointed to `G:/YoloV8/images` (external drive). Update path before running on a different machine.
- **Class imbalance**: Some age groups (e.g., 81+) have very few samples; consider oversampling or weighted loss.
- **3 epochs** is very short for a 10-class detector — treat as a proof-of-concept; more epochs and data would improve mAP.
- Label files from `pre/` use a custom format (`classID_gender_age x y w h`) that must be converted to standard YOLO format before training.
