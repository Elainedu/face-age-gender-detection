# YOLOv8 Face Age-Gender Auto-Labeling Tool

## Task Overview

An automated annotation pipeline that reads face chip images from the IMDB-WIKI dataset, uses YOLOv8 to detect the face bounding box, and extracts age/gender metadata from the filename to produce YOLO-format label files — eliminating manual annotation.

**Task type**: Data preprocessing / auto-labeling
**Model used for detection**: YOLOv8m (`yolov8m.pt`, pretrained on COCO)
**Platform**: Local Windows machine (CPU inference)

---

## Background

The IMDB-WIKI dataset distributes face chips with filenames encoding age, gender, and timestamp:

```
{age}_{gender}_{race}_{timestamp}.jpg.chip.jpg
```

Examples:
- `1_0_0_20161219140623097.jpg.chip.jpg` → age=1, gender=0 (male)
- `60_1_0_20170110141405608.jpg.chip.jpg` → age=60, gender=1 (female)

Since the labels are embedded in filenames, annotation only needs to:
1. Detect the face bounding box using YOLOv8
2. Parse age and gender from the filename
3. Write a YOLO-format `.txt` label file

---

## Development Process & Debugging

This tool went through several iterations as the YOLOv8 API changed between versions.

### Attempt 1 — `results[detection]['box']` (FAILED)

```python
for detection in results:
    bbox = detection['box']   # IndexError: too many indices
```

**Error**: `IndexError: too many indices for tensor of dimension 2`
YOLOv8 `predict()` returns a `Results` object list, not a dict-indexable list.

### Attempt 2 — `results.xyxy[0]` (FAILED)

```python
for detection in results[0].xyxy.numpy():   # AttributeError
```

**Error**: `'Results' object has no attribute 'xyxy'`
The old YOLOv5-style `.xyxy` attribute does not exist in YOLOv8.

### Attempt 3 — `results[0].boxes.data` (SUCCESS)

```python
for detection in results[0].boxes.data.numpy():
    bbox = detection[0:4].tolist()   # [x1, y1, x2, y2]
    class_id = int(detection[5])
```

This is the correct YOLOv8 API. `boxes.data` returns a tensor with columns:
`[x1, y1, x2, y2, confidence, class_id]`

---

## Final Working Pipeline

### Single image (`01-face-detection-label-extraction.ipynb`)

```python
from PIL import Image
from ultralytics import YOLO

yolo = YOLO('./yolov8m.pt')
filename = './1_0_0_20161219140623097.jpg.chip.jpg'

# Parse metadata from filename
parts = filename.split('_')
age    = parts[0].lstrip('./')
gender = parts[1]

# Detect face
img = Image.open(filename)
results = yolo.predict(img)

# Write YOLO label
for detection in results[0].boxes.data.numpy():
    x1, y1, x2, y2 = detection[0:4]
    cx = (x1 + x2) / 2 / img.width
    cy = (y1 + y2) / 2 / img.height
    w  = (x2 - x1) / img.width
    h  = (y2 - y1) / img.height
    label_line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
```

### Batch processing (`02-batch-face-detection-annotation.ipynb`)

Loops over all files in `dataset/` folder and saves annotated images + labels to `pre/`:

```python
input_folder  = './dataset/'
output_folder = './pre/'

for filename in os.listdir(input_folder):
    img = Image.open(os.path.join(input_folder, filename))
    results = yolo.predict(img)
    # ... save annotated image and label file to output_folder
```

---

## File Structure

```
yolov8-face-age-gender-labeling/
├── 01-face-detection-label-extraction.ipynb   # single-image debug notebook
├── 02-batch-face-detection-annotation.ipynb   # batch processing notebook
└── (yolov8m.pt)                               # model weight — download separately
```

Related output (in assignment-02):
```
assignment-02-yolov8-face-detection/
└── yolov8-face-dataset-raw/
    └── 123/pre/     # auto-labeled results (image + .txt per face chip)
```

---

## How to Run

1. Install dependencies:
   ```bash
   pip install ultralytics pillow
   ```
2. Download `yolov8m.pt` (auto-downloads on first YOLO call, or from ultralytics)
3. Place face chip images in `./dataset/`
4. Run `02-batch-face-detection-annotation.ipynb`
5. Labeled files appear in `./pre/`

---

## Key Observations

- YOLOv8 was pretrained on COCO's `person` class. Face chips from IMDB-WIKI are usually detected as `person` (class 0) — which is close enough to get the bounding box.
- Each face chip is already cropped tight around the face, so detection confidence is high (>0.5 typically).
- The output label format `classID_gender_age x y w h` in `pre/` is a custom format for reference; it needs to be mapped to proper integer class IDs (0–9) before use in YOLOv8 training.
- CPU inference is ~400ms per image; for large datasets use GPU or batch predict.

---

## API Reference

Correct YOLOv8 result access pattern:
```python
results = model.predict(img)         # returns list of Results objects
r = results[0]                       # first image result
r.boxes.data                         # tensor: [x1,y1,x2,y2,conf,cls]
r.boxes.xyxy                         # tensor: [x1,y1,x2,y2]
r.boxes.cls                          # tensor of class IDs
r.plot()                             # annotated image as numpy array
```
