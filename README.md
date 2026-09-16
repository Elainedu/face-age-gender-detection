# Face Age / Gender Detection (YOLOv8)

Single-shot face detection **and** age / gender classification with
YOLOv8m. The model predicts a face bounding box and one of 10 combined
`(age-group, gender)` classes in a single forward pass.

## Repo Layout

This repository is intentionally split into two top-level folders:

```
face-age-gender-detection/
|-- updated/       # Current, actively maintained version -- USE THIS
`-- _original/     # Archive of earlier coursework submissions
    |-- assignment-02-yolov8-face-detection
    `-- yolov8-face-age-gender-labeling
```

- **`updated/`** contains the current end-to-end project (train / demo /
  auto-label / configs / notebooks). All commands below assume you have
  changed into this directory first.
- **`_original/`** keeps the earlier assignment submissions purely for
  reference. It is not required to run the project and can be ignored for
  day-to-day work.

```bash
cd updated
```

## Overview

Age groups (5) x Gender (2) = **10 classes**, all handled by a single YOLO
head:

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Male 0-20 | 5 | Female 0-20 |
| 1 | Male 21-40 | 6 | Female 21-40 |
| 2 | Male 41-60 | 7 | Female 41-60 |
| 3 | Male 61-80 | 8 | Female 61-80 |
| 4 | Male 81+ | 9 | Female 81+ |

Base weights: `yolov8m.pt` (COCO-pretrained). We keep the default YOLOv8
detection head and only swap in a custom 10-class dataset.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (recommended for training)
- 16 GB+ system RAM

Python dependencies (see `updated/requirements.txt`):

```
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
gradio>=4.0.0
tqdm>=4.65.0
pyyaml>=6.0
```

Install once:

```bash
cd updated
pip install -r requirements.txt
```

## Quick Start

All scripts live in `updated/` -- run them from there.

### 1. Auto-label the dataset

The IMDB-WIKI face-chip filename already encodes age and gender, so we can
generate YOLO labels from the filenames alone:

```bash
# Full run
python auto_label.py --input data/raw_images --output data/labels

# Sanity check on 100 images
python auto_label.py --input data/raw_images --output data/labels --limit 100
```

Filename convention:

```
{age}_{gender}_{other}_{timestamp}.jpg.chip.jpg
# e.g. 60_1_0_20170110141405608.jpg.chip.jpg -> age=60, gender=female
```

### 2. Train

```bash
# Default: yolov8m.pt, 3 epochs, batch 8
python train.py

# Custom
python train.py --model yolov8m.pt --epochs 10 --batch 16

# Train + validate
python train.py --epochs 5 --validate
```

Training config lives in `configs/person.yaml`:

```yaml
train: 'path/to/images/train'
val:   'path/to/images/val'
nc: 10
names: ['male0-20','male21-40','male41-60','male61-80','male81+',
        'female0-20','female21-40','female41-60','female61-80','female81+']
```

### 3. Inference

```bash
# Single image
python inference.py --image test.jpg

# Batch directory
python inference.py --image-dir test_images/

# Adjust confidence
python inference.py --image test.jpg --conf 0.5
```

### 4. Gradio demo

```bash
python demo.py
# or with a custom checkpoint
python demo.py --model results/train/weights/best.pt
```

The demo supports image upload, multi-face detection, per-class colored
bounding boxes, adjustable confidence threshold, and gender / age
distribution stats.

## Model

- **Base**: `yolov8m.pt` (COCO-pretrained, medium variant).
- **Head**: default YOLOv8 detection head, re-trained for 10 classes.
- **Input size**: 640 x 640.
- **Loss**: standard YOLO detection loss (box + cls + dfl).

## Notes

- **Weights are not committed.** Only `yolov8m.pt` (base pretrained
  weights) is included; trained `best.pt` files can be tens to hundreds of
  MB and are omitted from git. Retrain locally to reproduce.
- **Dataset source**: [IMDB-WIKI Face Chips]
  (https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) -- please
  download from the original site and respect its license.
- **Class imbalance**: the 81+ age groups have far fewer samples, which
  hurts recall on the tails. Consider oversampling or class weights for
  serious runs.
- **Label noise**: ages come from filename metadata and are only as good
  as the source; expect some noise in the ground truth.
- **Quick baseline vs full run**: the default 3 epochs is a smoke-test.
  For a usable model plan on at least 10-20 epochs.

## References

- Ultralytics YOLOv8: <https://github.com/ultralytics/ultralytics>
- IMDB-WIKI: <https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/>

## License

Released for educational use as part of a deep learning course project.
The YOLOv8 weights and IMDB-WIKI dataset remain under their original
licenses.
