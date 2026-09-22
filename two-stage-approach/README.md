# DL Project 2 - YOLOv8 Age/Gender Detection

> Two-stage YOLOv8 pipeline for age-gender classification of people in images:
> a COCO-pretrained detector finds every person, then a custom 10-class YOLOv8
> classifier predicts age band + gender per crop.

## Table of Contents

- [What This Does](#what-this-does)
- [Model and Dataset](#model-and-dataset)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Training Pipeline](#training-pipeline)
- [Key Files](#key-files)
- [Requirements](#requirements)
- [Running Demo and Inference](#running-demo-and-inference)
- [Training from Scratch](#training-from-scratch)
- [Notes](#notes)
- [Files Not in Repo](#files-not-in-repo)
- [License](#license)

## What This Does

Given an image, the pipeline:

1. Runs a COCO-pretrained `yolov8n.pt` detector, keeping only class 0
   (`person`) with confidence >= 0.30.
2. Crops each detected person (with a small padding), and feeds the crop
   into a **second** YOLOv8 model trained on a custom 10-class label set
   (age band x gender).
3. Draws a per-class colored bounding box and label on the original image.

**Input**: RGB image (JPG / PNG, any size).
**Output**: annotated image + textual summary listing detected people, per
person `(gender, age band, confidence)`, and aggregate male/female counts.

The two-stage design decouples "where are the people" from "who is this
person", so the person detector can stay a stock COCO model while the
age-gender classifier can be retrained without touching detection quality.

## Model and Dataset

### Models

| Stage | Model | Variant | Purpose |
|-------|-------|---------|---------|
| 1. Person detection | `yolov8n.pt` | nano (~3.2 M params) | Fast COCO person detection; class 0 only |
| 2. Age-gender class. | `yolov8m.pt` -> retrained | medium (~25.9 M params) | 10-class age-gender head |

A lighter alternative is trained via `train_balanced.py`, which uses
`yolov8n.pt` on the balanced dataset (faster iteration, lower ceiling).

All training uses input size **640 x 640** with Ultralytics' default
augmentation (mosaic, HSV jitter, flip).

### Classes (nc = 10)

```
0 male0-20      5 female0-20
1 male21-40     6 female21-40
2 male41-60     7 female41-60
3 male61-80     8 female61-80
4 male81~       9 female81~
```

### Dataset

Sourced from **IMDB-WIKI face chips** (filenames encode `age_gender_...`).
Two configurations ship in the repo:

| Config | Train images | Val images | Notes |
|--------|--------------|------------|-------|
| `person.yaml` | 675 | 242 | Raw split - reflects natural class imbalance |
| `person_balanced.yaml` | 1,500 | 1,500 | 150 per class oversampled (`balance_dataset.py`) |

Total raw dataset staged in `資料集處理(全部資料集down)/123/` is
~9,780 source images and ~19,560 image+label files (roughly 37K files
total including preprocessed splits).

## System Architecture

```
                    +--------------------------+
                    | Input image              |
                    +------------+-------------+
                                 |
                                 v
                    +--------------------------+
                    | Stage 1: yolov8n.pt      |
                    | (COCO, classes=[0])       |
                    +------------+-------------+
                                 |
                                 |  N person boxes (x1,y1,x2,y2)
                                 v
                    +--------------------------+
                    | Crop + pad(10 px)        |
                    | reject if < 30 x 30 px   |
                    +------------+-------------+
                                 |
                       for each crop
                                 |
                                 v
                    +--------------------------+
                    | Stage 2: best.pt         |
                    | 10-class age-gender      |
                    | conf >= 0.01, pick top-1 |
                    +------------+-------------+
                                 |
                                 v
                    +--------------------------+
                    | Draw colored box +       |
                    | "Gender age-band NN%"    |
                    | + aggregate stats        |
                    +--------------------------+
```

## Repository Layout

```
dl-project2-yolov8-detection/
|-- .gitignore                         # Ignores *.pt, runs/, big datasets, etc.
|-- README.md                          # This file
|-- YoloV8/                            # Actual code + working splits
|   |-- train.py                       # Train yolov8m on person.yaml (20 epochs)
|   |-- train_balanced.py              # Train yolov8n on balanced split (10 epochs)
|   |-- train.ipynb                    # Notebook version of train.py (3 epochs)
|   |-- demo.py                        # Gradio - full 2-stage age/gender demo
|   |-- demo_simple.py                 # Gradio - COCO-only person detection
|   |-- balance_dataset.py             # Oversample each class to 150 images
|   |-- fix_labels.py                  # Convert legacy 7-col labels -> YOLO 5-col
|   |-- check_model.py                 # Sanity-check trained weights on test_image/
|   |-- quick_test.py                  # Smoke test the base yolov8n.pt
|   |-- test_model.py                  # Batch predict with runs/detect/train/best.pt
|   |-- person.yaml                    # Dataset config: images/train, images/val
|   |-- person_balanced.yaml           # Balanced dataset config
|   |-- images/
|   |   |-- train/                     # 675 files
|   |   |-- val/                       # 242 files
|   |   |-- train_balanced/            # 1,500 (oversampled)
|   |   `-- val_balanced/              # 1,500 (oversampled)
|   |-- labels/
|   |   |-- train/, val/               # YOLO .txt labels
|   |   |-- train_balanced/, val_balanced/
|   |   `-- *.cache                    # Ultralytics label cache (regenerated)
|   `-- test_image/                    # Ad-hoc demo/test images
`-- 資料集處理(全部資料集down)/         # Raw dataset + preprocessing notebooks
    `-- 123/
        |-- dataset/                   # ~9,780 raw IMDB-WIKI chips (.jpg)
        |-- pre/                       # Preprocessed .jpg + .txt pairs
        |-- t1.ipynb                   # Preprocessing notebook 1
        `-- t2.ipynb                   # Preprocessing notebook 2 (labelling)
```

The Chinese-named folder (`資料集處理(全部資料集down)/`) means "dataset
processing (all datasets downloaded)". It is the raw-data staging area and
is separate from the training-ready splits in `YoloV8/`.

## Training Pipeline

```
Raw IMDB-WIKI chips (資料集處理/123/dataset/)
        |
        | t1.ipynb / t2.ipynb  (preprocessing + labelling)
        v
資料集處理/123/pre/  (image + .txt pairs)
        |
        | fix_labels.py  (legacy 7-col -> YOLO 5-col, normalized)
        v
YoloV8/images/{train,val} + YoloV8/labels/{train,val}
        |
        +--> balance_dataset.py  (oversample each class to 150)
        |         v
        |    images/{train,val}_balanced + labels/{train,val}_balanced
        |         |
        |         v
        |    python train_balanced.py  (yolov8n, 10 epochs, batch 16)
        |
        `--> python train.py           (yolov8m, 20 epochs, batch 8)
                       |
                       v
                runs/detect/train*/weights/best.pt
                       |
                       v
                python demo.py --model runs/detect/train*/weights/best.pt
```

## Key Files

| File | Purpose |
|------|---------|
| `YoloV8/train.py` | Minimal Ultralytics driver: `YOLO('yolov8m.pt').train(data='person.yaml', epochs=20, batch=8, imgsz=640)`. |
| `YoloV8/train_balanced.py` | Same idea but with `yolov8n.pt`, `person_balanced.yaml`, 10 epochs, batch 16, `patience=5`. Faster iteration on the oversampled split. |
| `YoloV8/train.ipynb` | 3-epoch notebook version of `train.py` for quick Colab/Kaggle runs. |
| `YoloV8/demo.py` | Full Gradio demo. Two-stage: COCO person detection -> age-gender classification per crop. Args: `--model` (classifier .pt), `--share`, `--port`. |
| `YoloV8/demo_simple.py` | Simplified demo that only runs the COCO person detector - useful when the custom classifier is not trained yet. |
| `YoloV8/balance_dataset.py` | Class-balancing utility. Reads `labels/{train,val}`, groups files by class, copies with replacement up to `TARGET = 150` per class, writes to `*_balanced/`. |
| `YoloV8/fix_labels.py` | One-off migration script. Converts old 7-column labels (`coco_cls gender age x y w h` in pixel coords) into standard YOLO 5-column labels (`class x_center y_center w h` normalized). |
| `YoloV8/check_model.py` | Loads `runs/detect/train6/weights/best.pt`, runs on `test_image/`, prints predicted class + confidence. |
| `YoloV8/quick_test.py` | Loads base `yolov8n.pt`, runs on `test_image/` (falls back to `images/train`), prints all detected COCO classes. |
| `YoloV8/test_model.py` | Batch prediction with `runs/detect/train/weights/best.pt`, `conf=0.1`. |
| `YoloV8/person.yaml` | `path: .`, `train: images/train`, `val: images/val`, `nc: 10`, class name list. |
| `YoloV8/person_balanced.yaml` | Same, but pointing at the `_balanced` splits. |
| `資料集處理/123/t1.ipynb`, `t2.ipynb` | Preprocessing notebooks that produced `pre/`. |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA 11.8+ strongly recommended)
- Ultralytics 8.0+
- OpenCV (`opencv-python`) 4.8+
- Gradio 4.0+
- Pillow, NumPy

No `requirements.txt` is committed; install with:

```bash
pip install ultralytics opencv-python gradio pillow numpy
```

## Running Demo and Inference

### Simple person-only demo (no training required)

```bash
cd YoloV8
python demo_simple.py
```

Downloads `yolov8n.pt` automatically on first run, launches Gradio at
`http://127.0.0.1:7860`, and detects people in uploaded images using the
COCO pretrained head. Good for validating the environment.

### Full two-stage age-gender demo

```bash
cd YoloV8
python demo.py                                          # uses runs/detect/train7/weights/best.pt
python demo.py --model runs/detect/train/weights/best.pt
python demo.py --port 8080 --share                      # public share link
```

### Quick sanity checks

```bash
cd YoloV8
python quick_test.py       # base yolov8n.pt on test_image/
python test_model.py       # trained best.pt on test_image/
python check_model.py      # spot-check trained best.pt classes + confidences
```

## Training from Scratch

1. **Prepare the dataset.** The IMDB-WIKI chips should live in
   `資料集處理(全部資料集down)/123/dataset/`. Run `t1.ipynb` / `t2.ipynb`
   to generate `pre/` (paired `.jpg` + `.txt`).
2. **Migrate labels** if they were written in the legacy 7-column format:

   ```bash
   cd YoloV8
   python fix_labels.py
   ```

3. **Split into `images/{train,val}` and `labels/{train,val}`.**
4. (Optional) **Balance the classes**:

   ```bash
   cd YoloV8
   python balance_dataset.py
   # writes images/{train,val}_balanced + labels/{train,val}_balanced
   ```

5. **Train**:

   ```bash
   cd YoloV8

   # Full model on raw split
   python train.py                # yolov8m, 20 epochs, batch 8

   # Faster model on balanced split
   python train_balanced.py       # yolov8n, 10 epochs, batch 16
   ```

6. **Outputs** land under `runs/detect/train*/`:

   ```
   runs/detect/train/
     weights/
       best.pt          # highest val mAP
       last.pt          # final epoch
     results.png
     confusion_matrix.png
     val_batch*.jpg
   ```

   Later runs land in `train2/`, `train3/`, ... Ultralytics auto-increments.
   `demo.py` defaults to `runs/detect/train7/weights/best.pt`; pass
   `--model` to override.

## Notes

- **GPU is effectively required.** Training `yolov8m` for 20 epochs on
  ~700 images is feasible on CPU (hours) but unpleasant; a single
  consumer GPU (RTX 30xx+) trains this in tens of minutes.
- **Webcam / video**: not wired in. Ultralytics accepts
  `model.predict(source=0)` or a video path; adapt `demo.py` if needed.
- **Class imbalance is real.** The `81~` age bands have very few samples in
  the raw split - hence `balance_dataset.py`. Consider heavier oversampling
  or class weights if fine-grained recall on the tails matters.
- **Age labels are noisy** because they come from IMDB-WIKI filename
  metadata. Do not expect the model to distinguish, say, 39 from 41 years
  reliably.
- **Two-stage vs single-stage**: this project uses stage-1 COCO detection
  because IMDB-WIKI chips are pre-cropped faces and provide no realistic
  full-body bounding boxes for training a single-stage age-gender detector.

## Files Not in Repo

The `.gitignore` deliberately excludes:

- `*.pt`, `*.pth`, `*.ckpt`, `*.h5`, `*.bin`, `*.safetensors` - all model
  weights. Retrain with `train.py` / `train_balanced.py` to reproduce.
- `best.pt`, `last.pt` - trained checkpoints (tens to hundreds of MB).
- `runs/` - Ultralytics training output directory (includes checkpoints,
  plots, sample predictions).
- `*.zip`, `*.tar.gz`, `*.7z` - large archives.
- `.venv/`, `venv/`, `.vscode/`, `.idea/` - dev environment.

The `资料集處理(全部資料集down)/123/dataset/` and `pre/` folders **are**
committed for reproducibility of the exact split used, at the cost of the
large file count. If you fork this repo, consider moving those out.

## License

Released for educational use as part of a deep learning course project.

- **YOLOv8 / Ultralytics**: AGPL-3.0 (see the upstream repo).
- **IMDB-WIKI dataset**: research-only license from ETH Zurich; obtain
  from the original site and respect its terms.
- **COCO pretrained weights**: distributed by Ultralytics under their
  licence.
