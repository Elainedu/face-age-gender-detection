# Face Age / Gender Detection (YOLOv8)

> Single-shot face detection **and** age/gender classification with YOLOv8m.
> One forward pass, one bounding box, one of ten combined `(age-group, gender)` classes.

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

The model takes an image and returns, for every face it finds:

- a **bounding box** in image coordinates, and
- one of **10 classes** encoding both **age band** and **gender**.

Because age and gender are baked into the label set, no separate
classification head is needed - the standard YOLOv8 detection head handles
everything in one pass.

**Input**: RGB image (any aspect ratio; letterboxed to 640 x 640).
**Output**: list of `(x1, y1, x2, y2, class_id, confidence)` per face.

Typical use cases: crowd demographics, customer analytics, targeted
signage, dataset labelling for downstream research.

## Model and Dataset

**Base weights**: `yolov8m.pt` (YOLOv8 medium, COCO pretrained, ~25.9 M
params). The default detection head is re-trained on a 10-class label set.
Other variants (`n`, `s`, `l`, `x`) are supported via `--model` if you need
smaller/larger trade-offs.

**Classes** = 5 age bands x 2 genders = **10**:

| ID | Class      | ID | Class        |
|----|------------|----|--------------|
| 0  | Male 0-20  | 5  | Female 0-20  |
| 1  | Male 21-40 | 6  | Female 21-40 |
| 2  | Male 41-60 | 7  | Female 41-60 |
| 3  | Male 61-80 | 8  | Female 61-80 |
| 4  | Male 81+   | 9  | Female 81+   |

**Dataset**: [IMDB-WIKI face chips](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).
Each filename encodes the ground truth:

```
{age}_{gender}_{other}_{timestamp}.jpg.chip.jpg
# e.g. 60_1_0_20170110141405608.jpg.chip.jpg  -> age=60, gender=1(female) -> class 7
```

`auto_label.py` parses these filenames and writes YOLO-format `.txt`
labels. Because IMDB-WIKI chips are pre-cropped faces, the bounding box is
set to the center 90 % of the image.

## System Architecture

```
   +-----------------------------+
   |  Input image                |
   +--------------+--------------+
                  |
                  v
   +-----------------------------+
   |  Ultralytics preprocess     |
   |  letterbox -> 640 x 640     |
   +--------------+--------------+
                  |
                  v
   +-----------------------------+
   |  YOLOv8m backbone (CSP)     |
   |  + PAN/FPN neck             |
   +--------------+--------------+
                  |
                  v
   +-----------------------------+
   |  Detection head (nc = 10)   |
   |  anchor-free, one class per |
   |  box = (age-band, gender)   |
   +--------------+--------------+
                  |
                  v
   +-----------------------------+
   |  NMS + confidence threshold |
   +--------------+--------------+
                  |
                  v
   +-----------------------------+
   |  Visualisation              |
   |  colored box per class,     |
   |  label + confidence overlay |
   +-----------------------------+
```

## Repository Layout

```
face-age-gender-detection/
|-- README.md
|-- updated/                   # USE THIS - current maintained code
|   |-- train.py               # YOLOv8 training entry point
|   |-- inference.py           # Single-image / batch prediction CLI
|   |-- demo.py                # Gradio web UI
|   |-- auto_label.py          # IMDB-WIKI filename -> YOLO labels
|   |-- requirements.txt
|   |-- configs/
|   |   `-- person.yaml        # YOLO dataset config, nc = 10
|   |-- data/
|   |   `-- labels/            # Generated labels (train/, val/) + .cache
|   `-- notebooks/
|       `-- train-yolov8m-face-detection.ipynb
`-- _original/                 # Archived original submissions - reference only
    |-- assignment-02-yolov8-face-detection/
    `-- yolov8-face-age-gender-labeling/
        |-- 01-face-detection-label-extraction.ipynb
        |-- 02-batch-face-detection-annotation.ipynb
        `-- yolov8m.pt          # Base weights snapshot
```

> Everything you actually run lives in `updated/`. `_original/` is
> preserved for provenance (original coursework submission + labelling
> notebooks). All commands below assume you have `cd updated`.

## Training Pipeline

```
IMDB-WIKI chips (raw jpgs)
        |
        | auto_label.py --input raw_images --output data/labels
        v
YOLO-format labels (.txt, one per image)
        |
        | Split into train/ and val/ (images + labels)
        v
configs/person.yaml points at the split
        |
        | python train.py --model yolov8m.pt --epochs 10 --batch 16
        v
results/train/weights/best.pt (+ last.pt, training plots)
        |
        | python inference.py --image test.jpg
        | python demo.py --model results/train/weights/best.pt
        v
Predictions / Gradio UI
```

## Key Files

| File | Purpose |
|------|---------|
| `updated/train.py` | Ultralytics training driver. Loads a base checkpoint (`yolov8n/s/m/l/x.pt`), calls `model.train(data=..., epochs=..., batch=..., imgsz=640)`, saves to `results/train/`. Flags: `--model --data --epochs --batch --imgsz --device --validate`. |
| `updated/inference.py` | Single image (`--image`) or batch directory (`--image-dir`) prediction. Prints per-face class + confidence, saves annotated images to `results/inference/` or `results/batch_inference/`. |
| `updated/demo.py` | Gradio web app on port 7860. Wraps a `FaceAgeGenderDetector` class that runs `YOLO.predict`, colors boxes per class, and outputs a demographic summary. |
| `updated/auto_label.py` | Parses IMDB-WIKI filenames (`{age}_{gender}_..._{ts}.jpg.chip.jpg`) into YOLO-format `.txt` labels. Uses the full-image bbox (`0.5 0.5 0.9 0.9`) because chips are pre-cropped faces. `--limit N` for a smoke test. |
| `updated/configs/person.yaml` | Ultralytics dataset config: `train`, `val` paths, `nc: 10`, class names. |
| `updated/notebooks/train-yolov8m-face-detection.ipynb` | Original Jupyter version of `train.py`; useful when running in Colab/Kaggle. |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (installed with CUDA 11.8+ recommended)
- Ultralytics 8.0+
- OpenCV 4.8+
- Gradio 4.0+
- NumPy, Pandas, Matplotlib, Pillow, tqdm, PyYAML
- 16 GB+ system RAM for training; a CUDA GPU strongly recommended.

Install:

```bash
cd updated
pip install -r requirements.txt
```

## Running Demo and Inference

### Gradio demo

```bash
cd updated
python demo.py
# or with a specific checkpoint
python demo.py --model results/train/weights/best.pt
```

Features: upload image, adjustable confidence slider, multi-face detection,
per-class colored bounding boxes, and a gender/age distribution summary.

### CLI inference

```bash
cd updated

# Single image
python inference.py --image test.jpg

# Whole directory
python inference.py --image-dir test_images/

# Custom confidence
python inference.py --image test.jpg --conf 0.5

# Explicit checkpoint
python inference.py --image test.jpg --model results/train/weights/best.pt
```

Annotated outputs go to `results/inference/` (single) or
`results/batch_inference/` (batch).

## Training from Scratch

1. **Get IMDB-WIKI face chips** from
   [ETH Zurich CVL](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).
   Extract into `data/raw_images/`.

2. **Auto-label**:

   ```bash
   cd updated
   # smoke test first
   python auto_label.py --input data/raw_images --output data/labels --limit 100
   # full run
   python auto_label.py --input data/raw_images --output data/labels
   ```

3. **Arrange images and labels** into Ultralytics' expected layout:

   ```
   <dataset root>/
     images/train/*.jpg
     images/val/*.jpg
     labels/train/*.txt
     labels/val/*.txt
   ```

   Then edit `configs/person.yaml` so `train:` and `val:` point at
   `images/train` and `images/val`.

4. **Train**:

   ```bash
   # Default: yolov8m.pt, 3 epochs, batch 8 (smoke test)
   python train.py

   # Realistic run
   python train.py --model yolov8m.pt --epochs 20 --batch 16

   # Train and validate at the end
   python train.py --epochs 10 --validate
   ```

5. **Outputs** land under `results/`:

   ```
   results/
     train/
       weights/
         best.pt        # highest val mAP
         last.pt        # final epoch
       results.png      # loss + mAP curves
       confusion_matrix.png
       val_batch*.jpg   # sample predictions
   ```

## Notes

- **GPU required for reasonable training time.** On CPU, one epoch of
  `yolov8m` over the IMDB-WIKI subset is measured in hours.
- **Class imbalance**: `81+` age bands have very few samples, which hurts
  recall on the tails. Consider oversampling, focal loss, or dropping them
  to 5 classes if that matters for your use case.
- **Label noise**: age labels come from IMDB-WIKI filename metadata and are
  themselves noisy - do not expect perfect fine-grained accuracy.
- **Bounding boxes are synthetic**: `auto_label.py` assumes the whole chip
  is the face and writes a fixed `0.5 0.5 0.9 0.9` bbox. For end-to-end
  detection on wild images, retrain with real face bounding boxes.
- **Quick baseline vs full run**: the default 3 epochs is a smoke test. For
  anything usable, plan on at least 10-20 epochs.
- **Webcam**: not shipped. Ultralytics supports `model.predict(source=0)`
  if you want to wire it up.

## Files Not in Repo

Excluded intentionally:

- `updated/results/train/weights/best.pt`, `last.pt` - trained checkpoints
  (tens to hundreds of MB). Retrain locally.
- `updated/data/raw_images/`, `updated/data/images/{train,val}/` - the
  IMDB-WIKI face chips. Download from the ETH Zurich site.
- `runs/`, `.cache` files - regenerated by Ultralytics on each run.

The `_original/yolov8-face-age-gender-labeling/yolov8m.pt` checkpoint is a
COCO-pretrained base and can be re-downloaded automatically by Ultralytics
on first use.

## License

Released for educational use as part of a deep learning course project.

- **YOLOv8 / Ultralytics**: AGPL-3.0 (see the upstream repo).
- **IMDB-WIKI dataset**: research-only license from ETH Zurich. Download
  from the original site and respect its terms.
