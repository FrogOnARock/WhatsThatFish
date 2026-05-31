# OD Model Tuning Guide

Hyperparameters and augmentation levers organized by intended outcome. Ranges shown are for YOLO11l on the LILA binary fish/no-fish dataset at 640×640 with batch=16.

---

## Hyperparameters

### Optimizer

| Parameter | Current | Range | Notes |
|---|---|---|---|
| `lr0` | 0.010438 | 5e-3 – 5e-2 | Initial LR; tuned via Ray Tune. Scale proportionally if changing batch size (double batch → double lr0). |
| `lrf` | 0.01 | 0.001 – 0.1 | Final LR = lr0 × lrf. Lower = more aggressive cosine decay by end of training. |
| `momentum` | 0.937 | 0.85 – 0.98 | SGD momentum. Higher = smoother updates but slower to change direction. YOLO default (0.937) is well-validated. |
| `weight_decay` | 0.00389 | 1e-4 – 1e-2 | L2 regularization. Tuned via Ray Tune. Higher = stronger regularization, useful if val loss diverges from train loss. |
| `warmup_epochs` | 3 | 1 – 5 | Epochs to ramp lr from warmup_bias_lr to lr0. Increase if training is unstable in early epochs. |
| `warmup_momentum` | 0.8 | 0.5 – 0.95 | SGD momentum during warmup only. |
| `warmup_bias_lr` | 0.1 | 0.05 – 0.2 | Starting LR for bias params during warmup. |

---

### Loss Weights

| Parameter | Current | Range | Notes |
|---|---|---|---|
| `box` | 6.0317 | 4.0 – 10.0 | Weight on bbox regression (CIoU) loss. Higher = tighter boxes, better mAP@0.5:0.95. Lower = model prioritises classification over localisation. |
| `cls` | 1.17041 | 0.5 – 2.0 | Weight on classification (BCE) loss. Higher = stronger push to distinguish fish/background. |
| `dfl` | 1.5 | 0.5 – 2.5 | Distribution Focal Loss weight on bbox distribution. Controls precision of offset regression. Impacts mid-to-late training. Being tuned in current sweep. |

**Loss weight interactions:**
- `box` ↑ + `cls` ↓ → prioritise localisation over classification; better mAP@0.5:0.95, risk of more false positives
- `cls` ↑ + `box` ↓ → prioritise classification; better precision/recall curves, looser boxes
- `dfl` ↑ → sharper bbox distributions; helps when cropping detections to feed the classifier

---

### NMS / Inference

| Parameter | Current | Notes |
|---|---|---|
| `conf` | 0.25 | Confidence threshold. Lower = higher recall, more false positives. See outcome table below. |
| `iou` | 0.70 | NMS IoU threshold. Coupled to conf — see outcome table below. |
| `max_det` | 300 | Max detections per image. Only relevant for dense schools; 300 is safe. |

**conf × iou outcome table:**

| Goal | conf | iou |
|---|---|---|
| Standard deployment (precision/recall balanced) | 0.25 | 0.70 |
| Ecological survey mode (maximise recall, accept FP) | 0.15 | 0.50 |
| High precision (minimise false positives) | 0.40 | 0.70 |
| Classifier input quality (tightest crops only) | 0.35 | 0.75 |

Lower `conf` widens the candidate pool passed to NMS; lower `iou` then suppresses more aggressively from that pool. Keep them coupled — loosening conf without tightening iou leads to duplicate detections surviving NMS.

---

### Transfer Learning

| Parameter | Current | Range | Notes |
|---|---|---|---|
| `freeze` | 5 | 0 – 10 | Layers frozen from the backbone head. Higher = less GPU memory, less domain adaptation. Lower = more plasticity for underwater domain shift. 5 is a good balance for LILA. |
| `epochs` | 50 (tune: 20) | — | Tune runs at 20 for speed. Final training at 50 with patience=20 for early stopping. |
| `patience` | 20 | 10 – 30 | Early stopping patience in epochs. |

---

## Augmentation Levers (`od_dataloader.py`)

### Currently Applied (train only)

| Transform | Parameters | Effect |
|---|---|---|
| `ColorJitter` | brightness=0.4, saturation=0.8, hue=0.015 | Simulates variable underwater lighting and colour cast. |
| `RandomHorizontalFlip` | p=0.5 (default) | Adds left/right orientation diversity. Realistic — fish swim both ways. |
| `ScaleJitter` | target=(640,640), scale=(0.5, 2.0) | Varies apparent object size before final Resize. Drives mAP@0.5:0.95 improvement. |
| `Resize` | (640, 640) | Normalises to fixed input size. Always last spatial op. |

---

### Additional Levers by Intended Outcome

#### Improve mAP@0.5:0.95 (tighter localisation)
```python
v2.RandomIoUCrop(min_scale=0.3, max_scale=1.0)  # aggressive crop forces tight box regression
```
Increase `scale_range` on ScaleJitter toward `(0.3, 2.5)` for more extreme scale variation.

---

#### Improve Recall (catch more fish, accept more FP)
Augmentations that make fish harder to detect force the model to learn more robust features:
```python
v2.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))  # simulates backscatter / low visibility
v2.RandomAdjustSharpness(sharpness_factor=0, p=0.3)     # simulates murky water
```
Lower `conf` threshold at inference (0.15) in tandem.

---

#### Improve Generalisation Across LILA Domains (17 source datasets)
```python
v2.RandomGrayscale(p=0.1)          # handles monochrome / low-saturation sources
v2.RandomPosterize(bits=4, p=0.1)  # simulates compression artefacts in older footage
```

---

#### Simulate Underwater Colour Attenuation (domain robustness)
Water absorbs red before green before blue with depth. Amplifying this distribution shift during training helps the model generalise to varied depth footage:
```python
v2.ColorJitter(brightness=0.4, saturation=0.8, hue=0.03)  # widen hue range toward blue shift
```
Can also apply a channel-specific brightness reduction on the red channel directly via a custom transform.

---

#### Reduce Overfitting (train/val gap)
```python
v2.RandomErasing(p=0.3, scale=(0.02, 0.15))  # occludes small regions; forces reliance on full fish shape
```
Increase `weight_decay` toward 1e-2. Reduce `lr0` slightly.

---

## What NOT to Add

| Transform | Reason |
|---|---|
| `RandomVerticalFlip` | Fish do not orient upside-down in real dive footage. Introduces unrealistic training samples. |
| `RandomRotation` (large angles) | Same reason — fish stay roughly horizontal. Small angles (±10°) are fine for tilt. |
| Mosaic | Bypassed by custom dataloader. Would require reimplementation in `od_dataloader.py` with bbox-aware stitching. |

---

## Current Tuning Sweep Parameters

```python
PARAM_SPACE = {
    "lr0":          loguniform(5e-3, 5e-2),
    "box":          uniform(5.0, 9.0),
    "cls":          uniform(0.8, 1.5),
    "weight_decay": loguniform(1e-3, 1e-2),
    "dfl":          loguniform(0.5, 2.0),
}
# 10 trials, 20 epochs each, metric: mAP@0.5
```
