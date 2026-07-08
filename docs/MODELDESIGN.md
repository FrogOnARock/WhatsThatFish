# Model Design

WhatsThisFish is a **two-stage vision pipeline**: a YOLO11l detector localises the
fish, and a 5-channel hierarchical ResNet classifies the crop across three
taxonomic heads (family → genus → species).

```
image ──▶ [YOLO11l detector] ──▶ best box ──▶ crop ──▶ [CustomResnet] ──▶ {family, genus, species}
             conf=0.15                          letterbox 320²          3 ranked candidate lists
```

Both models are trained in PyTorch and served as **ONNX** (torch-free container).
The preprocessing does **not** live in the ONNX graph — it is PIL/NumPy and is
replicated byte-for-byte at serving time (see [§ Serving parity](#serving-parity)).

---

## Stage 1 — Object Detector (YOLO11l)

Binary **fish / no-fish** detector, trained on LILA then progressively adapted to
underwater iNaturalist imagery through a two-stage learning curriculum.

| Stage | Data | Input weights | Output |
|---|---|---|---|
| **LILA base** | ~103K domain-balanced images, ~1:1 pos/neg across 17 sources | `yolo11l.pt` | `od_best.pt` |
| **LC1** | iNat fish ranked by UIQM (best-quality first, ≤100K) | `od_best.pt` | `lc1_best.pt` |
| **LC2** *(deferred)* | same images sampled by `uiqm × conf` | `lc1_best.pt` | `lc2_best.pt` |

> **Shipping detector = LC1.** LC1 already clears every detection target
> (see [MODELMETRICS](./MODELMETRICS.md)), so **LC2 is deferred** — it's optional
> refinement, not a prerequisite for serving.

- **`CustomDetectionTrainer`** subclasses Ultralytics `DetectionTrainer` and
  overrides `get_dataloader` / `preprocess_batch` / `get_validator`. It skips
  Ultralytics' `/255` normalization because `ToTensor()` already normalizes
  upstream — otherwise the image would be double-normalized.
- **Coral negatives** — Anthozoa images (taxon 47533) are injected as
  `conf=1.0`, empty-annotation negatives during detection-mode bbox proposal,
  *without* running inference, so the detector learns "reef ≠ fish".
- **Hyperparameter tuning** (Ray Tune, 20 epochs/trial): tunes `lr0`, `box`,
  `weight_decay`; fixes `cls=0.4` (binary task) and `dfl=1.0` (pseudo-labels
  don't warrant tight edge distributions). Full runs use per-dataset flat YAML
  configs (`config/lc1_train_config.yaml`, etc.).

CLI: `python -m whatsthatfish.models.detection --dataset lc1 --type tune|full`

---

## Stage 2 — Hierarchical Classifier (CustomResnet)

`models/architecture/custom_resnet.py` — a ResNet-34-style BasicBlock backbone
(4 residual groups, 64→128→256→512 channels) with a **5-channel input stem** and
**three taxonomic heads**.

### 5-channel input (the additional dimensions)

The classifier sees more than RGB. Two extra channels expose **texture and edge
structure** that survive the colour cast and low contrast of underwater imagery,
where RGB alone is a weak signal. `AddMultiChannel` (`transforms/five_channel_conversion.py`)
emits a `(5, H, W)` float32 tensor:

| Ch | Content | Normalization |
|----|---------|---------------|
| 0–2 | RGB | `/255` → [0, 1] |
| 3 | **Scharr gradient magnitude** (edges) | min-max → [0, 1] |
| 4 | **Local Contrast Normalization** (texture) | min-max → [0, 1] |

Both extra channels are computed on the **grayscale** image
(`cv2.COLOR_RGB2GRAY`, float64) — see `transforms/lcn_gradient_map.py`:

**Channel 3 — Scharr gradient.** The Scharr operator is a more rotationally
symmetric 3×3 alternative to Sobel. We take both partials and their magnitude:

```
gx = Scharr(I, dx=1, dy=0)
gy = Scharr(I, dx=0, dy=1)
magnitude = sqrt(gx² + gy²)
ch3 = minmax(magnitude) · 255   # then /255 in AddMultiChannel → [0,1]
```

This sharpens fin rays, scale boundaries, and body outline — features that
distinguish species even when hue is washed out.

**Channel 4 — Local Contrast Normalization.** Emphasises local texture
independent of absolute brightness (so a shadowed fish and a sunlit one look
alike to the head). Subtract a Gaussian-blurred local mean, divide by the
Gaussian-estimated local standard deviation (floored at `1e-4`), min-max rescale:

```
μ_local = GaussianBlur(I, 5×5)
v = I − μ_local
σ_local = sqrt( GaussianBlur(v², 5×5) )
ch4 = minmax( v / max(σ_local, 1e-4) ) · 255   # then /255 → [0,1]
```

A flat, contrast-free region returns all zeros (guarded `max == min` case).

> **Design note:** channels are min-max normalized *per image*, so they encode
> relative structure, not absolute magnitudes — robust to exposure differences
> across dive photos.

### Stem inflation (pretrained → 5 channels)

Pretrained variants load torchvision **ResNet-34 ImageNet** weights (`layers=[3,4,6,3]`).
The first `7×7` conv is `(3, 64)` in ImageNet but we need `(5, 64)`, so the stem
is **inflated**: the 3 RGB filters are copied verbatim; the 2 extra channels are
initialised to the **mean of the RGB filters, rescaled by `3/in_dim`** so the
expected activation magnitude is preserved. From-scratch variants use Kaiming
normal on conv, constant 1/0 on BatchNorm.

### Progressive heads (family → genus → species)

Three FC heads read the 512-dim pooled features, computed **coarse-to-fine**.
Toggle `head_mode`:

- **`progressive`** (default) — each head's logits are projected to a 64-dim
  bottleneck and **concatenated onto the pooled features** feeding the next,
  finer head. **Parent logits are detached**, so a child's loss never backprops
  into its parent head. Raw logits, no softmax.
- **`parallel`** — three independent linear heads off the pooled features
  (ablation baseline).

### Training (`Classifier` facade, `models/classifier.py`)

`train()` / `tune()` / `predict()` share one `_fit()` core (a full run is tuning
with a single fixed config). Heavy deps (ray / sqlalchemy / dataloaders) are
lazy-imported so `predict()` and serving stay slim.

- **Curriculum loss weighting** (3 phases): `[0,0,1]` family-only →
  `[0,0.6,0.4]` +genus → `[0.6,0.3,0.1]` all three. Transitions gated by time
  (`min_time_per_phase`) **and** performance (family/genus curriculum gates in
  `cls_metrics.py`). Val loss always uses phase-3 weights so `best.pt` selection
  is comparable across phases.
- **Per-head loss** — inverse-frequency class-weighted CrossEntropy +
  `label_smoothing=0.1`.
- **Discriminative LRs** (pretrained) — two optimizer groups: head/stem at
  `head_lr`, backbone at `backbone_lr` (~10× slower), under OneCycleLR
  (`max_lr=[head_lr, backbone_lr]`). Backbone frozen for `freeze_epochs`
  (BN held in `eval` so ImageNet running stats survive), then unfrozen.
  From-scratch uses a single group at `lr`/`max_lr`.
- **Tuning** — in-process random search (no Ray workers — a 2nd CUDA process
  OOM'd the single GPU); each trial runs the same `_fit()` on a fresh
  model/optimizer/scheduler/metrics, so no state bleeds between trials; lowest
  val loss wins.

CLI: `python -m whatsthatfish.models.classifier --type tune|full`

### Transform pipeline

- **Train:** `RandomRotation(0–90°)` → `ElasticTransform(α=75)` →
  `RandomHorizontalFlip` → `ColorJitter` → `RandomAdjustSharpness` →
  `LetterboxResize(320)` → `AddMultiChannel()` → `(5, 320, 320)`.
- **Val:** `LetterboxResize(320)` → `AddMultiChannel()` only.
- **Sampler:** UIQM-weighted `WeightedRandomSampler` on the train split
  (null guard `max(uiqm or 0.0, 1e-6)`).

`LetterboxResize` pads to a square canvas preserving aspect ratio (symmetric
zero-pad, bilinear), so the fish is never distorted before the heads see it.

---

## Serving parity

Serving is **torch-free**: `OnnxBoundingBoxInference` and `OnnxClassInference`
back onnxruntime sessions with the exact `.infer()` contracts of their torch
counterparts, so `PredictionService` is unchanged. The catch:

- The `crop → LetterboxResize(320) → AddMultiChannel` chain stays in **Python/NumPy**
  and must match training **byte-for-byte** — a parity test (`onnxruntime` vs
  torch, top-1 agreement / logits within tolerance) guards this in the suite.
- Classifier ships **INT8** (real Gemm-head speedup); detector ships **FP32**
  (dynamic INT8 is slower on its Conv stack). Both run on `CPUExecutionProvider`.

See [MODELMETRICS.md](./MODELMETRICS.md) for the INT8 accuracy gate and
[DATASOURCE.md](./DATASOURCE.md) for how the training data is built.
