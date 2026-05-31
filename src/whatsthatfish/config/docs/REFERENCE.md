# Personal Reference — YOLO Training + torchvision v2 Transforms

---

## Ultralytics Training Parameters

Full reference for every parameter in `default.yaml`. Those marked with ★ are actively relevant to this project.

---

### Task & Mode

**`task`** — What the model does. `detect` for bounding boxes, `segment` for pixel masks, `classify` for image-level labels, `pose` for keypoints, `obb` for oriented bounding boxes. Determines which head is attached to the backbone.

**`mode`** — Which pipeline to run. `train`, `val`, `predict`, `export`, `track`, `benchmark`. Mostly set automatically.

---

### Core Training ★

**`model`** — Path to a `.pt` checkpoint or a `.yaml` architecture definition. `.pt` initialises with pretrained weights (transfer learning). `.yaml` trains from scratch.

**`data`** — Path to the dataset YAML. For this project `class_config.yaml` — defines number of classes, class names, and `train`/`val` paths. Since the custom dataloader ignores those paths, they're set to `/dev/null`.

**`epochs`** ★ — Total training epochs. Ultralytics default is 100. Tune runs use 20 for speed; final training uses 50 with early stopping.

**`time`** — Alternative to `epochs`: stop after N hours regardless of epoch count. Useful for hard-deadline training runs. Overrides `epochs` if set.

**`patience`** ★ — Early stopping: halt if val mAP@0.5 doesn't improve for N consecutive epochs. Set to 20 to avoid wasting compute on plateaus.

**`batch`** ★ — Images per gradient update. Default 16. Memory scales roughly linearly with batch size. Larger batches = more stable gradients but less stochastic noise (can hurt generalisation with SGD).

**`imgsz`** ★ — Input image size. Single int = square (640×640). Stride must divide evenly into this value. Larger = better small object detection, more memory.

**`save`** — Whether to save checkpoints. `best.pt` (best val mAP) and `last.pt` always saved when True.

**`save_period`** — Save checkpoint every N epochs in addition to best/last. Useful for diagnosing training dynamics. `-1` disables.

**`cache`** — Cache images in RAM (`True`/`'ram'`) or on disk (`'disk'`). Speeds up dataloading at the cost of memory. Irrelevant here since images stream from GCS.

**`device`** — GPU/CPU selection. `0` for first GPU, `[0,1]` for multi-GPU DDP, `'cpu'` to force CPU.

**`workers`** ★ — Number of dataloader worker processes. Each spawns a separate Python process with its own GCS client (via `worker_init_fn`). Set to 8; bounded by CPU cores and GCS connection overhead.

**`project`** / **`name`** — Output directory. Results saved to `project/name/`. Defaults to `runs/detect/trainN`.

**`exist_ok`** — If False, increments `trainN` to avoid overwriting existing results.

**`pretrained`** ★ — Use pretrained backbone weights. Always True here — YOLO11l weights from ImageNet give a massive head start on low-level feature detection.

**`optimizer`** ★ — `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp`, or `auto`. `auto` selects SGD for YOLO. Explicitly set to `SGD` in `train_config.yaml` to avoid surprises. SGD typically generalises better than Adam for detection at the cost of needing more careful lr tuning.

**`verbose`** — Print per-batch loss logs. Set to False in Ray Tune trials to reduce noise.

**`seed`** / **`deterministic`** — Reproducibility. `deterministic=True` makes results repeatable but can slow training (forces cuDNN into deterministic mode).

**`single_cls`** — Treat all classes as one. Useful for ablations. Has no effect here since the dataset is already binary (fish / no-fish).

**`rect`** — Rectangular batching: groups images of similar aspect ratio together to minimise padding. Saves compute but breaks random sampling. Incompatible with the custom dataloader.

**`cos_lr`** — Cosine LR scheduler instead of linear decay. When False (default), Ultralytics uses a cosine decay from `lr0` to `lr0 * lrf` over training, regardless of this flag. Setting True uses a pure cosine schedule without the warmup blend.

**`close_mosaic`** — Disable mosaic augmentation for the final N epochs. Lets the model stabilise on clean images before evaluation. Irrelevant since mosaic is bypassed by the custom dataloader.

**`resume`** — Resume from last checkpoint. Set to a checkpoint path or True to pick up `last.pt` from the most recent run.

**`amp`** ★ — Automatic Mixed Precision. Uses FP16 for forward pass and FP32 for gradient accumulation. Saves ~40% memory with negligible accuracy loss on 4070 Ti Super. Always True.

**`fraction`** — Use only a fraction of the dataset. Useful for quick smoke tests (`fraction=0.1`). Irrelevant with `max_samples` controlling this via the custom dataloader.

**`freeze`** ★ — Freeze the first N backbone layers. Frozen layers skip gradient computation and don't store activations, saving substantial GPU memory. Set to 5. Higher values = less domain adaptation but less memory.

**`multi_scale`** — Randomly vary input size during training by ±`multi_scale` fraction of `imgsz`. E.g. `0.5` = sizes from 320–960. Bypassed by the custom dataloader (ScaleJitter does this instead).

**`compile`** — `torch.compile()` the model for faster inference. Can break compatibility with custom trainers. Leave False.

**`nbs`** — Nominal batch size for loss gradient scaling. Loss is scaled by `batch/nbs` so that effective gradient magnitude stays consistent regardless of batch size. Don't change this.

---

### Loss Weights ★

**`box`** — Weight on the CIoU bounding box regression loss. Higher values push the model to prioritise tight box fitting over class confidence. Directly affects mAP@0.5:0.95 since IoU-averaged metrics penalise loose boxes heavily.

**`cls`** — Weight on the binary cross-entropy classification loss (fish vs. background). Higher values make the model focus more on the fish/no-fish decision.

**`dfl`** — Distribution Focal Loss weight. YOLO11 models predict bounding box offsets as probability distributions over a discretised range rather than point estimates. `dfl` controls how strongly the model is penalised for spread-out distributions (i.e., uncertain box predictions). Higher = sharper, more precise boxes. Primarily affects mid-to-late training once classification has stabilised.

**`pose`** / **`kobj`** / **`rle`** / **`angle`** — Loss weights for pose, keypoint objectness, RLE, and OBB tasks. All irrelevant for detection.

---

### Val / Test Settings

**`val`** — Run validation after each training epoch. Always True during training.

**`split`** — Which dataset split to evaluate on. `val` is standard. `test` if you have a held-out test set.

**`conf`** ★ — Confidence threshold for NMS filtering. During training validation, Ultralytics overrides this to `0.001` regardless of what you set here, to ensure the full precision-recall curve is sampled for mAP computation. This value applies at inference time only.

**`iou`** ★ — NMS IoU threshold. Boxes with IoU above this are considered duplicates and the lower-confidence one is suppressed. Coupled to `conf` — see TUNING_GUIDE.md outcome table.

**`max_det`** — Max detections per image after NMS. 300 is safe for all foreseeable fish densities.

**`half`** — Use FP16 at inference. Separate from training AMP. Halves inference memory at no quality cost on supported GPUs.

**`plots`** — Save confusion matrix, PR curve, F1 curve, and sample prediction images to the run directory after training.

**`save_json`** — Export predictions in COCO JSON format for use with the COCO API evaluation tools.

**`augment`** — Test-time augmentation (TTA): runs predictions at multiple scales/flips and merges results. Improves mAP by 1-3% at the cost of ~3x inference time. Useful for final evaluation, not during training.

**`agnostic_nms`** — Suppress boxes across all classes jointly rather than per-class. Relevant for multi-class; has no effect on binary detection.

---

### Built-in Augmentations (bypassed by custom dataloader)

These live in Ultralytics' `YOLODataset` which is not instantiated when `get_dataloader` is overridden. Listed here for completeness.

**`hsv_h`** / **`hsv_s`** / **`hsv_v`** — Random HSV colour jitter. Fraction of full range to shift hue, saturation, and value. Similar to `ColorJitter` but applied in HSV space directly.

**`degrees`** — Random rotation range in degrees (±). 0 = disabled.

**`translate`** — Random translation as a fraction of image size (±). 0.1 = ±10% shift.

**`scale`** — Random scale gain (±). 0.5 = zoom between 50%–150%.

**`shear`** — Random shear in degrees (±). Simulates perspective distortion.

**`perspective`** — Perspective warp strength. 0–0.001. Simulates camera angle changes.

**`flipud`** / **`fliplr`** — Vertical / horizontal flip probability. `fliplr=0.5` = flip half the time.

**`mosaic`** — Mosaic augmentation probability. Stitches 4 images into one, randomly placed. The primary driver of scale diversity in standard YOLO training. Replaced in this project by `ScaleJitter`.

**`mixup`** — MixUp probability. Blends two images and their labels by a random alpha weight. Improves generalisation on ambiguous cases.

**`cutmix`** — CutMix probability. Pastes a rectangular crop from one image onto another, mixing labels proportionally to area.

**`copy_paste`** — Copies object instances from one image and pastes them onto another. Increases instance density artificially. Useful for rare classes.

**`erasing`** — Random erasing probability (classification). Randomly zeros out a rectangle of pixels to force reliance on the full object rather than a single discriminative region.

---

### Export Settings

**`format`** — Export format. `torchscript` (default), `onnx` (cross-platform), `engine` (TensorRT, fastest on NVIDIA), `coreml` (Apple), `tflite` (mobile).

**`int8`** — INT8 quantisation. Halves model size and increases throughput ~2x at small accuracy cost. Requires calibration data.

**`dynamic`** — Dynamic input shapes for ONNX/TensorRT. Allows variable image sizes at inference.

**`simplify`** — Run ONNX graph simplifier before export. Removes redundant ops. Always True.

**`half`** — FP16 export. For TensorRT especially, enables Tensor Core acceleration.

---

## torchvision v2 Transforms Reference

All transforms in `v2` are aware of `tv_tensors` (Image, BoundingBoxes, Mask) and apply geometrically consistent transformations across all of them when passed as a tuple.

---

### Geometric (apply to both image and boxes)

**`Resize(size)`** — Resizes to exact `[H, W]`. Stretches to fit regardless of aspect ratio. Always the last spatial op in this pipeline to normalise variable sizes to 640×640. Scales bounding box coordinates proportionally.

**`RandomHorizontalFlip(p=0.5)`** — Mirrors image left-right with probability p. Flips `cx → W - cx` on bounding boxes. Appropriate for fish (swim both directions).

**`RandomVerticalFlip(p=0.5)`** — Mirrors image top-bottom. Not appropriate for fish — they don't orient upside down in real footage.

**`ScaleJitter(target_size, scale_range)`** — Randomly resizes input so the shorter side falls between `scale_range[0] * min(target_size)` and `scale_range[1] * min(target_size)`. Output size is variable — must be followed by `Resize` to fix to 640×640. Scales bounding box coordinates with the image.

**`RandomResizedCrop(size, scale, ratio)`** — Crops a random region then resizes to `size`. `scale` controls the fraction of the original area in the crop. Produces fixed-size output without needing a trailing `Resize`. More aggressive than ScaleJitter for small object augmentation.

**`RandomIoUCrop(min_scale, max_scale, min_aspect_ratio, trials)`** — Crops to a region that satisfies a minimum IoU with at least one bounding box. Ensures cropped images always contain at least partial objects. Good for forcing tight bbox regression.

**`RandomZoomOut(fill, side_range)`** — Pads the image with fill colour and scales down content. Simulates viewing from farther away. Useful for small object augmentation.

**`RandomRotation(degrees)`** — Rotates by a random angle within ±`degrees`. Rotates bounding boxes accordingly. Small values (±10°) simulate camera tilt.

**`RandomAffine(degrees, translate, scale, shear)`** — Combined rotation, translation, scale, and shear in one transform. More efficient than chaining. Boxes transformed accordingly.

**`RandomPerspective(distortion_scale, p)`** — Applies a random four-point perspective warp. Simulates extreme camera angles. `distortion_scale` 0–1; keep below 0.3 for realistic underwater footage.

**`Pad(padding, fill)`** / **`RandomCrop(size)`** — Pad then crop. Alternative to ScaleJitter for scale variation that preserves aspect ratio.

---

### Colour (apply to image only, boxes pass through)

**`ColorJitter(brightness, contrast, saturation, hue)`** — Random perturbation of colour properties. Applied in PIL space before tensor conversion.
- `brightness`: multiplier range [1-b, 1+b]
- `contrast`: same
- `saturation`: same; 0 = grayscale, 2 = hyper-saturated
- `hue`: additive shift in [-h, +h] of the hue wheel (max 0.5)

**`RandomGrayscale(p)`** — Converts to grayscale with probability p, but keeps 3 channels (replicates grey across RGB). Simulates monochrome cameras in LILA source datasets.

**`GaussianBlur(kernel_size, sigma)`** — Applies Gaussian blur. Simulates backscatter and low-visibility water conditions. `kernel_size` must be odd. `sigma` controls spread.

**`RandomAdjustSharpness(sharpness_factor, p)`** — Adjusts sharpness. `sharpness_factor=0` = fully blurred (like `GaussianBlur`). `sharpness_factor=2` = enhanced edges. Simulates murky vs. clear water.

**`RandomAutocontrast(p)`** — Stretches the histogram to use the full [0, 255] range. Can improve generalisation across varying exposure levels in different LILA sources.

**`RandomEqualize(p)`** — Applies histogram equalisation. More aggressive than autocontrast. Can overcorrect in already well-exposed images.

**`RandomPosterize(bits, p)`** — Reduces each channel to `bits` bits (2–8). Simulates compression artefacts or low bit-depth cameras.

**`RandomSolarize(threshold, p)`** — Inverts pixels above `threshold`. Rarely used; can simulate overexposure in shallow water.

---

### Erasing (apply to tensor after ToImage/ToDtype)

**`RandomErasing(p, scale, ratio, value)`** — Randomly zeros (or fills with noise) a rectangular region of the tensor. Forces the model to not rely on a single discriminative patch. `scale` controls the fraction of image area erased; `ratio` controls aspect ratio of the erased region. Applied after conversion to tensor.

---

### Type Conversion (always last)

**`ToImage()`** — Converts PIL Image to a `tv_tensors.Image` (uint8 tensor subclass). Must come before `ToDtype`.

**`ToDtype(dtype, scale=True)`** — Converts tensor dtype. `scale=True` with `dtype=torch.float32` divides by 255, mapping [0, 255] → [0.0, 1.0]. Always the final transform.

---

### Composition

**`Compose(transforms)`** — Chains transforms sequentially. Each transform receives the output of the previous one.

**`RandomApply(transforms, p)`** — Applies a list of transforms with probability p as a group.

**`RandomChoice(transforms)`** — Applies exactly one randomly selected transform from the list.

**`RandomOrder(transforms)`** — Applies all transforms in a random order each call.

---

## Pipeline Order Rules

1. Geometric transforms that change spatial dimensions (ScaleJitter, RandomResizedCrop) must come **before** the terminal `Resize`.
2. Colour transforms (ColorJitter, GaussianBlur) can go anywhere before `ToImage` — they operate on PIL.
3. `ToImage` → `ToDtype` must be **last** — everything after this operates on float tensors.
4. `RandomErasing` must come **after** `ToDtype` since it operates on the float tensor.
5. `BoundingBoxes.canvas_size` must match the actual image dimensions at the moment the transform pipeline starts. Update it from the loaded PIL before calling `transform(image, boxes)`.
