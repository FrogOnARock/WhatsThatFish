# WhatsThisFish

Fish identification tool for divers. Two-stage pipeline: a YOLO11l object detector (trained on the LILA Community Fish Detection Dataset) feeds cropped detections into a hierarchical multi-class species classifier (~1,500 species, three heads: species / genus / subfamily).

## Overview

WhatsThisFish detects and identifies fish species in underwater images and video. It combines two large-scale open datasets — iNaturalist (3.57M photos, 15,564 taxa) and the LILA Community Fish Detection Dataset (~103K sampled, domain-balanced images) — with a custom ResNet backbone extended to 5-channel input (RGB + Scharr gradient + LCN) and a YOLO11l detector fine-tuned on underwater imagery.

## Project Structure

```
whatsthatfish/
├── src/
│   ├── config.py                      # App config loader (S3, GCS, paths)
│   ├── retry.py                       # Shared tenacity retry decorators (db, s3, gcs, transfer)
│   ├── etl/                           # Data ingestion pipeline
│   │   ├── factory.py                 # Async orchestration (ALL / CLASSIFICATION / DETECTION)
│   │   ├── inaturalist_dataset.py     # iNaturalist S3 ingestion (taxa, observations, photos)
│   │   ├── download_lila.py           # LILA downloader: COCO parsing, balanced sampling, GCS upload
│   │   ├── photo_transfer.py          # Async S3→GCS transfer with WAL+Postgres crash recovery
│   │   └── gcs_client.py             # GCS client with auth and resumable uploads
│   ├── database/                      # Database layer
│   │   ├── base.py                    # SQLAlchemy declarative base with naming conventions
│   │   ├── config.py                  # Sync (psycopg2) + async (asyncpg) engine factories
│   │   └── models.py                  # All ORM models
│   ├── preprocessing/                 # Image scoring and annotation pipeline
│   │   ├── factory.py                 # PreProcessingFactory routing
│   │   ├── uiqm_quality_scorer.py     # UIQM: UICM + UISM + UICONM composite scorer
│   │   ├── capture_context_scorer.py  # Heuristic underwater classifier (channel means)
│   │   ├── clip_context.py            # CLIP ViT-B/32 underwater classifier
│   │   ├── score_runner.py            # Async scoring orchestration + WAL+Postgres tracking
│   │   ├── annotation_conversion.py   # COCO→YOLO bbox conversion (lila_yolo table)
│   │   └── prepare_inat.py            # inat_classification_dataset build (cluster split, label encoding)
│   ├── transforms/                    # Multi-channel image transforms
│   │   ├── five_channel_conversion.py # AddMultiChannel: PIL → (5, H, W) tensor (RGB + gradient + LCN)
│   │   └── lcn_gradient_map.py        # Scharr gradient map + local contrast normalization
│   ├── models/                        # Datasets, dataloaders, model architectures
│   │   ├── od_dataset.py              # ObjectDetectionDataset (GCS streaming, UIQM-weighted sampler)
│   │   ├── od_dataloader.py           # ODDataLoader + CustomDetectionTrainer (Ultralytics override)
│   │   ├── c_custom_resnet.py         # CustomResnet: BasicBlock, 5-channel input, 3 hierarchical heads
│   │   ├── c_dataset.py               # ClassificationDataset (GCS streaming, on-the-fly bbox crop)
│   │   └── c_dataloader.py            # Classification collate functions (custom + ultralytics variants)
│   ├── training/
│   │   ├── od_training.py             # YOLO11l training: Ray Tune HPO + train_final (saves od_best.pt)
│   │   ├── INIT_NOTES.md              # CustomResnet pretrained weight loading strategy (5-channel)
│   │   └── TRAINING_ISSUES.md
│   ├── inference/
│   │   └── bbox_inference.py          # BoundingBoxInference: highest-confidence box from YOLO11l
│   ├── evaluation/
│   │   └── evaluate.py                # mAP evaluation framework
│   └── config/
│       ├── class_config.yaml          # YOLO data config (train/val paths, class names)
│       ├── train_config.yaml          # YOLO training config (tuned hyperparameters)
│       ├── TUNING_GUIDE.md            # Outcome-driven hyperparameter reference (conf×iou table)
│       └── REFERENCE.md               # Full Ultralytics params + torchvision v2 transforms reference
├── alembic/                           # Database migrations (13 versions)
├── tests/                             # 71 unit tests
│   ├── test_od_dataloader.py          # Collate dict API + ObjectDetectionDataset (mocked GCS/DB)
│   ├── test_c_classification.py       # BasicBlock + CustomResnet forward pass shapes
│   ├── test_bbox_inference.py         # BoundingBoxInference: confidence selection, coord clipping
│   ├── test_c_dataloader.py           # collate_fn + collate_fn_ultralytics
│   ├── test_five_channel.py           # AddMultiChannel output shape, dtype, channel content
│   ├── test_lcn_gradient.py           # gradient_map + local_contrast_normalization
│   ├── test_scoring.py                # UIQM, ContextScorer, WAL, DB compaction, runner tracking
│   ├── test_pipeline.py               # TransferProgressTracker WAL + crash recovery
│   ├── test_factory.py                # DataFactory pipeline routing
│   └── test_integration.py            # Integration tests (requires Postgres docker)
├── data/                              # Local parquet files + LILA metadata
├── weights/                           # Saved model weights
│   └── od_best.pt                     # Best YOLO11l detector weights (mAP@0.5:0.95 = 0.564)
├── alembic.ini
└── pyproject.toml
```

## Architecture

### Stage 1 — Object Detector (Training Complete)

YOLO11l fine-tuned on the LILA Community Fish Detection Dataset (~103K images, binary fish/no-fish).

- **Dataset** — Domain-balanced sampling across 17 sources (5K pos / 12.5K neg per source, then per-image 1:1 rebalancing). Served via `ObjectDetectionDataset` with GCS streaming and UIQM-weighted sampling.
- **Trainer** — `CustomDetectionTrainer` subclasses Ultralytics `DetectionTrainer`; skips double-normalization, uses custom collate that returns Ultralytics-compatible batch dicts.
- **Hyperparameter tuning** — Ray Tune over 8 trials (lr0, box, cls, weight_decay, dfl); best config written to `train_config.yaml`.
- **Results** — mAP@0.5: **0.774**, mAP@0.5:0.95: **0.564** (peaked epoch 30 of 50; best weights in `weights/od_best.pt`).
- **Inference** — `BoundingBoxInference` returns highest-confidence box with coordinates clipped to image bounds.

### Stage 2 — Hierarchical Classifier (In Progress)

Three-head ResNet predicting species, genus, and subfamily simultaneously from cropped fish images.

- **Architecture** — `CustomResnet` (BasicBlock, ResNet34-equivalent depth): 5-channel input (RGB + Scharr gradient + LCN), `AdaptiveAvgPool2d`, three independent linear heads.
- **Weight initialization** — Pretrained ResNet34 ImageNet weights for RGB channels; mean of RGB weights for channels 3–4 (gradient, LCN). See `training/INIT_NOTES.md`.
- **Warmup strategy** — Freeze backbone for N epochs; unfreeze via `optimizer.add_param_group` (preserves momentum on first conv).
- **Loss** — Weighted CrossEntropyLoss across heads: 0.5 × species + 0.3 × genus + 0.2 × subfamily.
- **Dataset** — `ClassificationDataset`: GCS original images, on-the-fly crop from `proposed_bbox`, 5-channel `AddMultiChannel` transform, geographic cluster-based train/val split.

### Data Pipeline (Complete)

- **iNaturalist** — 3.57M photo records (Actinopterygii + Chondrichthyes, research-grade, active taxa); lazy Parquet streaming to avoid OOM on 400M+ row tables.
- **LILA** — COCO JSON parsing, domain-balanced sampling, GCS upload (~103K images).
- **Photo transfer** — Async S3→GCS streaming of ~450K iNat classification photos (300/taxon cap). WAL+Postgres crash recovery.
- **UIQM scoring** — Quality metric (colorfulness + sharpness + contrast); weights UIQM-positive samples higher during training.
- **CLIP context** — ViT-B/32 underwater classifier; only underwater photos enter UIQM scoring and classification dataset.

### Database (Complete)

PostgreSQL with SQLAlchemy 2.0 ORM and Alembic migrations (13 versions).

| Table | Purpose |
|---|---|
| `inat_taxa` | ~44K active taxa |
| `inat_filtered_observations` | 3.57M photo records |
| `lila_collected_images` | ~103K LILA sampled images |
| `lila_annotations` | COCO bboxes (x, y, w, h) |
| `lila_yolo` | COCO→YOLO normalized annotations (JSONB) |
| `inat_clip_context` | CLIP underwater classification (0=above, 1=underwater) |
| `inat_capture_context` | Heuristic underwater classification |
| `inat_image_quality` | UIQM scores (uicm, uism, uiconm, uiqm) |
| `lila_image_quality` | UIQM scores for LILA images |
| `inat_classification_dataset` | Final classification dataset (cluster split, label indices, proposed_bbox) |
| `successful_uploads` | WAL compaction tracking |

## Taxa Scope

- **Actinopterygii** (47178) — ray-finned fishes: ~50K taxa
- **Chondrichthyes** (196614) — sharks, rays, chimaeras: ~2K taxa
- Filtered to active taxa only: **43,991 taxa**, of which **15,564 have research-grade observations**
- Classification target: **~1,500 species** with ≥300 underwater images (~450K total, 300/taxon cap)

## Setup

Requires Python >= 3.12 and PostgreSQL.

```bash
cd whatsthatfish
pip install -e .
```

### Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (`postgresql://user:pass@host:port/dbname`) |
| `GCS_SECRET` | Path to GCS service account key (optional, falls back to default credentials) |

### Database Setup

```bash
alembic upgrade head
```

### Running Tests

```bash
# Unit tests (no infrastructure required)
.venv/bin/pytest tests/ -v --ignore=tests/test_integration.py

# Integration tests (requires Postgres docker)
docker compose -f docker-compose.test.yml up -d
.venv/bin/pytest tests/test_integration.py -v
```

## Current Status

### Completed

- iNaturalist S3 ingestion (taxa, observations, photos → parquet → Postgres)
- LILA domain-balanced sampling (~103K images) + GCS upload
- iNaturalist S3→GCS photo transfer (~450K classification photos, WAL+Postgres crash recovery)
- COCO→YOLO annotation conversion (`lila_yolo` table)
- UIQM quality scoring (inat + lila)
- CLIP underwater context classification (ViT-B/32)
- Multi-channel transforms: Scharr gradient map + LCN (`AddMultiChannel`)
- `ObjectDetectionDataset` + `CustomDetectionTrainer` (GCS streaming, UIQM-weighted sampler, Ultralytics-compatible collate)
- YOLO11l detector training + Ray Tune HPO (8 trials); best weights saved
- `BoundingBoxInference` (highest-confidence box, coordinate clipping)
- `CustomResnet`: BasicBlock, 5-channel input, three hierarchical heads, Kaiming init
- `ClassificationDataset` + collate functions (custom multi-head + ultralytics single-label)
- 71 unit tests across all components

### Next Steps

1. **Label map encoding** — `prepare_inat.py`: write 0-indexed species/genus/subfamily label maps to YAML alongside checkpoint; required before classifier training can begin
2. **crop_export.py** — Run `BoundingBoxInference` over ~450K iNat GCS images; store `proposed_bbox` into `inat_classification_dataset`; prerequisite for `ClassificationDataset.__getitem__`
3. **c_training.py** — Custom training loop for `CustomResnet`: pretrained weight loading (see `training/INIT_NOTES.md`), warmup phase, weighted CrossEntropyLoss across three heads, checkpoint saving with label maps
4. **YOLO11-cls baseline** — Standard Ultralytics classification training on species head only; comparison point against custom ResNet
5. **OD evaluation** — Run `evaluate.py` against `od_best.pt` at conf=0.25; report mAP@0.5 and mAP@0.5:0.95
6. **Classification evaluation** — Top-1 and top-3 accuracy per head (species, genus, subfamily)
7. **Inference pipeline** — Wire detector → crop → classifier for batch images and video frames
8. **Production export** — ONNX / TorchScript export of both models for deployment

## License

Non-commercial use. Dataset includes CC-BY-NC licensed content from iNaturalist.
