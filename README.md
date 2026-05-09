# WhatsThisFish

Fish identification tool for divers. Two-stage pipeline: a YOLO11 object detector (trained on the LILA Community Fish Detection Dataset) feeds cropped detections into a multi-class species classifier (~1,500 species).

## Overview

WhatsThisFish detects and identifies fish species in underwater images and video. It combines two large-scale open datasets — iNaturalist (3.57M photos, 15,564 taxa) and the LILA Community Fish Detection Dataset (~103K sampled, domain-balanced images) — with a modified YOLO model that processes multi-channel input to improve detection of camouflaged species in complex reef environments.

## Project Structure

```
whatsthatfish/
├── src/
│   ├── config.py                      # App config loader (S3, GCS settings)
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
│   │   └── annotation_conversion.py   # COCO→YOLO bbox conversion (lila_yolo table + .txt files)
│   ├── model/                         # Multi-channel YOLO model (to be built)
│   ├── training/                      # Training pipeline (to be built)
│   ├── inference/                     # Inference pipeline (to be built)
│   ├── evaluation/                    # Evaluation framework (to be built)
│   └── augmentation/                  # Augmentation pipeline (to be built)
├── alembic/                           # Database migrations (13 versions)
├── tests/                             # 50+ unit tests
├── data/                              # Local parquet files + LILA metadata
├── alembic.ini
└── pyproject.toml
```

## Architecture

### Data Pipeline (Complete)

- **iNaturalist S3** — Downloads taxonomy, observations, and photo metadata from the public `inaturalist-open-data` bucket. Converts CSV to Parquet and builds filtered datasets using lazy streaming (`pl.scan_parquet()` + `sink_parquet()`) to handle 400M+ row tables without OOM.
- **LILA** — Parses COCO JSON annotations, applies two-phase domain-balanced sampling (~103K images, 1:1 pos/neg, all 17 sources represented), downloads images and uploads to GCS.
- **Photo Transfer** — Async S3→GCS streaming of ~450K iNat classification photos (capped at 300/taxon). WAL+Postgres crash recovery ensures resume safety.
- **GCS** — `whats-that-fish` bucket with prefixes: `training/`, `validation/`, `contributions/`, `object_detection/`.

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
| `successful_uploads` | WAL compaction tracking |

### Preprocessing (Complete)

- **UIQM scoring** — Three sub-scores (colorfulness, sharpness, contrast) compose a single quality metric. iNat photos are only scored if marked underwater by CLIP.
- **CLIP context classification** — ViT-B/32 with 17 above-water and 14 underwater text prompts. Replaced earlier heuristic channel-mean approach.
- **COCO→YOLO conversion** — Normalizes bounding box coordinates to YOLO center format, stores in `lila_yolo` table and writes per-image `.txt` annotation files.

### Preprocessing (Multi-Channel Input) — To Be Built

Standard RGB augmented with additional channels (gradient magnitude, local contrast normalization) to improve detection of camouflaged fish against complex reef backgrounds. Requires modifying YOLO's first conv layer (duplicate RGB weights into extra channels to preserve pretrained features).

### Model, Training, Inference, Evaluation — To Be Built

- **Detector** — YOLO11l fine-tuned on LILA (~103K images, binary fish/no-fish); optional stage 3 domain adaptation
- **Classifier** — YOLO11-cls or CLIP fine-tune on ~450K iNat crops, ~1,500 species
- **Inference** — Detector → crop export → classifier pipeline for images and video
- **Evaluation** — Targets: mAP@0.5 ≥ 0.75, Recall@0.5 ≥ 0.90, cross-domain generalization

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

## Current Status

### Completed

- iNaturalist S3 ingestion (taxa, observations, photos → parquet → Postgres)
- LILA domain-balanced sampling (~103K images) + GCS upload + image download
- iNaturalist S3→GCS photo transfer (~450K classification photos, WAL+Postgres crash recovery)
- COCO→YOLO annotation conversion (`lila_yolo` table + `.txt` files)
- UIQM quality scoring (inat + lila)
- CLIP underwater context classification (ViT-B/32, replaced heuristic approach)
- Full database schema (10 tables, 13 Alembic migrations)
- 50+ unit tests (WAL behavior, DB compaction, scoring, factory routing)

### Next Steps

1. **Dataset YAML** — train/val split config pointing at `lila_yolo` `.txt` files
2. **Training pipeline** — YOLO11l detector on LILA; stage 2 fine-tune + stage 3 domain adaptation
3. **Multi-channel YOLO mod** — extend first conv layer for gradient map + LCN extra channels
4. **Inference pipeline** — detect → crop → classify for images and video
5. **Classification model** — YOLO11-cls on ~450K iNat crops, ~1,500 species
6. **Evaluation framework** — mAP, recall, cross-domain generalization metrics

## License

Non-commercial use. Dataset includes CC-BY-NC licensed content from iNaturalist.
