# WhatsThisFish

Fish identification tool for divers. A two-stage vision pipeline: a **YOLO11l object detector** (LILA → iNaturalist learning curriculum) localises the fish, and a **CLIP/ImageNet-initialised hierarchical classifier** (~1,500 species) identifies the crop across three taxonomic heads (family / genus / species). A FastAPI service and a React frontend expose the trained models as a web app.

## Overview

WhatsThisFish detects and identifies fish in underwater images. It combines two large open datasets — iNaturalist (3.57M photos, ~15.5K research-grade taxa) and the LILA Community Fish Detection Dataset (~103K domain-balanced images) — with a custom ResNet-34 backbone extended to **5-channel input** (RGB + Scharr gradient + LCN) and a YOLO11l detector progressively fine-tuned on underwater imagery.

The system is built as four layers:

1. **Data pipeline** (`etl/`, `preprocessing/`) — S3/GCS ingestion, quality scoring, underwater context classification, train/val splitting.
2. **Models** (`models/`) — architectures, datasets, dataloaders, and one lifecycle facade per model (`Classifier`, `Detector`) exposing `train()` / `tune()` / `predict()`.
3. **Evaluation** (`evaluation/`) — hierarchical classification metrics and reports.
4. **Product** (`serving/`, `frontend/`) — FastAPI read API + React SPA species catalogue.

## Project Structure

```
whatsthatfish/
├── src/whatsthatfish/
│   ├── config.py                         # App config + logger
│   ├── retry.py                          # Shared tenacity retry decorators (db, s3, gcs)
│   ├── etl/                              # Data ingestion
│   │   ├── factory.py                    # Async orchestration (ALL / CLASSIFICATION / DETECTION)
│   │   ├── inaturalist_dataset.py        # iNat S3 ingestion (taxa, observations, photos)
│   │   ├── download_lila.py              # LILA: COCO parsing, balanced sampling, GCS upload
│   │   ├── photo_transfer.py             # Async S3→GCS transfer + WAL/Postgres crash recovery
│   │   └── gcs_client.py                 # GCS client (auth + resumable uploads)
│   ├── database/                         # SQLAlchemy 2.0 ORM
│   │   ├── base.py · config.py           # Declarative base; sync (psycopg2) + async (asyncpg) factories
│   │   └── models.py                     # 13 ORM models (see Database below)
│   ├── preprocessing/                    # Scoring, context, dataset prep
│   │   ├── factory.py                    # PreProcessingFactory routing (--type / --dataset)
│   │   ├── uiqm_quality_scorer.py        # UIQM: UICM + UISM + UICONM composite
│   │   ├── capture_context_scorer.py     # Heuristic underwater classifier (channel means)
│   │   ├── clip_context.py               # CLIP ViT-B/32 underwater classifier (incl. coral prompts)
│   │   ├── score_runner.py               # Async scoring + WAL/Postgres tracking
│   │   ├── annotation_conversion.py      # COCO→YOLO bbox conversion (lila_yolo)
│   │   ├── prepare_inat.py               # Geographic KMeans split → classification/OD datasets
│   │   ├── inat_zero_index.py            # 0-index family/genus/species label maps
│   │   ├── app_taxa.py                   # Build app_taxa catalogue + LLM-enriched descriptions
│   │   └── retrieve_gcs_images.py        # Async bulk GCS image download
│   ├── transforms/
│   │   ├── letterbox_resize.py           # Aspect-preserving square pad (PIL→PIL)
│   │   ├── five_channel_conversion.py    # AddMultiChannel: PIL → (5,H,W) tensor
│   │   └── lcn_gradient_map.py           # Scharr gradient + local contrast normalization
│   ├── models/                          # architectures, data, and lifecycle facades
│   │   ├── architecture/
│   │   │   └── custom_resnet.py          # CustomResnet: 5ch ResNet-34, 3 heads (progressive|parallel)
│   │   ├── datasets/                     # ObjectDetectionDataset · ClassificationDataset
│   │   ├── loaders/                      # od_dataloader (+CustomDetectionTrainer) · c_dataloader
│   │   ├── classifier.py                 # Classifier: train() / tune() / predict() (merged trainer+tuner)
│   │   └── detection.py                  # Detector: train() / tune() / predict() over YOLO11l
│   ├── inference/
│   │   ├── bbox_inference.py             # BoundingBoxInference: best-confidence YOLO box
│   │   └── inat_bbox_proposal.py         # Bbox proposal pipeline (classification | detection modes)
│   ├── evaluation/
│   │   └── cls_metrics.py                # Hierarchical macro metrics + HTML/sunburst/PR reports
│   ├── serving/                          # FastAPI read API
│   │   ├── app.py                        # /health · /species · /image/{filename}
│   │   ├── schemas.py                    # Pydantic response models
│   │   └── utils.py                      # StorageConstructor (local | GCS image retrieval)
│   └── config/                           # YAML configs + docs/ (TUNING_GUIDE, REFERENCE)
├── frontend/                            # React 18 + Vite + TypeScript SPA
│   └── src/                              # pages (Main/History/Library) · components · api client
├── alembic/                             # Database migrations
├── tests/                               # Unit + integration tests
├── data/                               # Local parquet + LILA metadata
├── weights/                            # od_best.pt · lc1_best.pt · yolo11l.pt
└── runs/                               # detect/ (YOLO) · classification/ (CustomResnet)
```

## Architecture

### Stage 1 — Object Detector

YOLO11l fine-tuned for binary fish/no-fish detection, then progressively adapted to underwater iNat imagery via a two-stage **learning curriculum**.

- **LILA base** — domain-balanced sampling across 17 sources (~103K images, ~1:1 pos/neg). Served by `ObjectDetectionDataset` (GCS streaming, UIQM-weighted sampler) through `CustomDetectionTrainer`, which subclasses Ultralytics `DetectionTrainer` and skips double-normalization. **Done** — mAP@0.5 **0.774**, mAP@0.5:0.95 **0.564** → `weights/od_best.pt`.
- **LC1** — iNat fish images ranked by UIQM (best quality first, up to 100K) from `inat_obj_detection_dataset`. Fine-tunes `od_best.pt` → `lc1_best.pt`. **Done** (Ray Tune sweeps `lc1_tune2_*`, `lc1_tune3_*`; best config in `config/lc1_train_config.yaml`).
- **LC2** — same images sampled by `uiqm × conf`. Fine-tunes `lc1_best.pt` → `lc2_best.pt`. **Not yet run.**
- **Coral negatives** — Anthozoa images (taxon 47533) are forced as `conf=1.0` empty-annotation negatives in detection-mode bbox proposal, without inference.

`_WEIGHTS` in `models/detection.py` (`Detector`) maps each stage to `(input, output)`: `lila`→(`yolo11l.pt`,`od_best.pt`), `lc1`→(`od_best.pt`,`lc1_best.pt`), `lc2`→(`lc1_best.pt`,`lc2_best.pt`). Run with `python -m whatsthatfish.models.detection --dataset lc1 --type tune|full`.

### Stage 2 — Hierarchical Classifier

`CustomResnet` (`models/architecture/custom_resnet.py`): ResNet-34-style BasicBlock backbone, **5-channel input** (RGB + Scharr gradient + LCN), three taxonomic heads computed **family → genus → species**. Driven by the `Classifier` facade (`models/classifier.py`) — `train()` / `tune()` / `predict()` over one `_fit()` core, so a full run is just tuning with a single fixed config.

- **Head topology (toggle `head_mode`)**:
  - `progressive` (default) — each head's logits are projected to a 64-dim bottleneck and concatenated onto the pooled features feeding the next, finer head; **parent logits are detached** so a child's loss never backprops into its parent.
  - `parallel` — three independent linear heads off the 512-dim pooled features (ablation baseline).
- **Pretrained init** — torchvision ResNet-34 ImageNet weights; the 7×7 stem is **inflated** to 5 channels (RGB copied, extra channels = mean of RGB filters, rescaled `3/in_dim`). From-scratch variant uses Kaiming init.
- **Curriculum loss weighting** (3 phases): `[0,0,1]` species-only → `[0,0.6,0.4]` +genus → `[0.6,0.3,0.1]` all three; transitions gated by time **and** performance gates from `cls_metrics.py`. Val loss always uses phase-3 weights so `best.pt` selection is comparable across phases.
- **Per-head loss** — inverse-frequency class-weighted CrossEntropy + `label_smoothing=0.1`.
- **Discriminative LRs** — pretrained variants use two optimizer groups (head/stem fast, backbone ~10× slower) under OneCycleLR; progressive backbone unfreeze after `freeze_epochs` (BN held in eval during warmup). From-scratch uses a single group.
- **Tuning** — `Classifier.tune()`: in-process random search over `cls_model_param_space_config.yaml` (no Ray workers — a 2nd CUDA process OOM'd the single GPU); each trial runs the **same `_fit()`** as `train()` on a fresh model/optimizer/scheduler/metrics, so no state bleeds between trials; lowest val loss wins. Run with `python -m whatsthatfish.models.classifier --type tune|full`.
- **Status** — facade complete (merged trainer + tuner; tuning forwarding resolved — `tune()` passes the full sampled config through `_fit`). Currently mid-LR-sweep on the pretrained progressive variant; best run so far: species Top-1 **0.745**, Top-3 **0.858**, genus Top-1 **0.774**, family Top-1 **0.762** (species-only phase, geographic val). (`freeze_epochs` is honored but not yet in the param space, so add it there to sweep it.)

### Data Pipeline

- **iNaturalist** — 3.57M photo records (Actinopterygii + Chondrichthyes + Anthozoa, research-grade, active taxa); lazy Parquet streaming for the 400M+ row tables.
- **LILA** — COCO parsing, domain-balanced sampling, GCS upload (~103K images), COCO→YOLO conversion.
- **Photo transfer** — async S3→GCS of ~450K iNat classification photos (300/taxon cap), WAL+Postgres crash recovery.
- **Quality + context** — UIQM scoring and CLIP ViT-B/32 underwater classification (16 above-water + 19 underwater prompts, incl. coral); only underwater photos enter UIQM scoring and the classification dataset.
- **Split + labels** — geographic KMeans clustering (`prepare_inat.py`) for an honest generalization val split; 0-indexed family/genus/species label maps (`inat_zero_index.py`).

### Product Layer

- **`serving/` (FastAPI)** — a thin synchronous read API reusing the existing SQLAlchemy stack: `GET /health`, `GET /species` (full catalogue from `app_taxa`), `GET /image/{filename}` (local or GCS via `StorageConstructor`). **No inference endpoint yet** — detector→classifier prediction is not exposed over HTTP.
- **`frontend/` (React 18 + Vite + TS)** — SPA with Main / History / Library pages, image DropZone, classification result cards, and a species library backed by `/species`. Predictions are currently **mock-driven** (`api/mock-predictions.ts`) pending the inference endpoint.
- **`app_taxa.py`** — builds the `app_taxa` catalogue by joining the classification dataset with taxonomy, then enriches each species with LLM-generated description / location / depth (Claude `claude-sonnet-4-6`, concurrent, COALESCE null-guarded upserts).

## Database

PostgreSQL + SQLAlchemy 2.0 ORM with Alembic migrations. 13 tables:

| Table | Purpose |
|---|---|
| `inat_taxa` | ~44K active taxa (taxon_id, ancestry, rank, name) |
| `inat_filtered_observations` | 3.57M photo records keyed by photo_uuid |
| `lila_collected_images` | ~103K LILA sampled images (width/height) |
| `lila_annotations` | COCO bboxes (x, y, w, h) |
| `lila_yolo` | COCO→YOLO normalized annotations (JSONB) |
| `inat_clip_context` | CLIP underwater classification (0=above, 1=underwater) |
| `inat_capture_context` | Heuristic underwater classification |
| `inat_image_quality` · `lila_image_quality` | UIQM scores (uicm, uism, uiconm, uiqm) |
| `inat_classification_dataset` | Classifier training set (cluster split, 0-indexed labels, proposed_bbox crops) |
| `inat_obj_detection_dataset` | LC1/LC2 detector training set (+ Anthozoa negatives) |
| `app_taxa` | Serving catalogue: species + LLM-enriched metadata |
| `successful_uploads` | WAL compaction tracking |

## Taxa Scope

- **Actinopterygii** (47178) — ray-finned fishes
- **Chondrichthyes** (196614) — sharks, rays, chimaeras
- **Anthozoa** (47533) — corals/anemones (object-detection negatives only; excluded from classification via ancestry filter)
- Classification target: **~1,500 species** with ≥300 underwater images (~450K total, 300/taxon cap, 80/20 geographic train/val split)

## Setup

Requires Python ≥ 3.12, PostgreSQL, and (for the frontend) Node 18+.

```bash
# Backend
cd whatsthatfish
pip install -e .          # or: uv sync
alembic upgrade head

# Frontend
cd frontend
npm install
npm run dev               # Vite dev server on :5173
```

### Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `GCS_SECRET` | Path to GCS service-account key (falls back to default credentials) |
| `ANTHROPIC_API_KEY` | For `app_taxa.py` species-description enrichment |

### Running the API

```bash
uv run uvicorn whatsthatfish.serving.app:app --reload --port 8000
# docs at http://localhost:8000/docs
```

### Running Tests

```bash
.venv/bin/pytest tests/ -v --ignore=tests/test_integration.py   # unit
docker compose -f docker-compose.test.yml up -d                 # integration (Postgres)
.venv/bin/pytest tests/test_integration.py -v
```

## Current Status

### Done
- iNat S3 ingestion (taxa, observations, photos), LILA balanced sampling + GCS, S3→GCS photo transfer (WAL recovery)
- UIQM scoring, CLIP underwater context, COCO→YOLO conversion, geographic KMeans split, 0-indexed labels
- `ObjectDetectionDataset`/`ClassificationDataset` + dataloaders; all 5-channel transforms
- **YOLO11l detector trained** (`od_best.pt`, mAP@0.5 0.774) and **LC1 fine-tune** (`lc1_best.pt`)
- `CustomResnet` (progressive/parallel heads, 5ch stem inflation) + `Classifier` facade (`models/classifier.py`: curriculum, discriminative LRs, merged train/tune/predict)
- `cls_metrics.py` hierarchical metrics + HTML/sunburst/PR reports
- `BoundingBoxInference`, two-mode `inat_bbox_proposal.py`, per-dataset YOLO configs + Ray Tune spaces
- FastAPI serving (`/health`, `/species`, `/image`), `app_taxa` catalogue with LLM enrichment, React frontend scaffold

### In Progress
- **Classifier LR sweep** (`python -m whatsthatfish.models.classifier --type tune`) on the pretrained progressive variant
- Bbox proposals (detection mode) populating `inat_obj_detection_dataset.annotation`

### Not Started
- LC2 fine-tune (`lc2_best.pt`); classification-mode bbox proposals (need final LC2 model)
- Post-OD evaluation at conf=0.15 (Recall@0.5 target)
- Detector→classifier inference path + HTTP predict endpoint (frontend still on mocks)
- ONNX export, Cloud Run serving, training VM automation, drift logging, CI

## Evaluation Targets

**Detector (YOLO11l):** mAP@0.5 ≥ 0.75 · mAP@0.5:0.95 ≥ 0.50 · Recall@0.5 ≥ 0.90 (survey mode, conf=0.15)
**Classifier (1,500 species, macro on geographic val):** species Top-1 ≥ 65% / Top-3 ≥ 80% · genus Top-1 ≥ 78% · family Top-1 ≥ 88%

## License

Non-commercial use. Includes CC-BY-NC content from iNaturalist.
