# Data Sources

Two large open datasets feed the pipeline: **iNaturalist** (classification +
detection curriculum) and the **LILA Community Fish Detection Dataset** (detector
base training).

---

## iNaturalist Open Data

Public S3 bucket `inaturalist-open-data` — **unsigned/anonymous**, accessed over
HTTPS with `aiohttp` (no credentials). Ingested to Parquet, then Postgres.

### Scope (taxonomic filter)

| Class | taxon_id | Role |
|---|---|---|
| **Actinopterygii** | 47178 | ray-finned fishes — classification + detection |
| **Chondrichthyes** | 196614 | sharks, rays, chimaeras — classification + detection |
| **Anthozoa** | 47533 | corals/anemones — **detection negatives only** (excluded from classification via ancestry filter) |

Only **research-grade**, **active-taxa** photos are kept.

### Volumes

| File / table | Size | Notes |
|---|---|---|
| `taxa.parquet` | 1.6M rows | full iNat taxonomy |
| `observations.parquet` | ~400M rows | lazy-scan only (`pl.scan_parquet` + `sink_parquet`) |
| `photos.parquet` | ~413M rows | lazy-scan only |
| `dataset.parquet` | 3.57M records | filtered photo set (the working corpus) |

> Large Parquet files (photos, observations) **must** be processed lazily —
> materialising 400M rows OOMs the machine.

### Pipeline stages

1. **Ingest** (`etl/inaturalist_dataset.py`) — taxa / observations / photos →
   Parquet → filtered `dataset.parquet` → Postgres.
2. **Photo transfer** (`etl/photo_transfer.py`) — async S3→GCS of ~450K
   classification photos, capped **300/taxon**, with **WAL + Postgres crash
   recovery** (resumable if interrupted).
3. **Underwater context** (`preprocessing/clip_context.py`) — CLIP ViT-B/32
   classifies each photo above-water vs underwater (16 + 19 prompts, incl. coral
   reef prompts so coral frames read as underwater). Only underwater photos
   proceed.
4. **Quality scoring** (`preprocessing/uiqm_quality_scorer.py`) — UIQM
   (UICM + UISM + UICONM) on underwater photos; drives the training sampler and
   the LC1/LC2 curriculum ordering.
5. **Split + labels** (`preprocessing/prepare_inat.py`, `inat_zero_index.py`) —
   geographic KMeans split (below) + 0-indexed family/genus/species label maps.

### Classification target

**~1,500 species** with ≥300 underwater images, **300/taxon cap**, ~450K total,
**80/20 geographic train/val split**.

---

## LILA Community Fish Detection Dataset

COCO-format annotations: **1.9M images, 935K bboxes, 17 source sub-datasets**,
with severe domain imbalance (`salmon_cv` alone ≈ 73%).

- **Domain-balanced sampling** (`etl/download_lila.py`) → ~**103K images**
  (~51.8K positive, ~51.8K negative), **all 17 sources represented**, uploaded to
  GCS `object_detection/`.
- **COCO→YOLO conversion** (`preprocessing/annotation_conversion.py`) — outer-join
  images+annotations, normalize xywh, upsert to `lila_yolo` (JSONB).

---

## Geographic train/val split (the honest split)

The val set must measure **generalization to new places**, not memorised sites.
`prepare_inat.py` runs **GPU KMeans on observation lat/lon (radians)**, then holds
out whole clusters for validation — so no dive site straddles the train/val
boundary.

- **K = 80** clusters was selected via a sweep (elbow / silhouette). The decision
  record — cluster visualisations and the search log — lives in
  [`docs/artifacts/kmeans_search/`](./artifacts/kmeans_search/)
  (`best_k.txt`, `kmeans_log.txt`, `kmeans_search.png`, per-K cluster plots).
- The val split carries a **`val_topup`** flag distinguishing two regimes,
  reported separately so headline macro metrics aren't contaminated:
  - **geographic held-out** — whole clusters withheld → true generalization
    (**this is the headline eval**).
  - **IID top-up** — rare-class rows added for coverage → *not* a generalization
    signal.

See [MODELMETRICS.md](./MODELMETRICS.md) for how these two regimes are reported.

---

## Where the data lands (Postgres)

| Table | Purpose |
|---|---|
| `inat_taxa` | ~44K active taxa (taxon_id, ancestry, rank, name) |
| `inat_filtered_observations` | 3.57M photo records keyed by photo_uuid |
| `lila_collected_images` | ~103K LILA sampled images (width/height) |
| `lila_annotations` · `lila_yolo` | COCO bboxes · COCO→YOLO normalized (JSONB) |
| `inat_clip_context` · `inat_capture_context` | CLIP · heuristic underwater classification |
| `inat_image_quality` · `lila_image_quality` | UIQM scores (uicm, uism, uiconm, uiqm) |
| `inat_classification_dataset` | classifier training set (cluster split, 0-indexed labels, crop bboxes) |
| `inat_obj_detection_dataset` | LC1/LC2 detector set (+ Anthozoa negatives) |
| `app_taxa` | serving catalogue: species + LLM-enriched description/location/depth |
| `successful_uploads` | WAL compaction tracking |

GCS bucket `whats-that-fish` prefixes: `training/`, `validation/`,
`contributions/`, `object_detection/`.

---

## Licensing

Non-commercial. Includes **CC-BY-NC** content from iNaturalist; LILA sources
retain their individual licenses.
