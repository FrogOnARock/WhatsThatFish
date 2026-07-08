# WhatsThisFish

Fish identification for divers. A two-stage vision pipeline — a **YOLO11l object
detector** (LILA → iNaturalist learning curriculum) localises the fish, and a
**5-channel hierarchical classifier** (~1,500 species) identifies the crop across
three taxonomic heads (family / genus / species). A torch-free FastAPI + ONNX
service and a React frontend expose the trained models as a web app.

```
photo ──▶ YOLO11l detector ──▶ crop ──▶ CustomResnet ──▶ {family, genus, species} + bbox + underwater conf
```

## Documentation

| Doc | Contents |
|---|---|
| **[MODELDESIGN](docs/MODELDESIGN.md)** | Both model architectures; the **5-channel input** (RGB + Scharr + LCN) and how the extra channels are computed; heads, curriculum, ONNX serving parity. |
| **[DATASOURCE](docs/DATASOURCE.md)** | iNaturalist + LILA ingestion, taxonomic scope, quality/context filtering, and the **geographic** train/val split. |
| **[MODELMETRICS](docs/MODELMETRICS.md)** | Targets vs observed (detector mAP/recall, classifier macro Top-1/3, INT8 gate). |
| **[NEXTSTEPS](docs/NEXTSTEPS.md)** | Remaining CI/CD + scheduled (CRON) jobs: delta ingest, training gate, retraining VM, drift alerting. |

## Architecture

Four layers:

1. **Data pipeline** (`etl/`, `preprocessing/`) — S3/GCS ingestion, UIQM quality
   scoring, CLIP underwater context, geographic train/val split.
2. **Models** (`models/`) — architectures, datasets, dataloaders, and one
   lifecycle facade per model (`Classifier`, `Detector`) exposing
   `train()` / `tune()` / `predict()`.
3. **Evaluation** (`evaluation/`) — hierarchical classification metrics + reports.
4. **Product** (`serving/`, `frontend/`) — FastAPI API (detector→classifier
   `/predict`, species catalogue, observation tracking) + React SPA.

See [MODELDESIGN](docs/MODELDESIGN.md) for the detector curriculum
(LILA → LC1 → LC2), the classifier's 5-channel input and progressive heads, and
the ONNX serving path.

## Taxa scope

- **Actinopterygii** (47178) — ray-finned fishes
- **Chondrichthyes** (196614) — sharks, rays, chimaeras
- **Anthozoa** (47533) — corals/anemones (**detection negatives only**; excluded
  from classification via ancestry filter)

Classification target: **~1,500 species** with ≥300 underwater images (~450K
total, 300/taxon cap, 80/20 geographic split). Full data details in
[DATASOURCE](docs/DATASOURCE.md).

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

### Environment variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `GCS_SECRET` | Path to GCS service-account key (falls back to default credentials) |
| `GOOGLE_OAUTH_CLIENT_ID` | Google ID-token audience for serving auth |
| `ANTHROPIC_API_KEY` | For `app_taxa.py` species-description enrichment |
| `ALLOWED_ORIGINS` | Comma-separated prod CORS origins (localhost auto-allowed off Cloud Run) |

### Running the API

```bash
uv run uvicorn whatsthatfish.serving.main:app --reload --port 8000
# docs at http://localhost:8000/docs
```

### Running tests

```bash
docker compose -f docker-compose.test.yml up -d      # test Postgres (:5433)
.venv/bin/pytest tests/ -q
```

## Status

**Trained & working:** full data pipeline; YOLO11l detector (`od_best.pt`) + LC1
fine-tune (`lc1_best.pt`, all detection targets met incl. conf=0.15 survey
recall); **final classifier train complete** (`CustomResnet` + `Classifier`
facade, species/genus targets met); hierarchical metrics/reports; torch-free ONNX
serving (`/predict`, `/species`, observation tracking) with INT8 classifier; React
SPA; `release.yml` model-release gate (ONNX export + INT8/parity checks + image
build); Cloud Run + Cloud SQL + GCS + OAuth/CORS **provisioned in prod**.

**Remaining:** family-accuracy improvement; Secret Manager migration; frontend CI
+ Cloudflare deploy; the scheduled retraining/drift loop; LC2 detector *(deferred
— LC1 already clears every target)* — all tracked in
[NEXTSTEPS](docs/NEXTSTEPS.md).

## License

Non-commercial use. Includes CC-BY-NC content from iNaturalist; LILA sources
retain their individual licenses.
