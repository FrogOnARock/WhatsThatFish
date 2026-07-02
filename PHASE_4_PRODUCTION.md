# Phase 4 — Production (ONNX export · serving · frontend)

Three deliverables — (A) ONNX export, (B) serving container on Cloud Run, (C) frontend
hosting — plus the connective tissue (Cloud SQL, GCS contributions + signed URLs, secrets,
CI/CD, OAuth origins). Sequenced at the bottom; do **A first** — it's the gate for the whole
CPU-serving thesis.

---

## A. ONNX export

The two models export independently; the **preprocessing does NOT live in the ONNX graph**
(it's PIL/numpy) and must be replicated byte-for-byte at serving time.

### A.1 Detector (YOLO11) — Ultralytics native
```python
from ultralytics import YOLO
YOLO("od_best.pt").export(format="onnx", nms=True, dynamic=True, simplify=True, opset=17, imgsz=640)
```
- **`nms=True`** bakes NMS into the graph → the ONNX output is already final boxes, so the
  serving inferrer doesn't reimplement NMS. Worth it.
- `imgsz=640` must match training. `dynamic=True` for variable batch.
- Output: `od_best.onnx`. Validate it still hits Recall@0.5 at conf=0.15.

### A.2 Classifier (CustomResnet) — torch.onnx.export
- **Reuse the self-describing checkpoint** (the `arch` block we added: `layers/in_dim/head_mode`).
  `Classifier.predict()` already rebuilds the exact graph from it — the export script does the
  same, then `torch.onnx.export`.
- Input: fixed `(N, 5, 320, 320)` (post-letterbox). Three **named outputs**
  (`species`, `genus`, `family`). `dynamic_axes` on batch only. `opset>=17`.
```python
torch.onnx.export(
    model.eval(), dummy_5ch, "classifier.onnx",
    input_names=["input"], output_names=["species", "genus", "family"],
    dynamic_axes={"input": {0: "batch"}, "species": {0: "batch"}, ...}, opset_version=17,
)
```

### A.3 Preprocessing parity (the sharp edge)
The crop-margin → `LetterboxResize(320)` → `AddMultiChannel` (5-channel: RGB + Scharr + LCN)
chain stays in Python. **`AddMultiChannel` currently returns a torch tensor** — write a
**numpy-only** variant for serving so the container needs no torch (the math is identical; just
emit `np.ndarray` instead of a tensor). This is the single biggest container-size lever.
- Add a parity test: `onnxruntime` output vs torch output on a val sample (top-1 agreement, or
  logits within ~1e-3). Put it in the suite.

### A.4 Quantization (the CPU gate)
- Dynamic INT8 via `onnxruntime.quantization.quantize_dynamic` on both models.
- **Decision gate:** re-measure macro Top-1 on the **geographic** val split. If it drops
  materially, ship FP32 or fall back to GPU serving. Also measure p50/p95 CPU latency — this is
  what decides whether the "Cloud Run CPU, scale-to-zero" thesis holds.

---

## B. Serving container (Cloud Run, CPU, scale-to-zero)

### B.1 ONNX inferrers — same interface, swapped backend
- Write `OnnxBoundingBoxInference` and `OnnxClassInference` that implement the **same `.infer()`
  signatures + return shapes** as `inference/bbox_inference.py` / `inference/class_inference.py`
  (the classifier one returns the `{species, species_prob, genus, ...}` dicts `PredictionService`
  consumes). Back them with `onnxruntime.InferenceSession`.
- Swap them in `serving/main.py` `lifespan` behind an env switch (e.g. `MODEL_BACKEND=onnx`).
  **`PredictionService` does not change** — it depends only on the interface.

### B.2 Slim serving deps (no torch/ultralytics/ray)
Serving needs only: `fastapi`, `uvicorn`, `onnxruntime`, `pillow`, `numpy`,
`opencv-python-headless`, `sqlalchemy`, `psycopg2-binary` (or `asyncpg`), `google-auth`,
`google-cloud-storage`, `python-dotenv`, `pyyaml`. Dropping torch/ultralytics/ray is the
cold-start + image-size + cost win. Keep a separate `requirements-serving.txt` (or a `[serving]`
extra) so the prod image never pulls training deps.

### B.3 Dockerfile
- `python:3.12-slim`, install serving deps, copy app + the two `.onnx` files, entrypoint
  `uvicorn whatsthatfish.serving.main:app --host 0.0.0.0 --port $PORT` (**Cloud Run injects
  `$PORT`**). `.dockerignore` out data/, runs/, tests/, frontend/.

### B.4 Model artifacts — bundle vs GCS-pull
- **v1: bundle the ONNX files in the image** (immutable, reproducible, zero startup fetch).
  Move to GCS-pull-at-startup later if you want to hot-swap models without rebuilding.

### B.5 Cloud SQL (the app is DB-backed)
- Users/dives/observations live in Postgres. Provision **Cloud SQL (Postgres)**; connect from
  Cloud Run via the built-in connector (unix socket / Cloud SQL Auth Proxy). `DATABASE_URL`
  points at it.
- **Run `alembic upgrade head`** against Cloud SQL on deploy (a one-off Cloud Run Job or a
  pre-deploy step — not in the request path).
- Connection pooling: scale-to-zero + Cloud SQL connection caps → use a **small pool with
  `pool_pre_ping=True`** (or `NullPool`) so idle instances don't exhaust connections.

### B.6 Secrets
- `GOOGLE_OAUTH_CLIENT_ID`, DB password / `DATABASE_URL` → **Secret Manager**, mounted as env.
  Drop the scattered `load_dotenv()` calls for prod (the deferred de-scatter) — read straight env.

### B.7 Service account + IAM (signed-URL gotcha)
Cloud Run SA needs: **Cloud SQL Client**, **Storage Object Admin** (contributions bucket), and —
critically — **`iam.serviceAccounts.signBlob`**. `GCSContribution.retrieve_image` mints a V4
signed URL; on Cloud Run there's **no key file**, so `generate_signed_url` must use the
**IAM SignBlob** credentials path (service-account email + `signBlob`), not a local key. This
trips everyone once.

### B.8 CORS + B.9 Cloud Run config
- Add the production SPA origin to `serving/main.py` `allow_origins`.
- Cloud Run: 1–2 vCPU, 512Mi–1Gi, `min-instances=0`, sensible `max-instances`, request timeout.
  `/health` is already the readiness probe. ~$1/mo at low volume.

---

## C. Frontend hosting

### C.1 Build
- `npm run build` → static `dist/`. Inject `VITE_GOOGLE_CLIENT_ID` + a new `VITE_API_BASE`
  (the Cloud Run URL) **at build time** (Vite inlines `VITE_*` at build, not runtime).

### C.2 Host
- **Cloudflare Pages** (free, GitHub-CI, fast) — matches the CLAUDE.md hosting note. Alternatives:
  Firebase Hosting, GCS + Cloud CDN.

### C.3 CORS strategy — pick one
- **(a) Cross-origin (simplest first):** Cloud Run CORS allows the Pages domain; **GCS bucket
  CORS** allows the SPA origin so `AuthedImage`'s fetch→blob of the signed URL works (this is the
  deferred AuthedImage CORS item). 
- **(b) Same-origin:** front both API and SPA behind one domain (Cloudflare proxy / path routing)
  → no CORS at all. More setup, cleaner long-term.
- Alternative to GCS CORS entirely: have the API **stream photo bytes** instead of redirecting to
  a signed URL (trades API bandwidth for zero bucket-CORS config).

### C.4 Google OAuth origins (silent-failure gotcha)
- Add the production SPA origin to **Authorized JavaScript origins** on the Google OAuth client.
  Miss this and GIS sign-in fails silently in prod (exactly the blank-button class of bug).

---

## D. CI/CD (GitHub Actions)

- **On push/PR:** Postgres **service container** (mirror `docker-compose.test.yml`) → `pytest`;
  frontend `npm ci && npm run typecheck && npm test`.
- **On `v*` tag:** build serving image → push to **Artifact Registry** → deploy Cloud Run →
  `alembic upgrade head`. Frontend → Cloudflare Pages (or its native GitHub integration).
- **Model release:** export ONNX as part of a release job, **gate on eval** (macro Top-1 on the
  geographic val split + detector Recall@0.5), then bake into the image / push to GCS.

---

## E. Recommended sequence

1. **ONNX export + parity test + INT8 latency/accuracy gate** — decides CPU viability. Do first.
2. **ONNX inferrers + slim requirements + local `docker compose` (app + test pg)** — prove the
   container end-to-end locally before any cloud.
3. **Cloud SQL + Secret Manager + first Cloud Run deploy** — migrate, smoke `/health` + `/species`.
4. **GCS contributions: signed-URL signBlob IAM + bucket CORS** (or switch to byte-streaming).
5. **Frontend build + Pages deploy + OAuth origins + CORS** — full sign-in → predict → save loop.
6. **CI/CD wiring.**
7. **Drift alerting** — per-inference logging already emits top-1 conf + entropy; add the Cloud
   Monitoring alert on the 7-day rolling confidence drop.

## F. Decision checklist (flag before building)
- [ ] INT8 vs FP32 (accuracy gate on geographic val).
- [ ] Bundle ONNX in image vs GCS-pull.
- [ ] Photo serving: signed-URL redirect (needs bucket CORS + signBlob) vs API byte-stream.
- [ ] CORS: cross-origin vs same-origin-behind-one-domain.
- [ ] numpy preprocessing parity confirmed against training (byte-match).
