# Next Steps

What remains to take WhatsThisFish from "trained models + working serving code" to
a self-maintaining production system. Grouped: modelling, CI/CD, scheduled
(CRON) jobs, and the production infra that gates them.

---

### Done
- **Final classifier train** — 50-epoch, 3-phase run complete (`best.pt` ep. 49);
  species/genus targets met on the geographic split (see [MODELMETRICS](./MODELMETRICS.md)).
- **conf=0.15 eval** — detector **Recall@0.5 ≥ 0.90** confirmed in survey mode
  (LC1 hits 0.905 at default conf; the lower threshold only raises it).

### Remaining
| Item | Detail |
|---|---|
| **Family accuracy** | Family Top-1 is **below target** on the geographic split (see [MODELMETRICS](./MODELMETRICS.md)) — try family-aware sampling / longer family-only phase / add `freeze_epochs` to `cls_model_param_space_config.yaml` and sweep it. The one open modelling gap. |
| **LC2 detector** (deferred) | Optional `uiqm × conf` fine-tune `lc1_best.pt` → `lc2_best.pt`. **Not a blocker** — LC1 already clears every detector target; LC2 is upside only. |
| **Classification-mode bbox proposals** | Currently run with LC1; re-run if/when an LC2 model supersedes it, to refresh `inat_classification_dataset` crop annotations. |

---

## 2. CI/CD (GitHub Actions)

### Done
- **`ci_tests.yml`** — pytest against a Postgres service container on push/PR.
- **`release.yml`** — on `v*` tag: pull weights from GCS → export ONNX →
  **weights-only gate** (`release and not accuracy`: parity + INT8 agreement +
  degradation) → Alembic migrate Cloud SQL → push image to Artifact Registry.

### Remaining
| Item | Detail |
|---|---|
| **Frontend CI** | Add `npm ci && npm run typecheck && npm test` (Vitest) to the push/PR workflow — currently backend-only. |
| **Frontend deploy** | Cloudflare Pages. Native Git integration (Cloudflare builds on push to prod branch) is the default; **no workflow file** if using that path. Remember `VITE_*` vars are inlined **at build time** — a Cloud Run URL change needs a frontend rebuild. |
| **Cloud Run deploy on tag** | Currently manual (the `deploy-cloud-run` step in `release.yml` is intentionally commented out for MVP). Un-comment / gate behind a manual approval when ready for auto-deploy. |
| **Accuracy-tier gate** | The `release and accuracy` tier (detector Recall/mAP + macro Top-1 on the **full geographic val split**) needs the DB + image corpus, so it runs in a **data-bearing env** (Cloud SQL job / post-train on the training VM), **not** in the tag workflow. Wire this as the promotion gate for new weights. |

---

## 3. Scheduled (CRON) jobs

The retraining loop is **event-gated, not time-forced**: a cheap weekly check
decides whether the expensive training VM ever spins up. Cloud Scheduler is the
clock; Cloud Run Jobs and a preemptible VM do the work.

### 3a. Data ingestion — *delta ingest* (weekly)
- **Trigger:** Cloud Scheduler → Cloud Run **Job**.
- **Does:** incremental iNat ingest (new research-grade photos for in-scope taxa
  since the last watermark), reusing the WAL + Postgres crash-recovery path so a
  preempted run resumes cleanly.
- **Emits:** new-row counts per taxon → feeds the gate below.

### 3b. Gating model training — *threshold check* (same job, after ingest)
- **Decision:** spin up the training VM **only if** the dataset grew **> 5%**
  **or** a new taxon crossed the minimum-observation bar.
- **Why:** training is the cost centre; most weeks the answer is "no", and the
  job exits having spent pennies.
- **Action on yes:** call the Compute Engine API to launch the training VM (3c).

### 3c. Model retraining — *preemptible training VM* (on demand)
- **Compute:** preemptible **L4** (~$0.35/hr), launched by a startup script (not
  Vertex Pipelines).
- **Chain:** ETL delta → detector curriculum (Ultralytics `--resume`) →
  classifier train → **eval gate** (macro Top-1 on geographic val + detector
  Recall@0.5) → **artifact push to GCS** *only if the gate passes* →
  **self-terminate**.
- **Resumability:** Ultralytics `--resume` + WAL so a preemption mid-run doesn't
  restart from zero.

### 3d. Performance logging — *drift alert* (continuous + rolling)
- **Already emitting:** `PredictionService._log_inference` writes one structured
  line per inference (top-1 species, top-1 confidence, output entropy,
  `detected`) → stdout → **Cloud Logging** on Cloud Run.
- **Remaining:** a **Cloud Monitoring** alert on a **7-day rolling drop** in mean
  confidence (or entropy rise). The alert notifies a human — **retraining stays a
  human decision**, it does not auto-fire the VM.

```
Cloud Scheduler (weekly)
        │
        ▼
Cloud Run Job:  delta ingest ──▶ threshold check ──┐  (>5% growth or new taxon?)
                                                    │ yes
                                                    ▼
                              Preemptible L4 VM: ETL → train → eval gate → GCS push → self-terminate
                                                    │ (weights promoted only if gate green)
                                                    ▼
                              Release workflow (v* tag) bakes ONNX into the serving image

Cloud Run (serving) ──▶ per-inference log ──▶ Cloud Logging ──▶ Cloud Monitoring 7-day drift alert ──▶ human
```

---

## 4. Production infra

Most of the stack is **provisioned and live**. One item remains.

### Remaining
| Item | Note |
|---|---|
| **Secret Manager** | Move `GOOGLE_OAUTH_CLIENT_ID`, `DATABASE_URL` (and any other secrets) into Secret Manager, mounted as env on Cloud Run; drop the scattered `load_dotenv()` for prod. The last infra gap. |

### Done (provisioned)
| Item | Note |
|---|---|
| **Cloud Run deploy** | Live: `min-instances=0`, `/health` readiness probe, scale-to-zero. |
| **Cloud SQL (Postgres)** | Connected via the built-in connector; small pool + `pool_pre_ping=True`. |
| **Service-account IAM** | Cloud Run SA has **Cloud SQL Client**, **Storage Object Admin**, and **`iam.serviceAccounts.signBlob`** (V4 signed URLs via the IAM SignBlob path — no key file). |
| **GCS bucket CORS** | SPA origin allowed so `AuthedImage`'s fetch→blob of the signed URL works. |
| **Prod CORS origin** | `ALLOWED_ORIGINS` set to the Cloudflare domain; localhost excluded in prod (gated on `K_SERVICE`). |
| **Google OAuth origins** | Production SPA origin added to **Authorized JavaScript origins** (GIS sign-in works in prod). |

---

## 5. Security follow-ups (from the security review)

- **Documented tradeoff:** the Google ID token is stored in `localStorage`
  (XSS-stealable, bounded by ~1h TTL, no refresh). Acceptable given zero
  `dangerouslySetInnerHTML` in the SPA — keep it that way. Revisit only if moving
  to Firebase session cookies.
- **Verify** `torch.load(weights_only=True)` against an existing `classifier_best.pt`
  on the next `predict()` run (checkpoint is tensors + plain metadata, so it
  should load; a legacy checkpoint with exotic globals would need
  `add_safe_globals`).

## Estimated cost

~**$15/month** (GCS storage dominates). Serving ~$1/mo; training only when the
weekly gate fires.
