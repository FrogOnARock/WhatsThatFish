# Phase 3 Roadmap — User Experience & Observation Tracking

Phase 3 turns the working prediction demo into a multi-user app: people sign in,
record what they saw (species + where + how deep), and review their history. The
detector→classifier pipeline and the serving layer already work end to end; this
phase is about **persistence, identity, and the observation lifecycle**.

Scope decided 2026-06-22. Serving/containerization and the extended-taxa
correction picker are **explicitly deferred** to a later phase (see end).

---

## 1. Authentication — Google only, GCP Identity Platform (managed)

Discord dropped (not worth the custom-OIDC hassle for this audience). Google-only,
via **GCP Identity Platform / Firebase Auth**:

- **Frontend**: Firebase JS SDK runs the Google sign-in popup/redirect, yields a
  Google **ID token**.
- **Backend**: a `firebase-admin` dependency verifies the ID token on each request
  (`get_current_user`), then **upserts into our own `users` table** keyed by the
  OIDC `sub` (`google_subject_id`). We still own the user row — Identity Platform
  is only the token issuer/verifier, not the source of truth for app data.
- **Transport**: send the verified ID token as a `Bearer` header from the SPA.
  (Identity Platform handles refresh client-side, so we don't manage sessions.)
- **CORS**: explicit SPA origin(s); add the deployed frontend origin when it exists.

Why managed over self-hosted here: with Discord gone, Google-only is a few lines of
`firebase-admin` verification vs. owning the full OAuth dance, secret rotation, and
CSRF. We're already all-in on GCP, so the coupling is low-risk.

**Secrets**: Firebase project config (frontend, public) + service-account creds
(backend) via env / Secret Manager on Cloud Run.

---

## 2. Data model

Mirrors iNaturalist's normalization — **photos are evidence attached to an
observation**, the observation carries the taxon — with a **dive** layered on top as
the place/time grouping. Normalize the schema now (cheap at design time, painful to
migrate later); the **initial UI can collapse it** to one-dive / one-observation /
one-photo per save, then expand to multi-photo and multi-observation later without a
migration.

```
users
  id uuid pk · google_subject_id · email · display_name
  avatar_url · created_at · last_login_at        UNIQUE(google_subject_id)

dive_sites                       -- deduplicated named locations
  id uuid pk · name (proper-cased) · name_key (normalized, for dedup)
  lat (nullable) · lng (nullable) · created_by_user_id fk→users · created_at
  UNIQUE(name_key)

dives                            -- one place + time; groups many observations
  id uuid pk · user_id fk→users · site_id fk→dive_sites (nullable)
  gps_lat (nullable) · gps_lng (nullable) · dived_at · notes · created_at
  INDEX(user_id, dived_at)

observations                     -- one taxon encounter within a dive
  id uuid pk · dive_id fk→dives · user_id fk→users   -- user_id denormalized for scoping
  predicted_taxon_id  fk→inat_taxa.taxon_id (nullable)
  corrected_taxon_id  fk→inat_taxa.taxon_id (nullable)
  confidence · depth_m (nullable, manual) · observed_at · created_at
  INDEX(user_id)

observation_photos               -- evidence; each photo independently classified
  id uuid pk · observation_id fk→observations
  image_path (contributions/{user_id}/{uuid}.jpg)
  bbox jsonb · predicted_taxon_id fk→inat_taxa.taxon_id · confidence · width · height · created_at
```

- **Taxon refs are the stable iNat `taxon_id`, NOT the model's zero-index.** The
  zero-index re-numbers on every retrain and would corrupt saved history; `taxon_id`
  is the domain's permanent identity. They FK the **full `inat_taxa`** (~44K, a
  superset of the trained `app_taxa`), so corrections can range over the whole
  in-scope taxonomy — the seam for the deferred taxa picker, **no schema change**
  needed. At save time the serving layer **translates the classifier's zero-index →
  `taxon_id` via `app_taxa`** before persisting.
- **Location & time live on the `dive`** (one site/coords per dive); **depth lives on
  the `observation`** (it varies by what you saw and where in the water column).
- **Per-photo model output** (`predicted_taxon_id`, `confidence`, `bbox`) lives on
  `observation_photos` — each photo is independent evidence. The **observation's**
  taxon is the user's chosen ID (`corrected_taxon_id` overrides `predicted_taxon_id`),
  aggregated across its photos.
- **Every observation belongs to a dive** (`dive_id` NOT NULL): a single quick ID
  creates a lightweight one-observation dive on save. This keeps location normalized
  in one place rather than duplicated onto standalone observations.
- New SQLAlchemy models in `database/models.py` + an Alembic migration.
  **(DONE — migration `b1a2cea8dd23`, applied.)**
- `corrected_taxon_id` is the seam for the deferred taxa-correction picker — null
  until the user overrides the prediction; FK to the full `inat_taxa` already
  allows any in-scope taxon.

---

## 3. Observation submission → GCS, and retrieval for history

Reuses the existing storage abstraction and the `contributions/` bucket prefix.

- **Write** ("Save to history"): authed `POST /observations` (multipart: image +
  prediction fields + dive/site + depth + optional GPS) →
  - resolve-or-create the **dive** (site + dived_at + optional GPS) and the
    **observation** (taxon + depth),
  - upload the photo bytes to `contributions/{user_id}/{uuid}.jpg` via a **new
    single-blob `ContributionStorage.upload(bytes, key)`** (extend
    `etl/gcs_client.py`, which today only has the bulk `gcs_upload(dir, ...)`),
  - insert the `observation_photos` row (image_path + bbox + per-photo prediction).
  - The MVP collapses this to one dive / one observation / one photo per call;
    adding a photo to an existing observation, or an observation to an existing
    dive, is the same endpoint with an existing parent id.
- **Read** (history): authed `GET /dives` / `GET /observations` → filtered by
  `user_id`; photos served via the **existing signed-URL pattern** (`GCSImage` →
  `RedirectResponse`), with an **ownership check** so user A can't fetch B's keys.

---

## 4. Location tagging — named sites with dedup, optional GPS, manual depth

- **Depth**: manual numeric input (`depth_m`, nullable). Done conceptually — the
  `ResultsView` "Where did you see it?" panel already stubs the inputs.
- **Sites**: a `dive_sites` table of **proper-cased named locations**. On entry, a
  **filter-search endpoint** (`GET /sites?q=…`, `ILIKE`/`pg_trgm` over `name_key`)
  surfaces similar existing sites so users pick rather than duplicate. They *can*
  add a new site (proper-case the name, derive `name_key`, insert).
- **GPS**: optional/nullable per-observation `gps_lat`/`gps_lng`. The browser
  Geolocation API only returns the **device's current position** (no search), and
  divers usually log off-site — so GPS is a one-tap convenience prefill, never the
  primary record. The named site is the real location signal.
- **Map pin / geocoding**: deferred — named-site search covers the minimum UX.

---

## 5. Test coverage review (Phase 3 surfaces)

Serving already has `test_serving_api.py`, `test_inference_pipeline.py`,
`test_integration.py`. New/under-covered areas to add:

- **Auth**: `get_current_user` with valid / expired / missing token (dependency
  override + a fake verifier).
- **Ownership scoping**: user A cannot read/fetch user B's dives, observations, or
  photos (the denormalized `user_id` + the ownership check on signed URLs).
- **Dive/observation/photo write** (resolve-or-create dive + observation, attach
  photo) + **site dedup** (search matches existing; new-site insert
  normalizes/proper-cases).
- **GCS write + signed-URL** read, mocked.
- **Recent serving logic** not yet covered: the `detected=false` whole-frame
  fallback, the `None`-box crop-skip in `ClassInference`, and the bbox pixel→xywh%
  conversion in `_create_bbox`.

Harness: FastAPI `TestClient` + dependency overrides + mock GCS + a fixture user.

---

## Sequencing

1. Data model (users, dive_sites, dives, observations, observation_photos) + Alembic migration.
2. Google auth (Identity Platform verify + `users` upsert + `get_current_user`).
3. Observation submission (dive/observation/photo write + GCS upload) + history
   retrieval (signed URLs, ownership).
4. Location tagging UX (site search/dedup + picker + optional GPS + manual depth).
5. Phase 3 test coverage pass.
6. (Carried) Finish serving error handling + inference logging — Task #1.

---

## Deferred to a later phase

- **Extended taxa correction picker** — let users correct a prediction to any taxon,
  not just the ~1,500 trained species. Blocked on sourcing **human-searchable
  aliases/common names** for the full ~44K taxa (iNat vernacular export / GBIF).
  The `corrected_taxon_id` FK (→ full `inat_taxa`) is the forward hook — the schema
  already supports any in-scope taxon, so this is purely a UI + alias-data task.
- **Containerization + ONNX export** — Dockerize for Cloud Run; export YOLO +
  CustomResnet to ONNX (INT8) and serve on `onnxruntime` CPU. Note: preprocessing
  is nearly torch-free already (`AddMultiChannel` uses cv2/numpy and only touches
  torch to assemble the final 5-ch tensor) — porting that one step to numpy lets the
  runtime drop torch entirely. Needs INT8 accuracy validation vs. the eval targets.
