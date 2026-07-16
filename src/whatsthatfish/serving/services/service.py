import math
import uuid

from PIL import UnidentifiedImageError

from whatsthatfish.config import _get_logger
from whatsthatfish.serving.db.repository import (
    TaxaRepository,
    UserRepository,
    ObservationRepository,
)
from whatsthatfish.serving.error import (
    InvalidPredictionRequest,
    InvalidPredictionResponse,
    ResourceNotFoundException,
    ValidationException,
)
from whatsthatfish.serving.schemas import (
    SpeciesEntry,
    RegionOut,
    Prediction,
    Bbox,
    Candidate,
    DiveCreate,
    DiveUpdate,
    DiveOut,
    ObservationCreate,
    ObservationUpdate,
    ObservationOut,
    PhotoOut,
    HistorySighting,
    HistorySpecies,
    FieldLog,
    TaxonOption,
    ModelStats,
    DiveSpecies,
    SiteOption,
    UserProfile,
    UserSettingsUpdate,
    UserStats,
)

# Dedicated logger: one structured line per inference for drift monitoring
# (top-1 confidence + entropy). On Cloud Run, stdout → Cloud Logging.
logger = _get_logger("whatsthatfish.serving.inference")

# Extended dive-log columns shared 1:1 between the DiveCreate/Update/Out schemas
# and the Dive model, so create/update/serialize can iterate rather than repeat.
_DIVE_LOG_FIELDS = (
    "visibility_m",
    "air_temp_c",
    "water_temp_c",
    "weight_kg",
    "exposure_suit",
    "depth_avg_m",
    "depth_max_m",
    "started_at",
    "bottom_time_min",
    "total_time_min",
    "end_pressure_bar",
    "dive_shop",
)


class UserService:
    def __init__(self, session):
        self.repo = UserRepository(session)

    def get_or_create(self, claims: dict):
        """Resolve the local User for a verified set of Google token claims,
        creating the row on first sign-in. Returns the persisted User."""
        return self.repo.upsert_from_claims(claims)

    def profile(self, user) -> UserProfile:
        """Serialise the user for the SPA (Google bits + app-owned fields)."""
        return UserProfile(
            id=str(user.id),
            email=user.email,
            display_name=user.display_name,
            avatar_url=user.avatar_url,
            preferred_name=user.preferred_name,
            unit_system=user.unit_system,
        )

    def update_settings(self, user, data: UserSettingsUpdate) -> UserProfile:
        """PATCH the user's app-owned profile. These columns are untouched by the
        Google claims sync, so the edit survives the next login."""
        changes = data.model_dump(exclude_unset=True)
        if "preferred_name" in changes:
            # Empty string clears the override → fall back to the Google name.
            name = (changes["preferred_name"] or "").strip()
            user.preferred_name = name or None
        if changes.get("unit_system") is not None:
            user.unit_system = changes["unit_system"]
        self.repo.save()
        return self.profile(user)


class TaxaService:
    def __init__(self, session):
        self.repo = TaxaRepository(session)

    def search_species(self, q: str, limit: int = 25) -> list[TaxonOption]:
        """Correction picker: species-rank fish/shark taxa matching `q`."""
        return [
            TaxonOption(taxon_id=r.taxon_id, name=r.name, common_name=r.common_name)
            for r in self.repo.search_species(q, limit)
        ]

    def get_stats(self) -> ModelStats:
        return ModelStats(**self.repo.get_counts())

    def get_species(self):
        # A trained species can exist in app_taxa BEFORE the LLM enrichment pass
        # fills common_name/description/location/depth/filename (all nullable).
        # Contract: keep the species VISIBLE in the catalogue with blank enrichment
        # (coerce NULL → ""/[]/0) rather than 500-ing on Pydantic validation or
        # hiding a class the model genuinely knows. The SPA's SpeciesEntry stays
        # non-null, so nothing downstream changes.
        rows = self.repo.query_species()

        # Structured ranges from species_regions (one query for the whole
        # catalogue); the flat `location` blurb stays until the UI cuts over.
        regions_by_taxon = self.repo.regions_for_taxa([r.taxon_id for r in rows])

        return [
            SpeciesEntry(
                species_id=row.zero_indexed_species,
                name=row.species,
                genus=row.genus,
                family=row.family,
                image_count=row.img_count or 0,
                description=row.description or "",
                filename=row.filename or "",
                common_name=row.common_name or "",
                location=row.location or [],
                regions=[
                    RegionOut(
                        id=r.id, name=r.name, kind=r.kind, parent_id=r.parent_id
                    )
                    for r in regions_by_taxon.get(row.taxon_id, [])
                ],
                depth=row.depth or "",
            )
            for row in rows
        ]


class PredictionService:
    def __init__(self, session, bbox_inferrer, class_inferrer):
        self.repo = TaxaRepository(session)
        self.bbox_inferrer = bbox_inferrer
        self.class_inferrer = class_inferrer

    def _create_candidate(self, candidate) -> Candidate:
        return Candidate(
            name=candidate.get("name", None),
            index=candidate.get("zero_index", None),
            conf=candidate.get("conf", None),
            summary=candidate.get("summary", None),
            common=candidate.get("common", None),
            habitat=candidate.get("habitat", None),
        )

    def _create_bbox(self, box) -> Bbox:
        # Detector gives absolute pixels (x1,y1,x2,y2) plus the original image
        # w/h. Convert to percent-of-image xywh so the frontend can position the
        # overlay on the CSS-scaled <img> without knowing its rendered size.
        w, h = box["w"], box["h"]
        return Bbox(
            x=box["x1"] / w * 100,
            y=box["y1"] / h * 100,
            w=(box["x2"] - box["x1"]) / w * 100,
            h=(box["y2"] - box["y1"]) / h * 100,
        )

    def query_species_candidate(
        self, candidates: list[int], conf: list[float]
    ) -> list[Candidate]:

        rows = self.repo.query_species_candidate(candidates)

        candidates = [
            {
                "name": row.species,
                "zero_index": row.zero_indexed_species,
                "summary": row.description,
                "common": row.common_name,
                "habitat": row.location,
            }
            for row in rows
        ]

        for i in range(len(candidates)):
            candidates[i]["conf"] = conf[i]

        return [self._create_candidate(candidate) for candidate in candidates]

    def query_genus_candidate(
        self, candidates: list[int], conf: list[float]
    ) -> list[Candidate]:

        rows = self.repo.query_genus_candidate(candidates)

        candidates = [
            {
                "name": row.genus,
                "zero_index": row.zero_indexed_genus,
            }
            for row in rows
        ]

        for i in range(len(candidates)):
            candidates[i]["conf"] = conf[i]

        return [self._create_candidate(candidate) for candidate in candidates]

    def query_family_candidate(
        self, candidates: list[int], conf: list[float]
    ) -> list[Candidate]:

        rows = self.repo.query_family_candidate(candidates)

        candidates = [
            {
                "name": row.family,
                "zero_index": row.zero_indexed_family,
            }
            for row in rows
        ]

        for i in range(len(candidates)):
            candidates[i]["conf"] = conf[i]

        return [self._create_candidate(candidate) for candidate in candidates]

    def get_candidates(self, result: dict):
        """Return the species catalogue: one entry per distinct trained species."""

        species_candidates = self.query_species_candidate(
            result["species"], result["species_prob"]
        )
        genus_candidates = self.query_genus_candidate(
            result["genus"], result["genus_prob"]
        )
        family_candidates = self.query_family_candidate(
            result["family"], result["family_prob"]
        )

        # The DB queries return rows ordered by zero-index (so conf could be
        # paired correctly); re-sort by confidence descending so the frontend's
        # index 0 is the top guess for each head.
        by_conf = lambda cands: sorted(cands, key=lambda c: c.conf, reverse=True)

        return (
            by_conf(species_candidates),
            by_conf(genus_candidates),
            by_conf(family_candidates),
        )

    def sort(self, object1, object2, key_sort: int = 0):
        return zip(*sorted(zip(object1, object2), key=lambda x: x[key_sort]))

    def get_prediction(self, img_batch: bytes | list[bytes]):

        if not img_batch:
            raise InvalidPredictionRequest(
                message="Empty request: no image bytes provided",
                body={"reason": "empty_body"},
            )

        # Stage 1 — detector. A decode error means the bytes aren't an image:
        # the client's fault → 422 + body. Anything else is a server-side
        # inference failure → 500, logged with a traceback, generic message out.
        try:
            bbox_results = self.bbox_inferrer.infer(img_batch)
        except UnidentifiedImageError as exc:
            raise InvalidPredictionRequest(
                message="Uploaded file is not a readable image",
                body={"reason": "unreadable_image"},
            ) from exc
        except Exception as exc:
            logger.exception("Detector inference failed")
            raise InvalidPredictionResponse(
                "Detection failed during inference"
            ) from exc

        # Stage 2 — classifier. The image already decoded in stage 1, so any
        # failure here is server-side.
        try:
            class_results = self.class_inferrer.infer(img_batch, bbox_results)
        except Exception as exc:
            logger.exception("Classifier inference failed")
            raise InvalidPredictionResponse(
                "Classification failed during inference"
            ) from exc

        species_list = []
        genus_list = []
        family_list = []
        for result in class_results:
            species, species_prob = self.sort(result["species"], result["species_prob"])
            genus, genus_prob = self.sort(result["genus"], result["genus_prob"])
            family, family_prob = self.sort(result["family"], result["family_prob"])

            result["species"] = species
            result["species_prob"] = species_prob

            result["genus"] = genus
            result["genus_prob"] = genus_prob

            result["family"] = family
            result["family_prob"] = family_prob

            species_can, genus_can, family_can = self.get_candidates(result=result)
            species_list.extend(species_can)
            genus_list.extend(genus_can)
            family_list.extend(family_can)

        # Skip None boxes (no detection): the species was still classified from
        # the full frame, but there's no box to overlay. `detected` lets the UI
        # signal "no fish detected — whole-frame guess, low confidence".
        bbox = [self._create_bbox(box) for box in bbox_results if box is not None]
        detected = len(bbox) > 0

        self._log_inference(species_list, detected)

        return Prediction(
            bbox=bbox,
            species=species_list,
            genus=genus_list,
            family=family_list,
            detected=detected,
        )

    def _log_inference(
        self, species_candidates: list[Candidate], detected: bool
    ) -> None:
        """Emit one structured line per inference for drift monitoring:
        top-1 species + confidence + entropy over the returned distribution.
        A sustained drop in confidence / rise in entropy is the retrain signal
        (Cloud Monitoring alert, per the roadmap). Never raises — logging must
        not break a successful prediction."""
        try:
            if not species_candidates:
                logger.warning(
                    "inference produced no species candidates (detected=%s)", detected
                )
                return
            top = max(species_candidates, key=lambda c: c.conf)
            probs = [c.conf for c in species_candidates if c.conf]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            logger.info(
                "inference top1=%s top1_conf=%.4f entropy=%.4f n_candidates=%d detected=%s",
                top.name,
                top.conf,
                entropy,
                len(probs),
                detected,
            )
        except Exception:
            logger.exception("inference logging failed (non-fatal)")


class ObservationService:
    """Write/read the dive → observation → photo tree for a signed-in user.

    Translates the classifier's zero-index ids to stable iNat taxon_ids at save
    time, owns the corrected/effective-label rule, and uploads photos through the
    environment-appropriate ContributionStorage. Every method is passed the
    authenticated `user` and scopes by `user.id`."""

    def __init__(self, session, storage):
        self.repo = ObservationRepository(session)
        self.taxa = TaxaRepository(session)
        self.storage = storage

    # ── dives ────────────────────────────────────────────────────────────────
    def create_dive(self, user, data: DiveCreate) -> DiveOut:
        site = (
            self.repo.resolve_or_create_site(
                data.site_name,
                user.id,
                google_place_id=data.google_place_id,
                lat=data.gps_lat,
                lng=data.gps_lng,
            )
            if data.site_name
            else None
        )
        dive = self.repo.create_dive(
            user.id,
            site_id=site.id if site else None,
            gps_lat=data.gps_lat,
            gps_lng=data.gps_lng,
            dived_at=data.dived_at,
            notes=data.notes,
            **{f: getattr(data, f) for f in _DIVE_LOG_FIELDS},
        )
        return self._dive_out(dive, site)

    def update_dive(self, user, dive_id, data: DiveUpdate) -> DiveOut:
        dive = self.repo.get_dive(user.id, dive_id)
        if dive is None:
            raise ResourceNotFoundException("Dive not found")
        site = dive.site
        if data.site_name is not None:
            site = self.repo.resolve_or_create_site(
                data.site_name,
                user.id,
                google_place_id=data.google_place_id,
                lat=data.gps_lat,
                lng=data.gps_lng,
            )
            dive.site_id = site.id
        for field in ("gps_lat", "gps_lng", "dived_at", "notes", *_DIVE_LOG_FIELDS):
            value = getattr(data, field)
            if value is not None:
                setattr(dive, field, value)
        self.repo.save()
        return self._dive_out(dive, site)

    def list_dives(self, user) -> list[DiveOut]:
        dives = self.repo.list_dives(user.id)
        # One taxa_display lookup covers every species across every dive, so the
        # per-dive species summaries don't each fire their own query.
        taxon_ids = {o.corrected_taxon_id for d in dives for o in d.observations}
        display = self.taxa.taxa_display(list(taxon_ids))
        return [self._dive_out(d, d.site, display) for d in dives]

    def user_stats(self, user) -> UserStats:
        """Summary counts for the Settings page."""
        return UserStats(**self.repo.user_stats(user.id))

    def search_sites(self, q: str, limit: int = 10) -> list[SiteOption]:
        """Autocomplete existing dive sites (substring on the normalized key) so
        users reuse a site instead of creating a near-duplicate."""
        return [
            SiteOption(id=s.id, name=s.name) for s in self.repo.search_sites(q, limit)
        ]

    # ── observations ─────────────────────────────────────────────────────────
    def create_observation(self, user, data: ObservationCreate) -> ObservationOut:
        if self.repo.get_dive(user.id, data.dive_id) is None:
            raise ResourceNotFoundException("Dive not found")

        # Translate zero-index → taxon_id (one query for both predicted + corrected).
        indices = [data.predicted_species_index]
        if data.corrected_species_index is not None:
            indices.append(data.corrected_species_index)
        taxon_of = self.taxa.species_index_to_taxon(indices)

        predicted_taxon_id = taxon_of.get(data.predicted_species_index)
        if predicted_taxon_id is None:
            raise ValidationException(
                f"Unknown species index {data.predicted_species_index}"
            )

        # Effective-label precedence: explicit taxon_id (full-list report pick) >
        # candidate index (user selected one of the model's guesses) > prediction.
        if data.corrected_taxon_id is not None:
            corrected_taxon_id = data.corrected_taxon_id
        elif data.corrected_species_index is not None:
            corrected_taxon_id = taxon_of.get(data.corrected_species_index)
            if corrected_taxon_id is None:
                raise ValidationException(
                    f"Unknown corrected species index {data.corrected_species_index}"
                )
        else:
            corrected_taxon_id = predicted_taxon_id

        obs = self.repo.create_observation(
            user.id,
            dive_id=data.dive_id,
            predicted_taxon_id=predicted_taxon_id,
            corrected_taxon_id=corrected_taxon_id,
            label_status=data.label_status,
            confidence=data.confidence,
            depth_m=data.depth_m,
            observed_at=data.observed_at,
        )
        return self._obs_out(obs)

    def update_observation(
        self, user, observation_id, data: ObservationUpdate
    ) -> ObservationOut:
        """Edit a sighting's effective label / status / depth. Site is NOT here —
        it lives on the dive (PATCH /dives), since it's shared across the dive."""
        obs = self.repo.get_observation(user.id, observation_id)
        if obs is None:
            raise ResourceNotFoundException("Observation not found")
        changes = data.model_dump(exclude_unset=True)
        if changes.get("corrected_taxon_id") is not None:
            obs.corrected_taxon_id = changes["corrected_taxon_id"]
        if changes.get("label_status") is not None:
            obs.label_status = changes["label_status"]
        if "depth_m" in changes:  # allow clearing to null
            obs.depth_m = changes["depth_m"]
        self.repo.save()
        return self._obs_out(obs)

    # ── photos ───────────────────────────────────────────────────────────────
    def add_photo(
        self,
        user,
        observation_id,
        image_bytes: bytes,
        bbox: dict | None = None,
        predicted_species_index: int | None = None,
        confidence: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> PhotoOut:
        if self.repo.get_observation(user.id, observation_id) is None:
            raise ResourceNotFoundException("Observation not found")

        # Backend-agnostic key; storage prefixes it (contributions/ or test-history/).
        key = f"{user.id}/{uuid.uuid4().hex}.jpg"
        self.storage.upload(key, image_bytes)

        predicted_taxon_id = None
        if predicted_species_index is not None:
            predicted_taxon_id = self.taxa.species_index_to_taxon(
                [predicted_species_index]
            ).get(predicted_species_index)

        photo = self.repo.create_photo(
            observation_id=observation_id,
            image_path=key,
            bbox=bbox,
            predicted_taxon_id=predicted_taxon_id,
            confidence=confidence,
            width=width,
            height=height,
        )
        return self._photo_out(photo)

    def get_photo_image(self, user, photo_id):
        """Serve a contribution photo back (FileResponse local / signed-URL
        redirect in cloud), scoped to the owner."""
        photo = self.repo.get_photo(user.id, photo_id)
        if photo is None:
            raise ResourceNotFoundException("Photo not found")
        return self.storage.retrieve_image(photo.image_path)

    # ── deletion (blobs first, then the cascade row delete) ──────────────────
    def _delete_blobs(self, keys: list[str]) -> None:
        """Best-effort blob cleanup. A single failed delete must not abort the
        whole request (the row delete is what the user asked for); the orphaned
        blob is logged for out-of-band cleanup rather than surfaced as a 500."""
        for key in keys:
            try:
                self.storage.delete(key)
            except Exception:
                logger.exception("failed to delete contribution blob %s", key)

    def delete_photo(self, user, photo_id) -> None:
        """Remove one photo (its blob + row). Ownership-scoped."""
        photo = self.repo.get_photo(user.id, photo_id)
        if photo is None:
            raise ResourceNotFoundException("Photo not found")
        self._delete_blobs([photo.image_path])
        self.repo.delete(photo)

    def delete_observation(self, user, observation_id) -> None:
        """Remove a whole sighting: every photo blob, then the observation row
        (photo rows go via delete-orphan cascade)."""
        obs = self.repo.get_observation(user.id, observation_id)
        if obs is None:
            raise ResourceNotFoundException("Observation not found")
        self._delete_blobs([p.image_path for p in obs.photos])
        self.repo.delete(obs)

    def delete_dive(self, user, dive_id) -> None:
        """Remove a dive and everything under it: all photo blobs across all its
        observations, then the dive row (observations + photos cascade)."""
        dive = self.repo.get_dive(user.id, dive_id)
        if dive is None:
            raise ResourceNotFoundException("Dive not found")
        self._delete_blobs(self.repo.dive_image_paths(user.id, dive_id))
        self.repo.delete(dive)

    def set_hero_photo(self, user, photo_id) -> None:
        """Mark a photo as the card image for its effective species (clears any
        prior hero for that species). Ownership-scoped."""
        if not self.repo.set_hero(user.id, photo_id):
            raise ResourceNotFoundException("Photo not found")

    # ── history (field log) ──────────────────────────────────────────────────
    def get_field_log(self, user) -> FieldLog:
        """Group all of the user's observations by the EFFECTIVE taxon
        (corrected_taxon_id) into field-log cards, each with its sightings +
        photos."""
        observations = self.repo.list_user_observations(user.id)

        groups: dict[int, list] = {}
        for obs in observations:
            groups.setdefault(obs.corrected_taxon_id, []).append(obs)

        display = self.taxa.taxa_display(list(groups.keys()))
        species = [
            HistorySpecies(
                taxon_id=taxon_id,
                species=getattr(display.get(taxon_id), "species", None),
                genus=getattr(display.get(taxon_id), "genus", None),
                family=getattr(display.get(taxon_id), "family", None),
                common_name=getattr(display.get(taxon_id), "common_name", None),
                sighting_count=len(obs_list),
                sightings=[self._sighting(o) for o in obs_list],
            )
            for taxon_id, obs_list in groups.items()
        ]
        return FieldLog(species=species, total_species=len(species))

    # ── mappers ──────────────────────────────────────────────────────────────
    def _dive_out(self, dive, site, display=None) -> DiveOut:
        # `display` is the shared taxa_display map from list_dives; for single-dive
        # callers (create/update) it's None, so resolve from this dive's own obs.
        observations = dive.observations
        if display is None:
            display = self.taxa.taxa_display(
                list({o.corrected_taxon_id for o in observations})
            )
        species, seen = [], set()
        for obs in observations:
            taxon_id = obs.corrected_taxon_id
            if taxon_id in seen:
                continue
            seen.add(taxon_id)
            d = display.get(taxon_id)
            species.append(
                DiveSpecies(
                    taxon_id=taxon_id,
                    name=getattr(d, "species", None),
                    common_name=getattr(d, "common_name", None),
                )
            )
        return DiveOut(
            id=dive.id,
            site_id=dive.site_id,
            site_name=site.name if site else None,
            gps_lat=dive.gps_lat,
            gps_lng=dive.gps_lng,
            dived_at=dive.dived_at,
            notes=dive.notes,
            verified=dive.verified,
            verified_source=dive.verified_source,
            created_at=dive.created_at,
            observation_count=len(observations),
            species=species,
            **{f: getattr(dive, f) for f in _DIVE_LOG_FIELDS},
        )

    def _obs_out(self, obs) -> ObservationOut:
        return ObservationOut(
            id=obs.id,
            dive_id=obs.dive_id,
            predicted_taxon_id=obs.predicted_taxon_id,
            corrected_taxon_id=obs.corrected_taxon_id,
            label_status=obs.label_status,
            confidence=obs.confidence,
            depth_m=obs.depth_m,
            observed_at=obs.observed_at,
        )

    def _photo_out(self, photo) -> PhotoOut:
        return PhotoOut(
            id=photo.id,
            observation_id=photo.observation_id,
            image_path=photo.image_path,
            bbox=photo.bbox,
            confidence=photo.confidence,
            width=photo.width,
            height=photo.height,
            is_hero=photo.is_hero,
        )

    def _sighting(self, obs) -> HistorySighting:
        site = obs.dive.site if obs.dive else None
        return HistorySighting(
            observation_id=obs.id,
            dive_id=obs.dive_id,
            dived_at=obs.dive.dived_at if obs.dive else None,
            site_name=site.name if site else None,
            depth_m=obs.depth_m,
            label_status=obs.label_status,
            photos=[self._photo_out(p) for p in obs.photos],
        )
