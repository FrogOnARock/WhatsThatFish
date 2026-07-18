from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
    DateTime,
    Text,
    ARRAY,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql.schema import UniqueConstraint
from .base import Base
from typing import Any
from datetime import datetime
import uuid


class InatTaxa(Base):
    __tablename__ = "inat_taxa"

    taxon_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ancestry: Mapped[str | None] = mapped_column(String)
    rank_level: Mapped[float | None] = mapped_column(Float)
    rank: Mapped[str | None] = mapped_column(String(50))
    name: Mapped[str] = mapped_column(String(255))
    common_name: Mapped[str | None] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    filtered_observations: Mapped[list["InatFilteredObservations"]] = relationship(
        back_populates="taxon"
    )

    __table_args__ = (
        Index("ix_inat_taxa_rank_level", "rank_level"),
        Index("ix_inat_taxa_ancestry_active", "ancestry", "active"),
    )


class InatFilteredObservations(Base):
    """Pre-filtered iNaturalist photo records for models training."""

    __tablename__ = "inat_filtered_observations"

    photo_uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    photo_id: Mapped[int] = mapped_column(BigInteger)
    observation_uuid: Mapped[str] = mapped_column(String(36))
    observer_id: Mapped[int] = mapped_column(BigInteger)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    taxon_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("inat_taxa.taxon_id"))
    observed_on: Mapped[str | None] = mapped_column(Date)
    extension: Mapped[str] = mapped_column(String(10))
    license: Mapped[str] = mapped_column(String(20))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    position: Mapped[int | None] = mapped_column(Integer)

    taxon: Mapped["InatTaxa | None"] = relationship(
        back_populates="filtered_observations"
    )
    classification_entry: Mapped["InatClassificationDataset | None"] = relationship(
        back_populates="observation"
    )

    __table_args__ = (
        Index(
            "ix_inat_filtered_observations_latitude_longitude", "latitude", "longitude"
        ),
        Index("ix_inat_filtered_observations_observed_on", "observed_on"),
        Index("ix_inat_filtered_observations_taxon_id", "taxon_id"),
    )


class LilaAnnotations(Base):
    __tablename__ = "lila_annotations"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    image_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("lila_collected_images.id")
    )
    category_id: Mapped[str] = mapped_column(String(255))
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    w: Mapped[float] = mapped_column(Float)
    h: Mapped[float] = mapped_column(Float)

    collected_images: Mapped["LilaCollectedImages"] = relationship(
        back_populates="annotations"
    )

    __table_args__ = (
        Index("ix_lila_annotations_image_id", "image_id"),
        Index("ix_lila_annotations_category_id", "category_id"),
    )


class LilaCollectedImages(Base):
    __tablename__ = "lila_collected_images"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    file_name: Mapped[str] = mapped_column(String(255), unique=True)
    dataset: Mapped[str] = mapped_column(String(255))
    is_train: Mapped[bool] = mapped_column(Boolean)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)

    annotations: Mapped[list["LilaAnnotations"]] = relationship(
        back_populates="collected_images"
    )

    __table_args__ = (
        Index("ix_lila_collected_images_file_name", "file_name"),
        Index("ix_lila_collected_images_dataset", "dataset"),
    )


class InatCaptureContext(Base):
    """Underwater vs above-water classification for iNaturalist images.

    Stores raw per-channel means alongside the derived `is_underwater` verdict
    """

    __tablename__ = "inat_capture_context"

    photo_uuid: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("inat_filtered_observations.photo_uuid"),
        primary_key=True,
    )
    mean_r: Mapped[float | None] = mapped_column(Float)
    mean_g: Mapped[float | None] = mapped_column(Float)
    mean_b: Mapped[float | None] = mapped_column(Float)
    stddev: Mapped[float | None] = mapped_column(Float)
    is_underwater: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (Index("ix_inat_capture_context_is_underwater", "is_underwater"),)


class InatClipContext(Base):
    """Underwater vs above-water classification for iNaturalist images.

    Previously we had leveraged a heuristic to determine what was underwater and abovewater.
    We now leverage a CLIP models to no-shot predict whether an image is one of the above classes
    the outputs of that are stored here.
    """

    __tablename__ = "inat_clip_context"

    photo_uuid: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("inat_filtered_observations.photo_uuid"),
        primary_key=True,
    )
    is_underwater: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (Index("ix_inat_clip_context_is_underwater", "is_underwater"),)


class InatImageQuality(Base):
    """UIQM quality scores for iNaturalist images."""

    __tablename__ = "inat_image_quality"

    photo_uuid: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("inat_filtered_observations.photo_uuid"),
        primary_key=True,
    )
    uicm: Mapped[float | None] = mapped_column(Float)
    uism: Mapped[float | None] = mapped_column(Float)
    uiconm: Mapped[float | None] = mapped_column(Float)
    uiqm: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (Index("ix_inat_image_quality_uiqm", "uiqm"),)


class InatClassificationDataset(Base):
    """
    The final table containing all taxa with more than 300 underwater observations.
    The observations have been ranked by UIQM with a max sample count per taxon of 300.
    These observations are clustered and then entered into this table where they will be leveraged
    in the CV Classification.
    Species, genus, family integers all added for top-3 classification.
    """

    __tablename__ = "inat_classification_dataset"

    photo_uuid: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("inat_filtered_observations.photo_uuid"),
        primary_key=True,
    )
    uiqm: Mapped[float | None] = mapped_column(Float)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    taxon_id: Mapped[int | None] = mapped_column(Integer)
    uiqm_rank: Mapped[int | None] = mapped_column(Integer)
    filename: Mapped[str | None] = mapped_column(String)
    is_underwater: Mapped[int | None] = mapped_column(Integer)
    ancestry: Mapped[str | None] = mapped_column(String)
    species: Mapped[int | None] = mapped_column(Integer)
    genus: Mapped[int | None] = mapped_column(Integer)
    family: Mapped[int | None] = mapped_column(Integer)
    cluster: Mapped[int | None] = mapped_column(Integer)
    train: Mapped[bool | None] = mapped_column(Boolean)
    val_topup: Mapped[bool | None] = mapped_column(Boolean)
    proposed_bbox: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    conf: Mapped[float | None] = mapped_column(Float)
    annotation: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    zero_indexed_family: Mapped[int | None] = mapped_column(Integer)
    zero_indexed_genus: Mapped[int | None] = mapped_column(Integer)
    zero_indexed_species: Mapped[int | None] = mapped_column(Integer)

    observation: Mapped["InatFilteredObservations"] = relationship(
        back_populates="classification_entry"
    )

    __table_args__ = (
        Index("ix_inat_classification_dataset_cluster", "cluster"),
        Index("ix_inat_classification_dataset_uiqm", "uiqm"),
        Index("ix_inat_classification_dataset", "train"),
    )


class InatObjDetectionDataset(Base):
    """iNat images used to train the YOLO detector (LC1 / LC2 curricula).

    Populated by InatPreparation alongside inat_classification_dataset.
    All rows go here (fish + coral negatives); inat_classification_dataset
    receives only the non-coral subset.
    """

    __tablename__ = "inat_obj_detection_dataset"

    photo_uuid: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("inat_filtered_observations.photo_uuid"),
        primary_key=True,
    )
    filename: Mapped[str | None] = mapped_column(String)
    uiqm: Mapped[float | None] = mapped_column(Float)
    train: Mapped[bool | None] = mapped_column(Boolean)
    proposed_bbox: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    conf: Mapped[float | None] = mapped_column(Float)
    annotation: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    __table_args__ = (
        Index("ix_inat_obj_detection_dataset_train", "train"),
        Index("ix_inat_obj_detection_dataset_uiqm", "uiqm"),
    )


class AppTaxa(Base):
    __tablename__ = "app_taxa"

    taxon_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    zero_indexed_species: Mapped[int | None] = mapped_column(Integer)
    zero_indexed_genus: Mapped[int | None] = mapped_column(Integer)
    zero_indexed_family: Mapped[int | None] = mapped_column(Integer)
    species: Mapped[str] = mapped_column(String)
    genus: Mapped[str] = mapped_column(String)
    family: Mapped[str] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(Text)
    common_name: Mapped[str | None] = mapped_column(String)
    location: Mapped[list[str] | None] = mapped_column(ARRAY(String))
    depth: Mapped[str | None] = mapped_column(String)
    filename: Mapped[str | None] = mapped_column(String)
    img_count: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (Index("ix_app_taxa_taxon_id", "taxon_id"),)


class LilaImageQuality(Base):
    """UIQM quality scores for LILA images.

    Populated by a one-time scoring pass over locally available LILA images.
    Sub-scores (uicm, uism, uiconm) are kept alongside the composite so that
    threshold experiments can re-weight components without re-scoring.
    """

    __tablename__ = "lila_image_quality"

    file_name: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("lila_collected_images.file_name"),
        primary_key=True,
    )
    uicm: Mapped[float | None] = mapped_column(Float)
    uism: Mapped[float | None] = mapped_column(Float)
    uiconm: Mapped[float | None] = mapped_column(Float)
    uiqm: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (Index("ix_lila_image_quality_uiqm", "uiqm"),)


class LilaYolo(Base):
    __tablename__ = "lila_yolo"

    file_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    annotation: Mapped[dict[str, Any]] = mapped_column(JSONB)

    __table_args__ = (Index("ix_lila_yolo_file_name", "file_name"),)


class SuccessfulUploads(Base):
    __tablename__ = "successful_uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identifier: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(255))
    uploaded_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("identifier", "source", name="uq_identifier_source"),
        Index("ix_successful_uploads_source", "source"),
    )


# ── Phase 3: user observation tracking ──────────────────────────────────────
# iNat-style normalization: a dive (one place + time) groups observations (one
# taxon each), and each observation has photos (evidence, classified per-photo).
# UUID PKs for user-generated rows so ids are unguessable and client-mintable.


class User(Base):
    """An authenticated user. Identity comes from Google via GCP Identity
    Platform; we key our own row on the OIDC subject so the app data (dives,
    observations) has a stable local owner independent of the token issuer."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    google_subject_id: Mapped[str] = mapped_column(String(255), unique=True)
    email: Mapped[str | None] = mapped_column(String(320))
    display_name: Mapped[str | None] = mapped_column(String(255))
    avatar_url: Mapped[str | None] = mapped_column(Text)
    # App-owned profile fields. These are NOT touched by the Google claims sync
    # (upsert_from_claims), so a user's chosen name/units survive every re-login.
    preferred_name: Mapped[str | None] = mapped_column(String(255))
    unit_system: Mapped[str] = mapped_column(
        String(10), server_default="metric", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime)

    dives: Mapped[list["Dive"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    observations: Mapped[list["Observation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )


class DiveRegion(Base):
    """A place in the geographic/political dive-location hierarchy
    (continent → country → dive area). Preseeded and globally shared; the anchor
    for species ranges (species_regions), user dive sites, and future quests.
    Self-referential via `parent_id` (the enclosing region).

    Dedup is scoped to the parent (uq parent_id + name_key), not global, so a
    country and a like-named dive area don't collide. Continents have a NULL
    parent — Postgres treats NULLs as distinct, so the seed script must dedup
    continents by explicit existence check rather than lean on the constraint."""

    __tablename__ = "dive_regions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    kind: Mapped[str] = mapped_column(String(20))  # 'continent' | 'country' | 'area'
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dive_regions.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(255))
    name_key: Mapped[str] = mapped_column(String(255))
    # Natural Earth ISO_A2 — mostly ISO 3166-1 alpha-2, but some rows carry
    # variants ('CN-TW', '-99', …), so no length cap. seed_regions and
    # derive_species_regions read the same field, so the join stays consistent.
    iso_country: Mapped[str | None] = mapped_column(String)
    lat: Mapped[float | None] = mapped_column(Float)  # centroid
    lng: Mapped[float | None] = mapped_column(Float)

    parent: Mapped["DiveRegion | None"] = relationship(
        "DiveRegion", remote_side=[id], backref="children"
    )
    sites: Mapped[list["DiveSite"]] = relationship(back_populates="region")

    __table_args__ = (
        UniqueConstraint("parent_id", "name_key", name="uq_dive_regions_parent_name"),
        Index("ix_dive_regions_kind", "kind"),
        Index("ix_dive_regions_iso_country", "iso_country"),
    )


class SpeciesRegion(Base):
    """Many-to-many: which regions a species is found in. Backfilled from
    observation coordinates (point-in-polygon over country polygons); later
    augmentable from LLM/manual sources. Replaces the flat app_taxa.location
    array as the structured source of species ranges.

    Composite PK gives the taxon-leading index; the explicit region_id index
    powers the reverse lookup (species-in-region — quests, discovery)."""

    __tablename__ = "species_regions"

    taxon_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("inat_taxa.taxon_id", ondelete="CASCADE"),
        primary_key=True,
    )
    region_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dive_regions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    source: Mapped[str | None] = mapped_column(String(20))  # observation|llm|manual
    obs_count: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (Index("ix_species_regions_region_id", "region_id"),)


class DiveSite(Base):
    """A deduplicated named location. `name` is the proper-cased display form;
    `name_key` is the normalized (lower/trimmed) dedup key carrying the UNIQUE
    constraint, so a site-search can surface near-duplicates before insert."""

    __tablename__ = "dive_sites"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255))
    name_key: Mapped[str] = mapped_column(String(255), unique=True)
    lat: Mapped[float | None] = mapped_column(Float)
    lng: Mapped[float | None] = mapped_column(Float)
    # Region the site sits in (many sites per region); Google Places id captured
    # when the site is resolved through Places Autocomplete.
    region_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dive_regions.id", ondelete="SET NULL")
    )
    google_place_id: Mapped[str | None] = mapped_column(String(255))
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    created_by: Mapped["User | None"] = relationship()
    region: Mapped["DiveRegion | None"] = relationship(back_populates="sites")
    dives: Mapped[list["Dive"]] = relationship(back_populates="site")


class Dive(Base):
    """One dive: a place + time that groups many observations. Location (site +
    optional GPS) lives here, normalized once per dive rather than per sighting."""

    __tablename__ = "dives"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    site_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dive_sites.id", ondelete="SET NULL")
    )
    gps_lat: Mapped[float | None] = mapped_column(Float)
    gps_lng: Mapped[float | None] = mapped_column(Float)
    dived_at: Mapped[datetime | None] = mapped_column(DateTime)
    notes: Mapped[str | None] = mapped_column(Text)  # "Dive Comments"
    # Extended dive-log fields, stored METRIC-CANONICAL (frontend converts per
    # unit_system). "Notable Nature"/"Photos Taken" are the observations/photos below.
    visibility_m: Mapped[float | None] = mapped_column(Float)
    air_temp_c: Mapped[float | None] = mapped_column(Float)
    water_temp_c: Mapped[float | None] = mapped_column(Float)
    weight_kg: Mapped[float | None] = mapped_column(Float)
    exposure_suit: Mapped[str | None] = mapped_column(String(100))
    depth_avg_m: Mapped[float | None] = mapped_column(Float)
    depth_max_m: Mapped[float | None] = mapped_column(Float)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)  # in-water start
    bottom_time_min: Mapped[int | None] = mapped_column(Integer)  # "Time on Dive"
    total_time_min: Mapped[int | None] = mapped_column(Integer)  # "Total Dive Time"
    end_pressure_bar: Mapped[float | None] = mapped_column(Float)  # ending PSI/Bar
    dive_shop: Mapped[str | None] = mapped_column(String(255))
    # PADI-verification seam (Phase 2 investigation). Non-null so history queries
    # don't COALESCE; verified_source names the origin once verification exists.
    verified: Mapped[bool] = mapped_column(
        Boolean, server_default="false", nullable=False
    )
    verified_source: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="dives")
    site: Mapped["DiveSite | None"] = relationship(back_populates="dives")
    observations: Mapped[list["Observation"]] = relationship(
        back_populates="dive", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (Index("ix_dives_user_id_dived_at", "user_id", "dived_at"),)


class Observation(Base):
    """One taxon encounter within a dive. The model's per-photo guesses live on
    ObservationPhoto; this row carries the user's chosen ID — `corrected_taxon_id`
    overrides `predicted_taxon_id` (the seam for the deferred taxa picker).

    Taxon refs are the **stable iNat `taxon_id`**, NOT the model's zero-index
    (which re-numbers on retraining and would corrupt history). They FK the full
    `inat_taxa` (~44K, superset of the trained `app_taxa`), so a correction can
    range over the whole in-scope taxonomy — the classifier's zero-index is
    translated to a taxon_id via `app_taxa` at save time.

    `user_id` is denormalized off the dive so ownership scoping is a single filter.
    Depth lives here (it varies by sighting within a dive)."""

    __tablename__ = "observations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dive_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dives.id", ondelete="CASCADE")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    predicted_taxon_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("inat_taxa.taxon_id")
    )
    # The EFFECTIVE label — always set (defaults to predicted_taxon_id on save),
    # so history groups on one column with no COALESCE. label_status records the
    # provenance for training reliability:
    #   predicted  — saved, model's guess unvalidated
    #   confirmed  — user validated the prediction as correct
    #   corrected  — user changed it to corrected_taxon_id (a suggested id)
    corrected_taxon_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("inat_taxa.taxon_id"), nullable=False
    )
    label_status: Mapped[str] = mapped_column(
        String(20), server_default="predicted", nullable=False
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    depth_m: Mapped[float | None] = mapped_column(Float)
    observed_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    dive: Mapped["Dive"] = relationship(back_populates="observations")
    user: Mapped["User"] = relationship(back_populates="observations")
    predicted_taxon: Mapped["InatTaxa | None"] = relationship(
        foreign_keys=[predicted_taxon_id]
    )
    corrected_taxon: Mapped["InatTaxa | None"] = relationship(
        foreign_keys=[corrected_taxon_id]
    )
    photos: Mapped[list["ObservationPhoto"]] = relationship(
        back_populates="observation", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (Index("ix_observations_user_id", "user_id"),)


class ObservationPhoto(Base):
    """A photo as evidence for an observation. Each photo is independently run
    through the detector→classifier pipeline, so its own box + top guess are
    stored here. `predicted_taxon_id` is the stable iNat taxon_id (translated from
    the model's zero-index). `image_path` is the GCS key under contributions/{user_id}/."""

    __tablename__ = "observation_photos"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    observation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("observations.id", ondelete="CASCADE")
    )
    image_path: Mapped[str] = mapped_column(String(512))
    bbox: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    predicted_taxon_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("inat_taxa.taxon_id")
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    # The user's chosen "card image" for this photo's effective species. At most
    # one photo per (user, corrected_taxon_id) is the hero; the service clears any
    # prior hero for that taxon when a new one is set. Fallback (none set) is the
    # first photo, so this is purely an override.
    is_hero: Mapped[bool] = mapped_column(
        Boolean, server_default="false", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    observation: Mapped["Observation"] = relationship(back_populates="photos")
    predicted_taxon: Mapped["InatTaxa | None"] = relationship()

    __table_args__ = (Index("ix_observation_photos_observation_id", "observation_id"),)
