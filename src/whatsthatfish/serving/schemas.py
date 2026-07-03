"""Pydantic response models — the serving-layer contract.

These mirror the frontend's `src/api/types.ts`. Keep the two in sync: a field
renamed here must be renamed there (and vice versa), since the SPA deserialises
exactly these shapes.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

LabelStatus = Literal["predicted", "confirmed", "corrected"]
UnitSystem = Literal["metric", "imperial"]


class UserProfile(BaseModel):
    """The authenticated user as the SPA needs it — id plus Google profile bits,
    plus the app-owned editable fields (preferred_name / unit_system)."""

    id: str
    email: str | None
    display_name: str | None
    avatar_url: str | None
    preferred_name: str | None = None
    unit_system: UnitSystem = "metric"


class UserSettingsUpdate(BaseModel):
    """PATCH the signed-in user's app-owned profile. Only provided keys change
    (exclude_unset). preferred_name='' clears the override; the Google-sourced
    name/email/avatar are never editable here."""

    preferred_name: str | None = None
    unit_system: UnitSystem | None = None


class UserStats(BaseModel):
    """Summary counts for the Settings page."""

    dives: int
    observations: int
    unique_species: int


class SpeciesEntry(BaseModel):
    """One row in the Species Library catalogue.

    Names are SCIENTIFIC (iNat `inat_taxa.name`); the schema has no common
    names. `species_id` is the zero-indexed class id — the stable identifier the
    classifier's species head emits, and what the frontend keys cards on.
    """

    species_id: int
    name: str
    genus: str
    family: str
    image_count: int
    common_name: str
    description: str
    location: list[str]
    filename: str
    depth: str


class SpeciesCatalogue(BaseModel):
    """Envelope so we can add catalogue-level metadata without reshaping the list."""

    species: list[SpeciesEntry]
    total: int


class SpeciesInfo(BaseModel):
    """The LLM-enriched fields for a species — common name, blurb, range, depth.

    The field descriptions double as the prompt contract the LLM fills in.
    """

    common_name: str = Field(description="Common English name, e.g. 'Clownfish'")
    description: str = Field(description="A 100 word description of the species")
    location: list[str] = Field(
        description="The most common location(s) for that species"
    )
    depth: str = Field(
        description="The average depth that species is found at in comma separated meters and feet metrics.  e.g., '1-5 meters, 0-15 feet'"
    )


class Bbox(BaseModel):
    """The detected box as PERCENT of the original image dims — top-left (x, y)
    plus width/height. Percent (not pixels) so it overlays correctly on the
    CSS-scaled <img>; xywh (not xyxy) so it maps 1:1 to CSS left/top/width/height."""

    x: float
    y: float
    w: float
    h: float


class Candidate(BaseModel):
    """
    The candidate predictions, along with their softmaxed probabilities
    """

    name: str
    index: int
    conf: float
    summary: str | None
    common: str | None
    habitat: list[str] | None


class Prediction(BaseModel):
    """One detector→classifier result: the best box(es) and the ranked species/
    genus/family candidates.

    `detected` is False when the detector found no fish: `bbox` is then empty but
    the classifier still ran on the full frame, so `species`/`genus`/`family` are
    populated — just out-of-distribution and low-trust. The UI uses this to warn.
    """

    bbox: list[Bbox]
    species: list[Candidate]
    genus: list[Candidate]
    family: list[Candidate]
    detected: bool


# ── History / observation tracking ──────────────────────────────────────────


class DiveCreate(BaseModel):
    """Create a dive. `site_name` resolves-or-creates a deduplicated dive_site;
    location/time live on the dive (depth is per-observation)."""

    site_name: str | None = None
    gps_lat: float | None = None
    gps_lng: float | None = None
    dived_at: datetime | None = None
    notes: str | None = None


class DiveUpdate(BaseModel):
    """PATCH a dive — every field optional; only provided keys are changed."""

    site_name: str | None = None
    gps_lat: float | None = None
    gps_lng: float | None = None
    dived_at: datetime | None = None
    notes: str | None = None


class DiveSpecies(BaseModel):
    """A distinct species logged on a dive — for the dive-detail popup's
    'what you saw' summary. Keyed on the effective (corrected) taxon."""

    taxon_id: int
    name: str | None
    common_name: str | None


class SiteOption(BaseModel):
    """One existing dive site, surfaced by the autocomplete so users reuse a
    site rather than creating a near-duplicate."""

    id: UUID
    name: str


class DiveOut(BaseModel):
    id: UUID
    site_id: UUID | None
    site_name: str | None
    gps_lat: float | None
    gps_lng: float | None
    dived_at: datetime | None
    notes: str | None
    created_at: datetime
    # Summary fields for the Dive Log table + detail popup.
    observation_count: int = 0
    species: list[DiveSpecies] = []


class ObservationCreate(BaseModel):
    """Save an identification. The client sends the model's zero-indices (what it
    has); the server translates them to stable iNat taxon_ids via app_taxa.

    Effective-label precedence: `corrected_taxon_id` (a real iNat taxon picked
    from the full species list — the report flow) > `corrected_species_index` (a
    model candidate the user selected) > the prediction itself.
    `label_status`: predicted (saved as-is) · confirmed (user validated) ·
    corrected (user changed it)."""

    dive_id: UUID
    predicted_species_index: int
    corrected_species_index: int | None = None
    corrected_taxon_id: int | None = None
    label_status: LabelStatus = "predicted"
    confidence: float | None = None
    depth_m: float | None = None
    observed_at: datetime | None = None


class ModelStats(BaseModel):
    """Trained-class counts for the UI (replaces hardcoded numbers)."""

    species: int
    genera: int
    families: int


class TaxonOption(BaseModel):
    """One selectable taxon in the correction picker — scientific name + common
    name where we have it (only the trained subset has common names)."""

    taxon_id: int
    name: str
    common_name: str | None


class ObservationUpdate(BaseModel):
    """PATCH a sighting — only the keys the client sends are changed (the service
    uses exclude_unset, so `depth_m: null` clears it, omitting it leaves it)."""

    corrected_taxon_id: int | None = None
    label_status: LabelStatus | None = None
    depth_m: float | None = None


class ObservationOut(BaseModel):
    id: UUID
    dive_id: UUID
    predicted_taxon_id: int | None
    corrected_taxon_id: int
    label_status: LabelStatus
    confidence: float | None
    depth_m: float | None
    observed_at: datetime | None


class PhotoOut(BaseModel):
    id: UUID
    observation_id: UUID
    image_path: str
    bbox: dict | None
    confidence: float | None
    width: int | None
    height: int | None


class HistorySighting(BaseModel):
    """One observation as shown in the field-log detail panel."""

    observation_id: UUID
    dive_id: UUID
    dived_at: datetime | None
    site_name: str | None
    depth_m: float | None
    label_status: LabelStatus
    photos: list[PhotoOut]


class HistorySpecies(BaseModel):
    """A field-log card: one effective taxon the user has logged, with its
    sightings. Keyed on corrected_taxon_id (the effective label)."""

    taxon_id: int
    species: str | None
    genus: str | None
    family: str | None
    common_name: str | None
    sighting_count: int
    sightings: list[HistorySighting]


class FieldLog(BaseModel):
    species: list[HistorySpecies]
    total_species: int
