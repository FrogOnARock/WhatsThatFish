"""Observation tracking — dives, observations, photos, and the field log.

Every route is auth-gated (get_current_user) and ownership-scoped: the service
filters by user.id, so a user can only touch their own rows.
"""

import json
from uuid import UUID

from fastapi import APIRouter, Depends, UploadFile, File, Form

from ..dependencies import get_current_user, get_observation_service
from ..error import ValidationException
from ..services.service import ObservationService
from ..utils import MAX_UPLOAD_BYTES
from ..schemas import (
    DiveCreate,
    DiveUpdate,
    DiveOut,
    ObservationCreate,
    ObservationUpdate,
    ObservationOut,
    PhotoOut,
    FieldLog,
    SiteOption,
    UserStats,
)
from whatsthatfish.database.models import User

router = APIRouter()


# ── dives ───────────────────────────────────────────────────────────────────
@router.post("/dives", response_model=DiveOut)
def create_dive(
    data: DiveCreate,
    user: User = Depends(get_current_user),
    svc: ObservationService = Depends(get_observation_service),
) -> DiveOut:
    return svc.create_dive(user, data)


@router.patch("/dives/{dive_id}", response_model=DiveOut)
def update_dive(
    dive_id: UUID,
    data: DiveUpdate,
    user: User = Depends(get_current_user),
    svc: ObservationService = Depends(get_observation_service),
) -> DiveOut:
    """Edit a dive's site/time/GPS/notes (only the fields provided)."""
    return svc.update_dive(user, dive_id, data)


@router.get("/dives", response_model=list[DiveOut])
def list_dives(
    user: User = Depends(get_current_user),
    svc: ObservationService = Depends(get_observation_service),
) -> list[DiveOut]:
    return svc.list_dives(user)


@router.get("/dive_sites", response_model=list[SiteOption])
def search_dive_sites(
    q: str = "",
    user: User = Depends(get_current_user),
    svc: ObservationService = Depends(get_observation_service),
) -> list[SiteOption]:
    """Autocomplete existing dive sites by substring, so users reuse a site
    rather than creating near-duplicates."""
    return svc.search_sites(q)


# ── observations ──────────────────────────────────────────────────────────────
@router.post("/observations", response_model=ObservationOut)
def create_observation(
    data: ObservationCreate,
    user: User = Depends(get_current_user),
    svc: ObservationService = Depends(get_observation_service),
) -> ObservationOut:
    return svc.create_observation(user, data)


@router.patch("/observations/{observation_id}", response_model=ObservationOut)
def update_observation(
    observation_id: UUID,
    data: ObservationUpdate,
    user: User = Depends(get_current_user),
    svc: ObservationService = Depends(get_observation_service),
) -> ObservationOut:
    """Edit a sighting's label / status / depth."""
    return svc.update_observation(user, observation_id, data)


# ── photos (multipart: the image plus its metadata) ──────────────────────────
@router.post("/observation_photos", response_model=PhotoOut)
async def add_photo(
    observation_id: UUID = Form(...),
    img: UploadFile = File(...),
    bbox: str | None = Form(None),  # JSON string {x,y,w,h} (percent)
    predicted_species_index: int | None = Form(None),
    confidence: float | None = Form(None),
    width: int | None = Form(None),
    height: int | None = Form(None),
    user: User = Depends(get_current_user),
    svc: ObservationService = Depends(get_observation_service),
) -> PhotoOut:
    if img.size is not None and img.size > MAX_UPLOAD_BYTES:
        raise ValidationException("Image exceeds the 15 MB upload limit")
    image_bytes = await img.read()
    try:
        parsed_bbox = json.loads(bbox) if bbox else None
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValidationException("Malformed bbox: expected a JSON object") from exc
    return svc.add_photo(
        user,
        observation_id,
        image_bytes,
        bbox=parsed_bbox,
        predicted_species_index=predicted_species_index,
        confidence=confidence,
        width=width,
        height=height,
    )


@router.get("/observation_photos/{photo_id}/image")
def photo_image(
    photo_id: UUID,
    user: User = Depends(get_current_user),
    svc: ObservationService = Depends(get_observation_service),
):
    """Serve a user's contribution photo (local file or signed-URL redirect)."""
    return svc.get_photo_image(user, photo_id)


# ── history / field log ───────────────────────────────────────────────────────
@router.get("/history", response_model=FieldLog)
def field_log(
    user: User = Depends(get_current_user),
    svc: ObservationService = Depends(get_observation_service),
) -> FieldLog:
    """The signed-in user's field log: species grouped by effective taxon, each
    with its sightings (dive date, site, depth) and photos."""
    return svc.get_field_log(user)


@router.get("/me/stats", response_model=UserStats)
def user_stats(
    user: User = Depends(get_current_user),
    svc: ObservationService = Depends(get_observation_service),
) -> UserStats:
    """Summary counts for the Settings page: dives, observations, unique species."""
    return svc.user_stats(user)
