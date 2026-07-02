from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..dependencies import get_session
from whatsthatfish.serving.services.service import TaxaService
from ..utils import StorageConstructor
from ..schemas import SpeciesCatalogue, TaxonOption, ModelStats
from ..error import ResourceNotFoundException, ValidationException

router = APIRouter()


@router.get("/stats", response_model=ModelStats)
def model_stats(session: Session = Depends(get_session)) -> ModelStats:
    """Trained-class counts (species/genera/families) — replaces hardcoded UI
    numbers. Cheap COUNT(DISTINCT) over app_taxa."""
    return TaxaService(session).get_stats()


@router.get("/taxa/species", response_model=list[TaxonOption])
def search_species(
    q: str = "", session: Session = Depends(get_session)
) -> list[TaxonOption]:
    """Correction picker source: rank='species' fish/shark taxa matching `q`
    (empty `q` returns the first alphabetical page). Public reference data."""
    return TaxaService(session).search_species(q)


@router.get("/image/{filename}")
async def get_image(filename: str):
    """Serve a catalogue image by filename via the environment-appropriate backend.

    Locally this returns the file directly; on Cloud Run it redirects to a
    signed GCS URL — the caller doesn't need to know which.
    """
    storage = StorageConstructor().constructor()
    url = storage.retrieve_image(filename=filename)
    return url


@router.get("/species", response_model=SpeciesCatalogue)
async def list_species(session: Session = Depends(get_session)) -> SpeciesCatalogue:
    """Endpoint backing the frontend's Species Library — the full catalogue plus a count."""
    service = TaxaService(session)
    entries = service.get_species()
    return SpeciesCatalogue(species=entries, total=len(entries))
