"""FastAPI serving app — the local (and eventual Cloud Run) HTTP layer.

Run locally:
    uv run uvicorn whatsthatfish.serving.app:app --reload --port 8000

Then: http://localhost:8000/species  ·  docs at http://localhost:8000/docs

This is intentionally a THIN read layer over the existing SQLAlchemy stack —
it reuses `get_session_factory()` (sync), so endpoints are plain `def` and
FastAPI runs them in its threadpool. No new DB wiring.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from ..database.config import get_session_factory
from .schemas import SpeciesCatalogue, SpeciesEntry, ModelPrediction
from ..database.models import AppTaxa
from .utils import StorageConstructor
from ..config import _get_logger

load_dotenv()
logger = _get_logger("uvicorn.error")

app = FastAPI(title="WhatsThisFish API", version="0.1.0")

# Vite dev server origin. When we deploy, add the Cloudflare Pages domain here
# (or front both behind one domain to avoid CORS entirely — see hosting notes).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_session_factory = get_session_factory()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — a cheap 200 so Cloud Run knows the container is up."""
    return {"status": "ok"}


def query_species(session: Session) -> list[SpeciesEntry]:
    """Return the species catalogue: one entry per distinct trained species."""

    rows = session.execute(select(AppTaxa)).scalars().all()

    return [
        SpeciesEntry(
            species_id=row.zero_indexed_species,
            name=row.species,
            genus=row.genus,
            family=row.family,
            image_count=row.img_count,
            description=row.description,
            filename=row.filename,
            common_name=row.common_name,
            location=row.location,
            depth=row.depth
        )
        for row in rows
    ]

@app.get("/image/{filename}")
def get_image(filename: str):
    """Serve a catalogue image by filename via the environment-appropriate backend.

    Locally this returns the file directly; on Cloud Run it redirects to a
    signed GCS URL — the caller doesn't need to know which.
    """
    storage = StorageConstructor().constructor()
    url = storage.retrieve_image(filename=filename)
    return url


@app.get("/species", response_model=SpeciesCatalogue)
def list_species() -> SpeciesCatalogue:
    """Endpoint backing the frontend's Species Library — the full catalogue plus a count."""
    with _session_factory() as session:
        entries = query_species(session)
    return SpeciesCatalogue(species=entries, total=len(entries))


@app.get("/predict", response_model=ModelPrediction)
def get_prediction(img: bytes | list[bytes]) -> ModelPrediction:
    """Run detector → crop → classifier on uploaded image bytes and return the
    best box plus predicted species. WIP — implementation owned by the author."""
    raise NotImplementedError
