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
from fastapi import FastAPI, Depends, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from contextlib import asynccontextmanager

from ..database.config import get_session_factory
from .schemas import SpeciesCatalogue, SpeciesEntry, Prediction, Bbox, Candidate
from ..database.models import AppTaxa, InatTaxa
from .utils import StorageConstructor
from ..config import _get_logger
from ..inference.class_inference import ClassInference
from ..inference.bbox_inference import BoundingBoxInference
from .service import TaxaService, PredictionService

load_dotenv()
logger = _get_logger("uvicorn.error")

async def lifespan(app: FastAPI):
    app.state.bbox_inferrer = BoundingBoxInference(conf=0.25)
    app.state.class_inferrer = ClassInference()
    yield
app = FastAPI(title="WhatsThisFish API", version="0.1.0", lifespan=lifespan)

# Vite dev server origin. When we deploy, add the Cloudflare Pages domain here
# (or front both behind one domain to avoid CORS entirely — see hosting notes).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


_session_factory = get_session_factory()


def get_session():
    with _session_factory() as session:
        yield session


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — a cheap 200 so Cloud Run knows the container is up."""
    return {"status": "ok"}


@app.get("/image/{filename}")
async def get_image(filename: str):
    """Serve a catalogue image by filename via the environment-appropriate backend.

    Locally this returns the file directly; on Cloud Run it redirects to a
    signed GCS URL — the caller doesn't need to know which.
    """
    storage = StorageConstructor().constructor()
    url = storage.retrieve_image(filename=filename)
    return url


@app.get("/species", response_model=SpeciesCatalogue)
async def list_species(session: Session = Depends(get_session)) -> SpeciesCatalogue:
    """Endpoint backing the frontend's Species Library — the full catalogue plus a count."""
    service = TaxaService(session)
    entries = service.get_species()
    return SpeciesCatalogue(species=entries, total=len(entries))


def get_prediction_service(
        request: Request,
        session: Session = Depends(get_session)
) -> PredictionService:
    return PredictionService(
        session=session,
        bbox_inferrer=request.app.state.bbox_inferrer,
        class_inferrer=request.app.state.class_inferrer
    )

@app.get("/predict/{img}", response_model=Prediction)
async def get_prediction(
    img: UploadFile = File(...), service: PredictionService = Depends(get_prediction_service)
) -> Prediction:
    """Run detector → crop → classifier on uploaded image bytes and return the
    best box plus predicted species. WIP — implementation owned by the author."""
    image_bytes = await img.read()
    return service.get_prediction(image_bytes)
