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
from fastapi.responses import JSONResponse
from starlette.responses import PlainTextResponse
from fastapi.encoders import jsonable_encoder

from ..config import _get_logger
from ..inference.class_inference import ClassInference
from ..inference.bbox_inference import BoundingBoxInference
from .routers import library, predictions, auth, history
from .error import BaseAppException, InvalidPredictionRequest


load_dotenv()
logger = _get_logger("uvicorn.error")


async def lifespan(app: FastAPI):
    app.state.bbox_inferrer = BoundingBoxInference(conf=0.15)
    app.state.class_inferrer = ClassInference()
    yield


app = FastAPI(title="WhatsThisFish API", version="0.1.0", lifespan=lifespan)

# Vite dev server origin. When we deploy, add the Cloudflare Pages domain here
# (or front both behind one domain to avoid CORS entirely — see hosting notes).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)
app.include_router(predictions.router)
app.include_router(library.router)
app.include_router(auth.router)
app.include_router(history.router)


@app.exception_handler(BaseAppException)
async def base_exception_handler(request: Request, exc: BaseAppException):
    # One handler covers EVERY BaseAppException subclass — FastAPI matches by the
    # exception's MRO, so ResourceNotFound (404), Validation (400),
    # Authentication (401), InvalidPredictionResponse (500) all land here and
    # return their own status_code. (Was reading exc.detail, which doesn't exist
    # on BaseAppException — it's exc.message — so the handler itself 500'd.)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(InvalidPredictionRequest)
async def invalid_prediction_request_handler(
    request: Request, exc: InvalidPredictionRequest
):
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — a cheap 200 so Cloud Run knows the container is up."""
    return {"status": "ok"}
