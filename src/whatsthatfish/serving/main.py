"""FastAPI serving app — the local (and eventual Cloud Run) HTTP layer.

Run locally:
    uv run uvicorn whatsthatfish.serving.app:app --reload --port 8000

Then: http://localhost:8000/species  ·  docs at http://localhost:8000/docs

This is intentionally a THIN read layer over the existing SQLAlchemy stack —
it reuses `get_session_factory()` (sync), so endpoints are plain `def` and
FastAPI runs them in its threadpool. No new DB wiring.
"""


from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pathlib import Path
import os

from ..config import _get_logger
from ..inference.class_inference import OnnxClassInference
from ..inference.bbox_inference import OnnxBoundingBoxInference
from .routers import library, predictions, auth, history
from .error import BaseAppException, InvalidPredictionRequest


load_dotenv()
logger = _get_logger("uvicorn.error")

# ONNX artifacts baked into the image (see Dockerfile COPY). Env overrides let
# local dev / the release job point elsewhere. Classifier ships INT8 (real
# Gemm-head speedup); detector ships FP32 (dynamic INT8 is slower on its Conv
# stack). Both run on CPUExecutionProvider inside the container.
_WEIGHTS = Path(__file__).parents[1] / "weights"
CLASSIFIER_ONNX = os.getenv("CLASSIFIER_ONNX", str(_WEIGHTS / "classifier.int8.onnx"))
DETECTOR_ONNX = os.getenv("DETECTOR_ONNX", str(_WEIGHTS / "lc1_best.onnx"))


async def lifespan(app: FastAPI):
    # Torch-free serving: both inferrers are onnxruntime sessions built once at
    # startup. Same .infer() contracts as the torch classes they replace, so the
    # PredictionService / router wiring is unchanged.
    app.state.bbox_inferrer = OnnxBoundingBoxInference(DETECTOR_ONNX, conf=0.15)
    app.state.class_inferrer = OnnxClassInference(CLASSIFIER_ONNX)
    yield


app = FastAPI(title="WhatsThisFish API", version="0.1.0", lifespan=lifespan)

# CORS origins. Local dev origins are always allowed; production origins come
# from ALLOWED_ORIGINS (comma-separated) so adding the Cloudflare Pages domain
# post-deploy is a `gcloud run --update-env-vars` flag, not a Docker rebuild.
# ALLOWED_ORIGIN_REGEX additionally covers Pages per-deploy preview subdomains
# (e.g. https://<hash>.whatsthatfish.pages.dev) which can't be enumerated.
_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_ENV_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX") or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_ORIGINS + _ENV_ORIGINS,
    allow_origin_regex=_ORIGIN_REGEX,
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
