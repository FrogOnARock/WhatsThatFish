from fastapi import APIRouter, File, UploadFile, Depends

from ..dependencies import get_prediction_service
from ..schemas import Prediction
from ..error import InvalidPredictionRequest
from whatsthatfish.serving.services.service import PredictionService
from ..utils import StorageConstructor, MAX_UPLOAD_BYTES


router = APIRouter()


@router.post("/predict", response_model=Prediction)
async def get_prediction(
    img: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service),
) -> Prediction:
    """Run detector, crop, classifier on uploaded image bytes and return the
    best box plus predicted species."""
    # Reject non-image uploads here, where the request metadata still exists, so
    # the 422 body carries the filename/content-type the service can't see.
    if not (img.content_type or "").startswith("image/"):
        raise InvalidPredictionRequest(
            message="Upload must be an image",
            body={"filename": img.filename, "content_type": img.content_type},
        )
    # Reject oversized uploads before buffering/inference. This endpoint is
    # unauthenticated, so the size cap is the main guard against a single request
    # amplifying memory / disk-spool / CPU-inference cost.
    if img.size is not None and img.size > MAX_UPLOAD_BYTES:
        raise InvalidPredictionRequest(
            message="Image exceeds the 15 MB upload limit",
            body={"filename": img.filename, "size": img.size},
        )
    image_bytes = await img.read()
    return service.get_prediction(image_bytes)


@router.get("/predict/sample/{filename}", response_model=Prediction)
async def get_prediction_sample(
    filename: str, service: PredictionService = Depends(get_prediction_service)
) -> Prediction:
    """Run inference pipeline on a sample image"""
    storage = StorageConstructor().constructor()
    image_bytes = storage.read_bytes(filename)
    return service.get_prediction(image_bytes)
