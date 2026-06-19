from abc import ABC, abstractmethod
from google.cloud import storage as gcs
from google.oauth2 import service_account
import os
from pathlib import Path
from fastapi.responses import FileResponse
from starlette.responses import RedirectResponse

from ..etl.gcs_client import GCSClient
from ..config import get_config


class ImageStorage(ABC):
    """Abstract image source — hides whether a frame lives on disk or in GCS.

    The serving layer just asks for a filename; concrete backends decide how
    to hand it back (a file response locally, a signed-URL redirect in cloud).
    """

    @abstractmethod
    def retrieve_image(self, filename): ...


class GCSImage(ImageStorage):
    """Cloud backend: serves training images from the GCS bucket.

    Returns a short-lived signed-URL redirect so the client fetches the blob
    straight from GCS rather than streaming bytes through the API.
    """

    def __init__(self):
        self.config = get_config().gcs
        self.client = GCSClient(self.config).get_gcs_client()
        self.bucket = self.client.bucket(self.config.bucket)
        self.prefix = self.config.prefixes["training"]

    def retrieve_image(self, filename: str):
        """Mint a 15-minute signed URL for the blob and redirect the client to it."""
        blob = self.bucket.blob(f"{self.prefix}/{filename}")
        url = blob.generate_signed_url(expiration=900)
        return RedirectResponse(url)


class LocalImage(ImageStorage):
    """Dev backend: serves images straight off the local classification-images folder."""

    def __init__(self, folder: str):
        self.folder_path = folder

    def retrieve_image(self, filename):
        """Stream the requested image file back as a JPEG response."""
        return FileResponse(f"{self.folder_path}/{filename}", media_type="image/jpeg")


class StorageConstructor:
    """Picks the right image backend for the current environment.

    Uses GCS when running on Cloud Run (detected via the `K_SERVICE` env var),
    otherwise falls back to reading from the local data folder.
    """

    def __init__(
        self, folder: str = Path(__file__).parents[1] / "data/classification_images"
    ):
        self.folder = folder

    def constructor(self):
        """Return a GCSImage on Cloud Run, else a LocalImage over `self.folder`."""
        if os.getenv("K_SERVICE"):
            return GCSImage()
        else:
            return LocalImage(folder=self.folder)
