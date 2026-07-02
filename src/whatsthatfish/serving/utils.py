from abc import ABC, abstractmethod
from google.oauth2 import service_account
import os
from pathlib import Path
from fastapi.responses import FileResponse
from starlette.responses import RedirectResponse
from google.api_core.exceptions import (
    ClientError as GCSClientError,
    ServerError as GCSServerError,
    ServiceUnavailable,
    TooManyRequests,
    GoogleAPICallError,
)


from ..retry import gcs_retry
from ..etl.gcs_client import GCSClient
from ..config import get_config
from .error import ResourceNotFoundException, BaseAppException


class ImageStorage(ABC):
    """Abstract image source — hides whether a frame lives on disk or in GCS.

    The serving layer just asks for a filename; concrete backends decide how
    to hand it back (a file response locally, a signed-URL redirect in cloud).
    """

    @abstractmethod
    def retrieve_image(self, filename): ...

    @abstractmethod
    def read_bytes(self, filename): ...


class GCSImage(ImageStorage):
    """Cloud backend: serves training images from the GCS bucket.

    Returns a short-lived signed-URL redirect so the client fetches the blob
    straight from GCS rather than streaming bytes through the API.
    """

    # @gcs_retry
    def __init__(self):
        self.config = (
            get_config().gcs
        )  # Going to have to change this to an environment variable
        self.client = GCSClient(self.config).get_gcs_client()
        self.bucket = self.client.bucket(self.config.bucket)
        self.prefix = self.config.prefixes["training"]

    def retrieve_image(self, filename: str):
        """Mint a 15-minute signed URL for the blob and redirect the client to it."""
        blob = self.bucket.blob(f"{self.prefix}/{filename}")
        url = blob.generate_signed_url(expiration=900)
        return RedirectResponse(url)

    def read_bytes(self, filename):
        blob = self.bucket.blob(f"{self.prefix}/{filename}")
        img_bytes = blob.download_as_bytes()
        return img_bytes


class LocalImage(ImageStorage):
    """Dev backend: serves images straight off the local classification-images folder."""

    def __init__(self, folder: str):
        self.folder_path = folder

    def retrieve_image(self, filename):
        """Stream the requested image file back as a JPEG response."""
        return FileResponse(f"{self.folder_path}/{filename}", media_type="image/jpeg")

    def read_bytes(self, filename):
        img_bytes = Path(f"{self.folder_path}/{filename}").read_bytes()
        return img_bytes


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
            try:
                return GCSImage()
            except ResourceNotFoundException as exc:
                # Backend failed to initialise (creds/bucket) — a server
                # availability problem, not a 404. 503 via the base handler.
                raise BaseAppException(
                    "Image storage backend unavailable", status_code=503
                ) from exc
        else:
            return LocalImage(folder=self.folder)


# ── User-contributed photos (write + read back) ─────────────────────────────
# Same env switch as the read-only image storage above, but write-capable. The
# stored `key` is backend-agnostic ({user_id}/{uuid}.jpg); each backend prefixes
# it (GCS contributions/ bucket prefix; local data/test-history/).


class ContributionStorage(ABC):
    """Abstract sink+source for user-submitted observation photos."""

    @abstractmethod
    def upload(self, key: str, data: bytes) -> str: ...

    @abstractmethod
    def retrieve_image(self, key: str): ...


class GCSContribution(ContributionStorage):
    """Cloud backend: writes to / signs URLs for the `contributions/` prefix."""

    def __init__(self):
        self.config = get_config().gcs
        self.client = GCSClient(self.config).get_gcs_client()
        self.bucket = self.client.bucket(self.config.bucket)
        self.prefix = self.config.prefixes["contributions"]

    def upload(self, key: str, data: bytes) -> str:
        blob = self.bucket.blob(f"{self.prefix}/{key}")
        blob.upload_from_string(data, content_type="image/jpeg")
        return key

    def retrieve_image(self, key: str):
        blob = self.bucket.blob(f"{self.prefix}/{key}")
        return RedirectResponse(blob.generate_signed_url(expiration=900))


class LocalContribution(ContributionStorage):
    """Dev backend: writes user photos under data/test-history/ for testing."""

    def __init__(self, folder):
        self.folder_path = Path(folder)

    def upload(self, key: str, data: bytes) -> str:
        dest = self.folder_path / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return key

    def retrieve_image(self, key: str):
        return FileResponse(str(self.folder_path / key), media_type="image/jpeg")


class ContributionConstructor:
    """Picks the contribution backend: GCS on Cloud Run, else local test-history."""

    def __init__(self, folder: str = Path(__file__).parents[1] / "data/test-history"):
        self.folder = folder

    def constructor(self) -> ContributionStorage:
        if os.getenv("K_SERVICE"):
            try:
                return GCSContribution()
            except ResourceNotFoundException as exc:
                raise BaseAppException(
                    "Contribution storage unavailable", status_code=503
                ) from exc
        else:
            return LocalContribution(folder=self.folder)
