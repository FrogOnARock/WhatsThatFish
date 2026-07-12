from abc import ABC, abstractmethod
import os
from pathlib import Path
from fastapi.responses import FileResponse
from starlette.responses import Response


from ..etl.gcs_client import GCSClient
from ..config import get_config
from .error import ResourceNotFoundException, BaseAppException

# Cap uploads well under Cloud Run's 32 MB request limit. Bounds the memory /
# disk-spool / inference cost of a single request — matters most on /predict,
# which is unauthenticated, but applied to every upload path.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


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
        # Key is "gcs_train" (value "training/"). NB: objects were uploaded under a
        # double slash (training//<file>), so keeping the trailing slash + the
        # f"{prefix}/{filename}" join below reproduces the stored path. Re-keying
        # the bucket to single-slash is a separate cleanup.
        self.prefix = self.config.prefixes["gcs_train"]

    def retrieve_image(self, filename: str):
        """Stream the blob's bytes back as JPEG. No signed URL: ADC in Cloud Run
        can't sign a URL (needs a private key or IAM SignBlob), and streaming these
        small crops straight through the API is fine at this volume."""
        blob = self.bucket.blob(f"{self.prefix}/{filename}")
        # Training images are immutable at a given filename — let the browser cache
        # them forever so a re-view is 0 network. `public` (shared caches OK) since
        # the catalogue is not user-private.
        return Response(
            content=blob.download_as_bytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    def read_bytes(self, filename):
        blob = self.bucket.blob(f"{self.prefix}/{filename}")
        img_bytes = blob.download_as_bytes()
        return img_bytes


class LocalImage(ImageStorage):
    """Dev backend: serves images straight off the local classification-images folder."""

    def __init__(self, folder: str):
        self.folder_path = Path(folder)

    def _resolve(self, filename: str) -> Path:
        """Collapse `filename` to its base name before joining, so a crafted
        `../` can't escape the images folder. Catalogue filenames are flat, so
        this never loses legitimate structure. Defense-in-depth: routing already
        blocks encoded slashes and prod serves GCS object keys, not files."""
        return self.folder_path / Path(filename).name

    def retrieve_image(self, filename):
        """Stream the requested image file back as a JPEG response."""
        return FileResponse(str(self._resolve(filename)), media_type="image/jpeg")

    def read_bytes(self, filename):
        return self._resolve(filename).read_bytes()


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

    @abstractmethod
    def read_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove the blob. Best-effort: an already-missing object is NOT an
        error (the DB row is the source of truth for what should exist, so a
        delete must still succeed if the blob was cleaned up out-of-band)."""
        ...


class GCSContribution(ContributionStorage):
    """Cloud backend: writes to / signs URLs for the `contributions/` prefix."""

    def __init__(self):
        self.config = get_config().gcs
        self.client = GCSClient(self.config).get_gcs_client()
        self.bucket = self.client.bucket(self.config.bucket)
        self.prefix = self.config.prefixes["gcs_contributions"]

    def upload(self, key: str, data: bytes) -> str:
        blob = self.bucket.blob(f"{self.prefix}/{key}")
        blob.upload_from_string(data, content_type="image/jpeg")
        return key

    def retrieve_image(self, key: str):
        blob = self.bucket.blob(f"{self.prefix}/{key}")
        # Same immutability, but `private`: these are user-submitted photos, so keep
        # them out of shared/proxy caches — only the owner's browser should cache.
        return Response(
            content=blob.download_as_bytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    def read_bytes(self, key: str) -> bytes:
        """Raw JPEG bytes for server-side re-inference (no HTTP round-trip)."""
        return self.bucket.blob(f"{self.prefix}/{key}").download_as_bytes()

    def delete(self, key: str) -> None:
        # `if_generation_match` unset → unconditional delete. Swallow a missing
        # object so deleting a row whose blob is already gone still succeeds.
        from google.cloud.exceptions import NotFound

        try:
            self.bucket.blob(f"{self.prefix}/{key}").delete()
        except NotFound:
            pass


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

    def read_bytes(self, key: str) -> bytes:
        return (self.folder_path / key).read_bytes()

    def delete(self, key: str) -> None:
        # missing_ok: an already-absent file is not an error (see abstract doc).
        (self.folder_path / key).unlink(missing_ok=True)


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
