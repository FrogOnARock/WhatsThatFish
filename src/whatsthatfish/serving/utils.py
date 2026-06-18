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

    @abstractmethod
    def retrieve_image(self, filename):
        ...


class GCSImage(ImageStorage):

    def __init__(self):
        self.config = get_config().gcs
        self.client = GCSClient(self.config).get_gcs_client()
        self.bucket = self.client.bucket(self.config.bucket)
        self.prefix = self.config.prefixes["training"]

    def retrieve_image(self, filename: str):
        blob = self.bucket.blob(f"{self.prefix}/{filename}")
        url = blob.generate_signed_url(expiration=900)
        return RedirectResponse(url)


class LocalImage(ImageStorage):

    def __init__(self, folder: str):
        self.folder_path = folder

    def retrieve_image(self, filename):
        return FileResponse(f"{self.folder_path}/{filename}", media_type="image/jpeg")

class StorageConstructor:

    def __init__(self, folder: str = Path(__file__).parents[1] / "data/classification_images"):
        self.folder = folder

    def constructor(self):
        if os.getenv("K_SERVICE"):
            return GCSImage()
        else:
            return LocalImage(folder=self.folder)

