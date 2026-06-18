"""Top-level ETL orchestration — the one entry point for building datasets.

`DataFactory` wires the iNaturalist, LILA, and photo-transfer stages together
and runs whichever subset the chosen `Dataset` selects.
"""

import asyncio
from enum import Enum
from pathlib import Path

from ..config import get_config, _get_logger
from ..database.config import get_session_factory
from .gcs_client import GCSClient
from .inaturalist_dataset import INaturalistDataset
from .download_lila import LilaDataset
from .photo_transfer import PhotoTransferPipeline


class Dataset(str, Enum):
    """Which pipeline(s) to run: everything, the iNat classification set, or LILA detection."""

    ALL = "all"
    CLASSIFICATION = "classification"
    DETECTION = "detection"


class DataFactory:
    """Builds and runs the ingestion stages off a single shared config/session/GCS setup."""

    def __init__(self):
        self.config = get_config()
        self.data_path = Path(__file__).parents[1] / "data" / "etl"
        self.logger = _get_logger("DataFactory")
        self.session_factory = get_session_factory()
        self.gcs = GCSClient(self.config.gcs)

    def _build_inat(self) -> INaturalistDataset:
        """Construct the iNaturalist S3-ingestion stage (taxa/observations/photos → Postgres)."""
        return INaturalistDataset(
            config=self.config.s3,
            session_factory=self.session_factory,
            data_path=str(self.data_path),
        )

    def _build_lila(self) -> LilaDataset:
        """Construct the LILA detection-dataset stage (sample, download, GCS upload)."""
        return LilaDataset(
            gcs=self.gcs,
            data_path=str(self.data_path),
            gcs_config=self.config.gcs,
            session_factory=self.session_factory,
        )

    def _build_photo_transfer(self) -> PhotoTransferPipeline:
        """Construct the async S3→GCS classification-photo transfer stage."""
        return PhotoTransferPipeline(
            gcs_config=self.config.gcs,
            s3_config=self.config.s3,
            data_path=str(self.data_path),
            session_factory=self.session_factory,
        )

    def run(self, dataset: Dataset = Dataset.ALL, taxa: list[int] | None = None):
        """Run the etl pipeline for the specified dataset(s).

        Args:
            dataset: Which pipeline(s) to run — all, classification (iNat), or detection (LILA)
            taxa: Override taxon IDs for iNat filtering (default: Actinopterygii + Chondrichthyes)
        """
        self.logger.info(f"Starting etl pipeline: {dataset.value}")

        if dataset in (Dataset.ALL, Dataset.CLASSIFICATION):
            self.logger.info("Running iNaturalist classification pipeline")
            inat = self._build_inat()
            inat.run(taxa=taxa)

        if dataset in (Dataset.ALL, Dataset.DETECTION):
            self.logger.info("Running LILA detection pipeline")
            lila = self._build_lila()
            lila.extract_lila_images()

        if dataset in (Dataset.ALL, Dataset.CLASSIFICATION):
            self.logger.info("Running photo transfer pipeline")
            photo_transfer = self._build_photo_transfer()
            asyncio.run(photo_transfer.run())

        self.logger.info("Data pipeline complete")


def main():
    """CLI entry point — runs the classification pipeline by default."""
    factory = DataFactory()
    factory.run(dataset=Dataset.CLASSIFICATION)


if __name__ == "__main__":
    main()
