import argparse
from enum import Enum
from pathlib import Path
from sqlalchemy import inspect
import asyncio

from whatsthatfish.preprocessing.app_taxa import BuildAppTaxa
from .clip_context import ClipModel
from .score_runner import ScoreRunner, ContextRunner
from .annotation_conversion import AnnotationConverter
from .inat_zero_index import ZeroIndexClassification
from .score_runner import ScoringProgressTracker
from .prepare_inat import InatPreparation
from dataclasses import dataclass
from ..database.models import (
    InatCaptureContext,
    InatImageQuality,
    LilaImageQuality,
    InatClipContext,
)
from ..database import get_session_factory
from ..config import get_config, _get_logger
from ..etl.gcs_client import GCSClient
from dotenv import load_dotenv

load_dotenv()

logger = _get_logger(__name__)


@dataclass
class Dataset(str, Enum):
    ALL = "all"
    SCORING = "scoring"
    ANN_CONV = "annotation"
    CONTEXT_ORIG = "context_orig"
    CONTEXT_CLIP = "context_clip"
    CLASS_PREP = "classification_preparation"
    APP_TAXA = "app_taxa"


@dataclass
class SourceDataset(str, Enum):
    INAT = "inat"
    LILA = "lila"
    BOTH = "both"


class PreProcessingFactory:
    def __init__(
        self, type: Dataset, source_dataset: SourceDataset = SourceDataset.BOTH
    ):

        self.data_path = Path(__file__).parents[1] / "data" / "preprocessing"
        self.session_factory = get_session_factory()
        self.config = get_config()
        self.gcs_config = self.config.gcs
        self.gcs_client = GCSClient(self.gcs_config)
        self.type = type
        self.source_dataset = source_dataset
        logger.info(
            f"PreProcessingFactory initialised — type={type}, source_dataset={source_dataset}, data_path={self.data_path}"
        )

    def _dest_table(self, dataset: str, runner: str):
        if dataset == "lila":
            return LilaImageQuality
        if dataset == "inat":
            return (
                InatImageQuality
                if runner == "scoring"
                else InatCaptureContext
                if runner == "context_orig"
                else InatClipContext
            )
        raise ValueError(f"Unknown dataset: {dataset!r}")

    def _load_score_runner(self, dataset: str, source: str):

        dest_table = self._dest_table(dataset, runner="scoring")
        logger.info(
            f"[Factory] Performing scoring for {dataset}, loading to {dest_table.__name__}"
        )

        tracker = ScoringProgressTracker(
            data_path=str(self.data_path),
            source=source,
            session_factory=self.session_factory,
            dest_table=dest_table,
            pk=str(inspect(dest_table).mapper.primary_key[0].name),
        )

        return ScoreRunner(
            gcs_config=self.gcs_config,
            session=self.session_factory,
            progress_tracker=tracker,
            dataset=dataset,
        )

    def _load_context_runner(self, dataset: str, source: str):

        dest_table = self._dest_table(dataset, runner="context_orig")
        logger.info(
            f"[Factory] Performing context for {dataset}, loading to {dest_table}"
        )

        tracker = ScoringProgressTracker(
            data_path=str(self.data_path),
            source=source,
            session_factory=self.session_factory,
            dest_table=dest_table,
            pk=str(inspect(dest_table).mapper.primary_key[0].name),
        )

        return ContextRunner(
            gcs_config=self.gcs_config,
            session=self.session_factory,
            progress_tracker=tracker,
            dataset=dataset,
        )

    def _load_clip_context_runner(self, dataset: str, source: str):

        dest_table = self._dest_table(dataset, runner="context_clip")
        logger.info(
            f"[Factory] Performing CLIP context for {dataset}, loading to {dest_table.__name__}"
        )
        tracker = ScoringProgressTracker(
            data_path=str(self.data_path),
            source=source,
            session_factory=self.session_factory,
            dest_table=dest_table,
            pk=str(inspect(dest_table).mapper.primary_key[0].name),
        )

        return ClipModel(
            gcs_config=self.gcs_config,
            session_factory=self.session_factory,
            progress_tracker=tracker,
        )

    def _load_annotation_runner(self):
        return AnnotationConverter(session_factory=self.session_factory)

    def _load_ds_runner(self):
        return InatPreparation(session_factory=self.session_factory)

    def _load_zero_index_runner(self):
        return ZeroIndexClassification(session_factory=self.session_factory)

    def _load_app_taxa_runner(self):
        return BuildAppTaxa(session_factory=self.session_factory)

    async def run(self):
        logger.info(f"PreProcessingFactory.run() starting — type={self.type}")

        if self.type == "all":
            logger.info("Step 1/7: iNat capture context scoring")
            await self._load_clip_context_runner(
                dataset="inat", source="inat_clip_context"
            ).run()

            logger.info("Step 2/7: iNat UIQM scoring")
            await self._load_score_runner(dataset="inat", source="inat_scoring").run()

            logger.info("Step 3/7: LILA UIQM scoring")
            await self._load_score_runner(dataset="lila", source="lila_scoring").run()

            logger.info("Step 4/7: Annotation conversion")
            self._load_annotation_runner().run()

            logger.info("Step 5/7: Classification DS Preparation")
            self._load_ds_runner().run()

            logger.info("Step 6/7: Zero Indexed Classification")
            self._load_zero_index_runner().run()

            logger.info("Step 7/7: Retrieving App Meta Data")
            self._load_app_taxa_runner().run()

        elif self.type == "scoring":
            if self.source_dataset in (SourceDataset.INAT, SourceDataset.BOTH):
                logger.info("Step 1/2: iNat UIQM scoring")
                await self._load_score_runner(
                    dataset="inat", source="inat_scoring"
                ).run()

            if self.source_dataset in (SourceDataset.LILA, SourceDataset.BOTH):
                logger.info("Step 2/2: LILA UIQM scoring")
                await self._load_score_runner(
                    dataset="lila", source="lila_scoring"
                ).run()

        elif self.type == "context_clip":
            logger.info("Step 1/1: iNat CLIP context scoring")
            await self._load_clip_context_runner(
                dataset="inat", source="inat_clip_context"
            ).run()

        elif self.type == "annotation":
            logger.info("Step 1/1: Annotation conversion")
            self._load_annotation_runner().run()

        elif self.type == "context_orig":
            logger.info("Step 1/1: iNat capture context scoring")
            await self._load_context_runner(dataset="inat", source="inat_context").run()

        elif self.type == "classification_preparation":
            logger.info("Step 1/2: Classification DS Preparation")
            self._load_ds_runner().run()

            logger.info("Step 2/2: Zero Indexed Classification")
            self._load_zero_index_runner().run()

        elif self.type == "app_taxa":
            logger.info("Step 1/1: Retrieving App Meta Data")
            self._load_app_taxa_runner().run()

        else:
            raise ValueError(f"Unknown pipeline type: {self.type!r}")

        logger.info(f"PreProcessingFactory.run() complete — type={self.type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline",
        epilog=(
            "Examples:\n"
            "  Score only LILA images:          --type scoring --dataset lila\n"
            "  Score only iNat images:          --type scoring --dataset inat\n"
            "  Run CLIP context (iNat only):    --type context_clip\n"
            "  Full pipeline (both datasets):   --type all\n"
            "  Build classification dataset:    --type classification_preparation\n"
            "  Build App Taxonomy:              --type app_taxa\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--type",
        choices=[d.value for d in Dataset],
        default=Dataset.ALL.value,
        metavar="TYPE",
        help=(
            f"Pipeline step to run. Choices: {', '.join(d.value for d in Dataset)}. "
            "Default: all"
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=[d.value for d in SourceDataset],
        default=SourceDataset.BOTH.value,
        metavar="DATASET",
        help=(
            f"Source dataset to process. Choices: {', '.join(d.value for d in SourceDataset)}. "
            "Only applies to scoring runs; ignored for context, annotation, and classification_preparation steps. "
            "Default: both"
        ),
    )
    args = parser.parse_args()
    asyncio.run(
        PreProcessingFactory(
            type=Dataset(args.type),
            source_dataset=SourceDataset(args.dataset),
        ).run()
    )
