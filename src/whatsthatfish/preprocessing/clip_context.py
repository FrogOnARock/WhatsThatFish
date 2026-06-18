"""CLIP-based underwater-vs-above-water context classifier for iNat photos.

Uses CLIP ViT-B/32 zero-shot: each image's embedding is compared against a
prototype for "above water" and one for "underwater" (each averaged over a set
of prompts). Coral-reef prompts are included so coral frames classify as
underwater. The argmax (0=above, 1=underwater) is written to inat_clip_context.
"""

import os
import asyncio
import torch
import clip
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from typing import Any

from gcloud.aio.storage import Storage as GCSAsyncStorage
from sqlalchemy.orm import sessionmaker

from ..config import _get_logger, GCSConfig
from .score_runner import ScoringProgressTracker
from ..database import InatFilteredObservations, SuccessfulUploads
from sqlalchemy import select, func, String
from ..retry import db_retry

logger = _get_logger(__name__)

_BATCH_SIZE = 10_000


class ClipModel:
    """Async CLIP zero-shot underwater classifier over GCS-stored iNat photos.

    Downloads images concurrently from GCS, scores each against the two text
    prototypes, and records the predicted label via a WAL-backed progress
    tracker so the (large) run is crash-safe and resumable.
    """

    def __init__(
        self,
        progress_tracker: ScoringProgressTracker,
        gcs_config: GCSConfig,
        session_factory: sessionmaker,
        concurrency: int = 50,
    ):

        self._session_factory = session_factory
        self._gcs_config = gcs_config
        self._progress_tracker = progress_tracker
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._semaphore = asyncio.Semaphore(concurrency)
        logger.info(f"[ClipModel] Loading ViT-B/32 on device={self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        logger.info(f"[ClipModel] Model loaded — concurrency={concurrency}")
        self.above_water_prompts = [
            "A fish being held in someone's hand",
            "An aquarium with a fish inside of it",
            "A person with a fish above water",
            "A person on a dock with a caught fish",
            "A beached fish",
            "Fish on the shore",
            "A fish at a fish market",
            "A fish on a cutting board",
            "A fish in a bucket of water",
            "A fish flopping on the deck of a boat",
            "A fish in a net above the water surface",
            "A dead fish on ice",
            "A fish being released back into the water",
            "A fish jumping out of the water",
            "A fish in a shallow tide pool seen from above",
            "A photograph taken from a boat",
        ]

        self.underwater_prompts = [
            "Fish in shallow water and good light",
            "A deep sea fish swimming by",
            "A well lit photo of a fish in shallow water",
            "A very colourful fish",
            "An animal underwater along the ocean floor",
            "A school of fish swimming underwater",
            "A fish hiding in coral",
            "A murky underwater photo of a fish",
            "A fish camouflaged against rocks underwater",
            "A fish swimming near the surface underwater",
            "An underwater photo of a fish in a river",
            "A fish near the sandy ocean bottom",
            "A diver photographing a fish underwater",
            "An underwater photograph",
            "An underwater coral reef scene",
            "Tropical fish among coral formations",
            "A vibrant coral reef with marine life",
            "A reef scene in clear blue ocean water",
            "An underwater photograph of a coral garden",
        ]

    def _tokenize_text(self):
        """Encode the prompt sets into two normalized class prototype vectors.

        Each class's prompt embeddings are averaged then re-normalized, giving
        one "above water" and one "underwater" anchor stacked for comparison.
        """
        above_water_text = clip.tokenize(self.above_water_prompts).to(self.device)
        under_water_text = clip.tokenize(self.underwater_prompts).to(self.device)

        with torch.no_grad():
            above_features = self.model.encode_text(above_water_text)
            underwater_features = self.model.encode_text(under_water_text)

        above_features = F.normalize(above_features, dim=-1).mean(dim=0)
        above_features = F.normalize(above_features, dim=0)

        underwater_features = F.normalize(underwater_features, dim=-1).mean(dim=0)
        underwater_features = F.normalize(underwater_features, dim=0)

        return torch.stack([above_features, underwater_features])

    def _encode_image(self, image):
        image = F.normalize(self.model.encode_image(image), dim=-1)
        return image

    def _similarity_score(self, image, text):
        """Softmax over cosine similarity to the two prototypes (above/under)."""
        similarity_scores = (image @ text.T).softmax(dim=-1)
        return similarity_scores

    async def _get_blob(
        self, path: str, file_name: str, gcs_storage: GCSAsyncStorage
    ) -> bytes:
        return await gcs_storage.download(
            self._gcs_config.bucket, f"{path}/{file_name}", timeout=30
        )

    async def _inference(self, row: dict[str, Any], text, gcs_storage):
        """Download one photo, classify it, and record the predicted label.

        Semaphore-gated so only `concurrency` downloads/encodes run at once.
        """
        async with self._semaphore:
            photo_uuid = row.get("photo_uuid")
            identifier = row.get("filename")
            path = self._gcs_config.prefixes.get("gcs_train")

            blob = await self._get_blob(path, identifier, gcs_storage)
            tensor = (
                self.preprocess(Image.open(BytesIO(blob))).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                encoded = self._encode_image(tensor)

            scores = self._similarity_score(encoded, text)
            pred = torch.argmax(scores).item()
            logger.debug(
                f"[ClipModel] {identifier} → pred={pred} scores={scores.tolist()}"
            )

            record = {"photo_uuid": photo_uuid, "is_underwater": pred}
            self._progress_tracker.record(record)

    @db_retry
    def _select_all_uploads(self):
        """All iNat photo identifiers known to be uploaded to GCS (candidates)."""
        logger.debug("[ClipModel] Querying SuccessfulUploads for source='inat'")
        with self._session_factory() as session:
            ids = (
                session.execute(
                    select(SuccessfulUploads.identifier).where(
                        SuccessfulUploads.source == "inat"
                    )
                )
                .scalars()
                .all()
            )
            logger.info(
                f"[ClipModel] Found {len(ids):,} uploaded iNat photos in SuccessfulUploads"
            )
            return set(ids)

    @db_retry
    def _select_files(self, photo_ids) -> list[dict[str, Any]]:
        """Resolve photo IDs to {photo_uuid, filename} rows for GCS download."""
        logger.debug(f"[ClipModel] Fetching filenames for {len(photo_ids):,} photo IDs")
        with self._session_factory() as session:
            rows = session.execute(
                select(
                    InatFilteredObservations.photo_uuid,
                    InatFilteredObservations.photo_id,
                    func.concat(
                        InatFilteredObservations.photo_id,
                        ".",
                        InatFilteredObservations.extension,
                    ).label("filename"),
                ).where(
                    func.cast(InatFilteredObservations.photo_id, String).in_(photo_ids)
                )
            ).all()

            result = [
                {"photo_uuid": r.photo_uuid, "filename": r.filename} for r in rows
            ]
            logger.info(
                f"[ClipModel] Retrieved {len(result):,} file rows from InatFilteredObservations"
            )
            return result

    async def run(self):
        """Score every not-yet-done iNat photo in resumable, compacted batches.

        Skips photos already recorded by the tracker, encodes the prompts once,
        then classifies in batches of `_BATCH_SIZE` with a fresh GCS client per
        batch, compacting the WAL after each.
        """
        logger.info("[ClipModel] Starting CLIP context scoring run")

        files = self._select_all_uploads()
        ids = set(self._progress_tracker.load())

        rows = self._select_files(files)
        rows = [r for r in rows if r.get("photo_uuid") not in ids]

        logger.info(
            f"[ClipModel] {len(files):,} uploaded | {len(ids):,} already done | {len(rows):,} to score"
        )

        if not rows:
            logger.info("[ClipModel] Nothing to score — exiting")
            self._progress_tracker.close()
            return

        logger.info("[ClipModel] Encoding text prompts")
        text = self._tokenize_text()

        total_batches = (len(rows) + _BATCH_SIZE - 1) // _BATCH_SIZE
        logger.info(
            f"[ClipModel] {len(rows):,} rows → {total_batches:,} batches of {_BATCH_SIZE:,}"
        )

        try:
            for batch_idx, start in enumerate(range(0, len(rows), _BATCH_SIZE), 1):
                batch = rows[start : start + _BATCH_SIZE]
                logger.info(
                    f"[ClipModel] Batch {batch_idx}/{total_batches} — {len(batch):,} rows"
                )

                async with GCSAsyncStorage(
                    service_file=os.environ.get("GCS_SECRET")
                ) as gcs_storage:
                    tasks = [self._inference(row, text, gcs_storage) for row in batch]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    errors = [r for r in results if isinstance(r, BaseException)]
                    if errors:
                        logger.warning(
                            f"[ClipModel] Batch {batch_idx}: {len(errors):,}/{len(results):,} failed — first: {type(errors[0]).__name__}: {errors[0]}"
                        )

                self._progress_tracker.compact()
                logger.info(
                    f"[ClipModel] Batch {batch_idx}/{total_batches} complete — compacted"
                )

            logger.info("[ClipModel] All CLIP scoring batches complete")

        finally:
            self._progress_tracker.close()
