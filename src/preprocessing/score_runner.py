import asyncio
import csv
import io
import os
from pathlib import Path
from typing import Type, Any

import aiohttp
import polars as pl
from gcloud.aio.storage import Storage as GCSAsyncStorage, Bucket
from sqlalchemy import select, func, String
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker
from ..database.config import get_session_factory

from ..config import _get_logger, GCSConfig
from ..database.models import InatImageQuality, InatCaptureContext, InatFilteredObservations, LilaImageQuality, \
    SuccessfulUploads, InatClipContext
from ..database.base import Base
from ..retry import transfer_retry, db_retry, gcs_retry
from .capture_context_scorer import ContextScorer
from .uiqm_quality_scorer import QualityScorer

logger = _get_logger(__name__)

_BATCH_SIZE = 10_000

class ScoringProgressTracker:
    """Tracks completed transfers using a WAL + Postgres pattern.

    The WAL (write-ahead log) is a CSV file that receives an append after
    every successful upload. Periodically the WAL is compacted into the
    successful_uploads table in Postgres and the WAL is truncated.

    On startup:
        completed = successful_uploads(source) ∪ WAL replay

    This gives us both crash safety (WAL) and durable storage (Postgres).

    Generic over identifier type — works with photo_id (iNat) or
    file_name (LILA) by configuring source at construction time.
    All identifiers are stored as strings internally.
    """

    def __init__(
        self,
        data_path: str,
        source: str,
        session_factory: sessionmaker,
        dest_table: Type[Base],
        pk: str = "photo_uuid",
        compact_every: int = 1000,
        wal_path: str | None = None,
    ):
        self._data_dir = Path(data_path)
        self._dest_table = dest_table
        self._session_factory = session_factory
        self._pk = pk
        self._wal_path = self._data_dir / (wal_path or f"{source}_transfer_progress_wal.csv")
        self._compact_every = compact_every

        self._completed: set[str] = set()
        self._wal_buffer: list[dict[str, str]] = []
        self._wal_file: io.TextIOWrapper | None = None
        self._wal_writer: csv.DictWriter | None = None
        self._since_last_compact: int = 0

        self._source = source
        logger.debug(f"[{source}] ScoringProgressTracker initialised — WAL: {self._wal_path}, compact_every={compact_every}")

    def _ensure_data_dir(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)


    def load(self) -> set[str]:
        """Load completed transfers from Postgres + replay any WAL entries."""
        self._ensure_data_dir()
        logger.info(f"[{self._source}] Loading progress from DB table '{self._dest_table.__tablename__}'")
        with self._session_factory() as session:
            rows = session.execute(
                select(self._dest_table.__table__.c[self._pk])
            ).scalars().all()
            self._completed = set(rows)
            if self._completed:
                logger.info(f"[{self._source}] Loaded {len(self._completed):,} completed entries from DB")
            else:
                logger.info(f"[{self._source}] No prior entries found in DB — starting fresh")

        wal_count = 0
        if self._wal_path.exists():
            logger.debug(f"[{self._source}] Replaying WAL from {self._wal_path}")
            with open(self._wal_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row:
                        self._completed.add(row.get(self._pk))
                        self._wal_buffer.append(dict(row))
                        wal_count += 1
            if wal_count:
                logger.info(f"[{self._source}] Replayed {wal_count:,} WAL entries")
            else:
                logger.debug(f"[{self._source}] WAL exists but is empty")
        else:
            logger.debug(f"[{self._source}] No WAL file found at {self._wal_path} — will create on first record")

        self._wal_file = open(self._wal_path, "a", newline="")
        self._wal_writer = csv.DictWriter(self._wal_file, self._dest_table.__table__.columns.keys())
        if self._wal_path.stat().st_size == 0:
            self._wal_writer.writeheader()

        logger.info(f"[{self._source}] Total completed (DB + WAL): {len(self._completed):,}")
        return self._completed

    def record(self, row: dict[str, str]) -> None:
        """Record a successful transfer — appends to WAL immediately."""

        ident = row.get(self._pk)
        self._completed.add(ident)
        self._wal_writer.writerow(row)
        self._wal_file.flush()
        self._wal_buffer.append(row)
        self._since_last_compact += 1

        if self._since_last_compact >= self._compact_every:
            self.compact()

    @db_retry
    def compact(self) -> None:
        """Bulk insert WAL entries to Postgres and truncate WAL."""

        if not self._wal_buffer:
            logger.debug(f"[{self._source}] compact() called but WAL buffer is empty — skipping")
            return

        n = len(self._wal_buffer)
        logger.info(f"[{self._source}] Compacting {n:,} WAL entries to DB")

        with self._session_factory() as session:
            if self._source == 'inat_context':
                stmt = insert(InatCaptureContext).values(self._wal_buffer)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[InatCaptureContext.photo_uuid],
                    set_={
                        "mean_r": stmt.excluded.mean_r,
                        "mean_g": stmt.excluded.mean_g,
                        "mean_b": stmt.excluded.mean_b,
                        "is_underwater": stmt.excluded.is_underwater,
                        "stddev": stmt.excluded.stddev
                    }
                )
                session.execute(stmt)
                session.commit()

            elif self._source == 'inat_clip_context':
                stmt = insert(InatClipContext).values(self._wal_buffer)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[InatClipContext.photo_uuid],
                    set_={
                        "is_underwater": stmt.excluded.is_underwater,
                    }
                )
                session.execute(stmt)
                session.commit()


            elif self._source == 'inat_scoring':
                stmt = insert(InatImageQuality).values(self._wal_buffer)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[InatImageQuality.photo_uuid],
                    set_={
                        "uicm": stmt.excluded.uicm,
                        "uism": stmt.excluded.uism,
                        "uiconm": stmt.excluded.uiconm,
                        "uiqm": stmt.excluded.uiqm
                    }
                )
                session.execute(stmt)
                session.commit()

            else:
                stmt = insert(LilaImageQuality).values(self._wal_buffer)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[LilaImageQuality.file_name],
                    set_={
                        "uicm": stmt.excluded.uicm,
                        "uism": stmt.excluded.uism,
                        "uiconm": stmt.excluded.uiconm,
                        "uiqm": stmt.excluded.uiqm
                    }
                )
                session.execute(stmt)
                session.commit()

        self._wal_buffer.clear()

        # Truncate the WAL
        self._wal_file.close()
        self._wal_file = open(self._wal_path, "w", newline="")
        self._wal_writer = csv.DictWriter(self._wal_file, self._dest_table.__table__.columns.keys())
        self._wal_writer.writeheader()
        self._since_last_compact = 0

        logger.info(f"[{self._source}] Compact complete — {n:,} entries flushed, WAL truncated, total scored: {len(self._completed):,}")

    def close(self) -> None:
        """Final compaction and cleanup."""
        logger.info(f"[{self._source}] Closing tracker — running final compaction")
        self.compact()
        if self._wal_file:
            self._wal_file.close()
        logger.info(f"[{self._source}] Tracker closed. Total scored this session: {len(self._completed):,}")

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def is_completed(self, identifier: str) -> bool:
        return str(identifier) in self._completed



class ContextRunner:
    def __init__(self, gcs_config: GCSConfig,
                 session: sessionmaker,
                 progress_tracker: ScoringProgressTracker,
                 dataset: str,
                 concurrency: int = 50):

        self._dataset = dataset
        self._gcs_config = gcs_config
        self._session_factory = session
        self._progress_tracker = progress_tracker
        self._semaphore = asyncio.Semaphore(concurrency)
        self.context_scorer = ContextScorer()
        self.quality_scorer = QualityScorer()
        logger.debug(f"[ContextRunner/{dataset}] Initialised with concurrency={concurrency}")

    async def _get_blob(self, path: str, file_name: str, gcs_storage: GCSAsyncStorage) -> bytes:
        return await gcs_storage.download(self._gcs_config.bucket, f"{path}/{file_name}", timeout=30)

    @transfer_retry
    async def _context_with_tracking(self, row: dict[str, str], gcs_storage: GCSAsyncStorage):
        async with self._semaphore:

            identifier = row.get("filename")
            path = self._gcs_config.prefixes.get("gcs_train")

            blob = await self._get_blob(path, identifier, gcs_storage)
            result = self.context_scorer.score_capture_context(blob)
            if isinstance(result, ValueError):
                logger.warning(f"[ContextRunner] Skipping {identifier} — scorer returned: {result}")
                return
            mean_r, mean_g, mean_b, stddev, classification = result

            context_dict = {
                "photo_uuid": row.get("photo_uuid"),
                "mean_r": float(mean_r),
                "mean_g": float(mean_g),
                "mean_b": float(mean_b),
                "stddev": float(stddev),
                "is_underwater": int(classification)
            }

            self._progress_tracker.record(context_dict)

    @db_retry
    def _select_all_uploads(self):
        logger.debug("[ContextRunner] Querying SuccessfulUploads for source='inat'")
        with self._session_factory() as session:
            ids = session.execute(
                select(SuccessfulUploads.identifier).where(SuccessfulUploads.source == "inat")
            ).scalars().all()
            logger.info(f"[ContextRunner] Found {len(ids):,} uploaded iNat photos in SuccessfulUploads")
            return set(ids)

    @db_retry
    def _select_files(self, photo_ids) -> list[dict[str, Any]]:
        logger.debug(f"[ContextRunner] Fetching filenames for {len(photo_ids):,} photo IDs")
        with self._session_factory() as session:
            rows = session.execute(
                select(InatFilteredObservations.photo_uuid, InatFilteredObservations.photo_id, func.concat(
                    InatFilteredObservations.photo_id, ".", InatFilteredObservations.extension
                ).label("filename")
               ).where(func.cast(InatFilteredObservations.photo_id, String).in_(photo_ids))
            ).all()

            result = [{"photo_uuid": r.photo_uuid, "filename": r.filename} for r in rows]
            logger.info(f"[ContextRunner] Retrieved {len(result):,} file rows from InatFilteredObservations")
            return result

    async def run(self):
        logger.info(f"[ContextRunner/{self._dataset}] Starting context scoring run")

        files = self._select_all_uploads()
        ids = set(self._progress_tracker.load())

        rows = self._select_files(files)
        rows = [r for r in rows if r.get("photo_uuid") not in ids]

        logger.info(f"[ContextRunner/{self._dataset}] {len(files):,} uploaded | {len(ids):,} already done | {len(rows):,} to score")

        if not rows:
            logger.info(f"[ContextRunner/{self._dataset}] Nothing to score — exiting")
            return

        total_batches = (len(rows) + _BATCH_SIZE - 1) // _BATCH_SIZE
        logger.info(f"[ContextRunner/{self._dataset}] {len(rows):,} rows → {total_batches:,} batches of {_BATCH_SIZE:,}")

        try:
            for batch_idx, start in enumerate(range(0, len(rows), _BATCH_SIZE), 1):
                batch = rows[start:start + _BATCH_SIZE]
                logger.info(f"[ContextRunner/{self._dataset}] Batch {batch_idx}/{total_batches} — {len(batch):,} rows")

                async with GCSAsyncStorage(service_file=os.environ.get("GCS_SECRET")) as gcs_storage:
                    context = [
                        self._context_with_tracking(row, gcs_storage)
                        for row in batch
                    ] if self._dataset == "inat" else []

                    results = await asyncio.gather(*context, return_exceptions=True)
                    errors = [r for r in results if isinstance(r, BaseException)]
                    if errors:
                        logger.warning(f"[ContextRunner/{self._dataset}] Batch {batch_idx}: {len(errors):,}/{len(results):,} tasks failed — first: {type(errors[0]).__name__}: {errors[0]}")

                self._progress_tracker.compact()
                logger.info(f"[ContextRunner/{self._dataset}] Batch {batch_idx}/{total_batches} complete — compacted")

            logger.info(f"[ContextRunner/{self._dataset}] All context scoring batches complete")

        finally:
            self._progress_tracker.close()



class ScoreRunner:
    def __init__(self, gcs_config: GCSConfig,
                 session: sessionmaker,
                 progress_tracker: ScoringProgressTracker,
                 dataset: str,
                 concurrency: int = 50):

        self._dataset = dataset
        self._gcs_config = gcs_config
        self._session_factory = session
        self._progress_tracker = progress_tracker
        self._semaphore = asyncio.Semaphore(concurrency)
        self.context_scorer = ContextScorer()
        self.quality_scorer = QualityScorer()
        logger.debug(f"[ScoreRunner/{dataset}] Initialised with concurrency={concurrency}")

    async def _get_blob(self, path: str, file_name: str, gcs_storage: GCSAsyncStorage) -> bytes:
        return await gcs_storage.download(self._gcs_config.bucket, f"{path}/{file_name}", timeout=30)

    def _get_blob_details(self, row) -> ValueError | tuple[str, str]:

        if self._dataset == 'inat':
            identifier = row.get("filename")
            path = self._gcs_config.prefixes.get("gcs_train")
            return identifier, path

        elif self._dataset == 'lila':
            identifier = row.get("filename")
            path = self._gcs_config.prefixes.get("gcs_object_detection")
            return identifier, path

        else:
            return ValueError("Dataset must be defined.")

    @transfer_retry
    async def _scoring_with_tracking(self, row: dict[str, str], gcs_storage: GCSAsyncStorage):
        async with self._semaphore:

            identifier, path = self._get_blob_details(row)
            blob = await self._get_blob(path, identifier, gcs_storage)
            uicm, uism, uiconm, uiqm = self.quality_scorer.compute_uiqm(blob)

            if self._dataset == "inat":
                score_dict = {
                    "photo_uuid": row.get("photo_uuid"),
                    "uicm": float(uicm),
                    "uism": float(uism),
                    "uiconm": float(uiconm),
                    "uiqm": float(uiqm)
                }
            else:
                score_dict = {
                    "file_name": row.get("filename"),
                    "uicm": float(uicm),
                    "uism": float(uism),
                    "uiconm": float(uiconm),
                    "uiqm": float(uiqm)
                }

            self._progress_tracker.record(score_dict)

    @db_retry
    def _select_files(self, files) -> ValueError | list[dict[str, Any]]:
        logger.debug(f"[ScoreRunner/inat] Fetching file rows for {len(files):,} photo UUIDs")
        with self._session_factory() as session:
            rows = session.execute(
                select(InatFilteredObservations.photo_uuid, InatFilteredObservations.photo_id, func.concat(
                    InatFilteredObservations.photo_id, ".", InatFilteredObservations.extension
                ).label("filename")
                       ).where(InatFilteredObservations.photo_uuid.in_(files))
            ).all()

            result = [{"photo_uuid": r.photo_uuid, "filename": r.filename, "photo_id": r.photo_id} for r in rows]
            logger.info(f"[ScoreRunner/inat] Retrieved {len(result):,} file rows from InatFilteredObservations")
            return result

    def _select_all_uploads(self, dataset: str) -> set[str]:
        with self._session_factory() as session:
            if dataset == "inat":
                logger.debug("[ScoreRunner/inat] Querying InatCaptureContext for underwater images (is_underwater in [1, 2])")
                ids = session.execute(
                    select(InatClipContext.photo_uuid).where(InatClipContext.is_underwater == 1)
                ).scalars().all()
            else:
                logger.debug(f"[ScoreRunner/{dataset}] Querying SuccessfulUploads for source='{dataset}'")
                ids = session.execute(
                    select(SuccessfulUploads.identifier).where(SuccessfulUploads.source == dataset)
                ).scalars().all()
            logger.info(f"[ScoreRunner/{dataset}] Found {len(ids):,} candidates to score")
            return set(ids)

    async def run(self):
        logger.info(f"[ScoreRunner/{self._dataset}] Starting UIQM scoring run")

        files = self._select_all_uploads(self._dataset)
        ids = set(self._progress_tracker.load())

        if self._dataset == "inat":
            rows = self._select_files(files)
            rows = [r for r in rows if r.get("photo_uuid") not in ids]
            logger.info(f"[ScoreRunner/inat] {len(files):,} uploaded | {len(ids):,} already scored | {len(rows):,} to score")
        else:
            files_to_process = files - ids
            rows = [{"filename": file} for file in files_to_process]
            logger.info(f"[ScoreRunner/lila] {len(files):,} uploaded | {len(ids):,} already scored | {len(rows):,} to score")

        if not rows:
            logger.info(f"[ScoreRunner/{self._dataset}] Nothing to score — exiting")
            self._progress_tracker.close()
            return

        total_batches = (len(rows) + _BATCH_SIZE - 1) // _BATCH_SIZE
        logger.info(f"[ScoreRunner/{self._dataset}] {len(rows):,} rows → {total_batches:,} batches of {_BATCH_SIZE:,}")

        try:
            for batch_idx, start in enumerate(range(0, len(rows), _BATCH_SIZE), 1):
                batch = rows[start:start + _BATCH_SIZE]
                logger.info(f"[ScoreRunner/{self._dataset}] Batch {batch_idx}/{total_batches} — {len(batch):,} rows")

                async with GCSAsyncStorage(service_file=os.environ.get("GCS_SECRET")) as gcs_storage:
                    scoring = [
                        self._scoring_with_tracking(row, gcs_storage)
                        for row in batch
                    ]
                    results = await asyncio.gather(*scoring, return_exceptions=True)
                    errors = [r for r in results if isinstance(r, BaseException)]
                    if errors:
                        logger.warning(f"[ScoreRunner/{self._dataset}] Batch {batch_idx}: {len(errors):,}/{len(results):,} tasks failed — first: {type(errors[0]).__name__}: {errors[0]}")

                self._progress_tracker.compact()
                logger.info(f"[ScoreRunner/{self._dataset}] Batch {batch_idx}/{total_batches} complete — compacted")

            logger.info(f"[ScoreRunner/{self._dataset}] All UIQM scoring batches complete")

        finally:
            self._progress_tracker.close()
