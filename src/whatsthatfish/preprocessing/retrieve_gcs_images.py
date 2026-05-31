import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from gcloud.aio.storage import Storage as GCSAsyncStorage
from sqlalchemy import select

from whatsthatfish.database import InatObjDetectionDataset
from ..database.config import get_session_factory
from ..database.models import InatClassificationDataset, LilaYolo

BUCKET = "whats-that-fish"
_BATCH_SIZE = 500


def _session():
    return get_session_factory()()


def _get_inat_od_blobs() -> list[tuple[str, str]]:
    """Returns (blob_path, filename) pairs for inat classification images."""
    with _session() as session:
        rows = session.execute(select(InatObjDetectionDataset.filename)).all()
    return [("training//" + str(row.filename), str(row.filename)) for row in rows]


def _get_inat_classification_blobs() -> list[tuple[str, str]]:
    """Returns (blob_path, filename) pairs for inat classification images."""
    with _session() as session:
        rows = session.execute(select(InatClassificationDataset.filename)).all()
    return [("training//" + str(row.filename), str(row.filename)) for row in rows]


def _get_lila_blobs() -> list[tuple[str, str]]:
    """Returns (blob_path, filename) pairs for lila detection images."""
    with _session() as session:
        rows = session.execute(select(LilaYolo.file_name)).all()
    return [("object_detection//" + str(row.file_name), str(row.file_name)) for row in rows]


async def _download_one(
    blob_path: str,
    filename: str,
    dest_dir: Path,
    gcs_storage: GCSAsyncStorage,
    semaphore: asyncio.Semaphore,
) -> Exception | None:
    async with semaphore:
        try:
            data = await gcs_storage.download(BUCKET, blob_path, timeout=30)
            (dest_dir / filename).write_bytes(data)
            return None
        except Exception as exc:
            return exc


async def _download_dataset(
    blobs: list[tuple[str, str]],
    dest_dir: Path,
    concurrency: int,
    label: str,
):
    dest_dir.mkdir(parents=True, exist_ok=True)
    already = {p.name for p in dest_dir.iterdir()}
    blobs = [(path, name) for path, name in blobs if name not in already]

    print(f"{label}: {len(already):,} already present, {len(blobs):,} to download")
    if not blobs:
        return

    semaphore = asyncio.Semaphore(concurrency)
    failed = 0
    total_batches = (len(blobs) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(blobs), _BATCH_SIZE), 1):
        batch = blobs[start : start + _BATCH_SIZE]
        async with GCSAsyncStorage(service_file=os.environ.get("GCS_SECRET")) as gcs:
            results = await asyncio.gather(
                *[_download_one(path, name, dest_dir, gcs, semaphore) for path, name in batch],
                return_exceptions=True,
            )
        batch_errors = [r for r in results if r is not None]
        failed += len(batch_errors)
        if batch_errors:
            print(f"  batch {batch_idx}/{total_batches}: {len(batch_errors)} failed — first: {batch_errors[0]}")
        else:
            print(f"  batch {batch_idx}/{total_batches}: {len(batch)} ok")

    succeeded = len(blobs) - failed
    print(f"{label}: done — {succeeded:,} succeeded, {failed:,} failed")

#TODO understand why this is erroring on OD images.
async def _retrieve_images_async(
    inat_dir: Path,
    inat_od_dir: Path,
    lila_dir: Path,
    concurrency: int = 50,
    dataset: str = "all",
):
    candidates = [
        (_get_inat_classification_blobs(), inat_dir, "inat_classification"),
        (_get_inat_od_blobs(), inat_od_dir, "inat_od"),
        (_get_lila_blobs(), lila_dir, "lila"),
    ]
    active = [(blobs, dest, label) for blobs, dest, label in candidates
              if dataset == "all" or dataset == label]
    for blobs, dest, label in active:
        await _download_dataset(blobs, dest, concurrency, label)


def retrieve_images(
    inat_dir: str = Path(__file__).parents[1]  / "data/classification_images",
    inat_od_dir: str = Path(__file__).parents[1]  / "data/inat_od_images",
    lila_dir: str = Path(__file__).parents[1]  / "data/od_images",
    concurrency: int = 50,
    dataset: str = "all",
):
    load_dotenv()
    asyncio.run(_retrieve_images_async(Path(inat_dir), Path(inat_od_dir), Path(lila_dir), concurrency, dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download GCS images to local directories",
        epilog=(
            "Examples:\n"
            "  Download iNat classification only:    --dataset inat_classification\n"
            "  Download LILA to classification dir:  --dataset lila --lila-dir data/classification_images\n"
            "  Download all (default):               --dataset all\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["inat_classification", "inat_od", "lila", "all"],
        default="all",
        metavar="DATASET",
        help="Which dataset to download. Choices: inat_classification, inat_od, lila, all. Default: all",
    )
    parser.add_argument(
        "--inat-dir",
        default="src/data/classification_images",
        metavar="DIR",
        help="Local destination for iNat images. Default: /data/classification_images",
    )
    parser.add_argument(
        "--lila-dir",
        default="src/data/od_images",
        metavar="DIR",
        help="Local destination for LILA images. Default: /data/od_images",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        metavar="N",
        help="Max concurrent GCS downloads. Default: 50",
    )
    args = parser.parse_args()
    retrieve_images(
        inat_dir=args.inat_dir,
        lila_dir=args.lila_dir,
        concurrency=args.concurrency,
        dataset=args.dataset,
    )
