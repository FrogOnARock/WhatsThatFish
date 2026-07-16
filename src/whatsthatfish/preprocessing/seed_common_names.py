"""Seed common names for the untrained iNat species via the Batch API.

`app_taxa` carries curated names for the ~1,500 trained species; the remaining
~37K in-scope species (Actinopterygii + Chondrichthyes) have
`inat_taxa.common_name` NULL. This backfills them cheaply: Claude Haiku over the
**Batch API** (50% off), ~25 species per request, structured JSON output, upsert
guarded to `common_name IS NULL` so an existing name is never clobbered.
Idempotent; re-run for a null-retry pass. Once seeded, the correction picker
(`TaxaRepository.search_species`) matches common names across the full taxonomy.

CLI: python -m whatsthatfish.preprocessing.seed_common_names [--limit N] [--dry-run]
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
from sqlalchemy import select, update, or_
from sqlalchemy.orm import sessionmaker

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from ..database import get_session_factory
from ..database.models import InatTaxa
from ..config import _get_logger

logger = _get_logger(__name__)

_MODEL = "claude-haiku-4-5"  # simple factual lookup — cheapest tier suffices
_CHUNK = 25  # species per request; amortizes the fixed per-request overhead
_FISH_ANCESTORS = ("%/47178/%", "%/196614/%")  # Actinopterygii, Chondrichthyes
_SYSTEM = (
    "You are an expert marine biologist providing the widely-used English "
    "common name for fish species."
)
# Batch API supports structured outputs on Haiku 4.5. Wrap the array in an
# object (top-level arrays aren't a valid json_schema root).
_SCHEMA = {
    "type": "object",
    "properties": {
        "species": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "taxon_id": {"type": "integer"},
                    "common_name": {"type": "string"},
                },
                "required": ["taxon_id", "common_name"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["species"],
    "additionalProperties": False,
}


class SeedCommonNames:
    """Batch-API common-name backfill for `inat_taxa`."""

    def __init__(self, session_factory: sessionmaker):
        self.session = session_factory
        self.client = anthropic.Anthropic(api_key=os.getenv("ANT_KEY"))

    def _pending(self, limit: int | None = None):
        """In-scope species rows still missing a common name."""
        stmt = (
            select(InatTaxa.taxon_id, InatTaxa.name)
            .where(
                InatTaxa.rank == "species",
                InatTaxa.active.is_(True),
                InatTaxa.common_name.is_(None),
                or_(*[InatTaxa.ancestry.like(p) for p in _FISH_ANCESTORS]),
            )
            .order_by(InatTaxa.taxon_id)
        )
        if limit:
            stmt = stmt.limit(limit)
        with self.session() as s:
            return s.execute(stmt).all()

    def _request(self, idx: int, chunk) -> Request:
        """Build one batch request for a chunk of ≤25 species."""
        listing = "\n".join(f"{r.taxon_id}: {r.name}" for r in chunk)
        return Request(
            custom_id=f"chunk-{idx}",
            params=MessageCreateParamsNonStreaming(
                model=_MODEL,
                max_tokens=1024,
                system=_SYSTEM,
                output_config={
                    "format": {"type": "json_schema", "schema": _SCHEMA}
                },
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Return the common English name for each species below. "
                            "Echo each taxon_id exactly. If a species has no "
                            "established common name, use its scientific name.\n\n"
                            + listing
                        ),
                    }
                ],
            ),
        )

    def _submit_and_wait(self, requests: list[Request], poll_s: int = 30):
        batch = self.client.messages.batches.create(requests=requests)
        logger.info(
            "Submitted batch %s (%d requests); polling every %ds…",
            batch.id,
            len(requests),
            poll_s,
        )
        while batch.processing_status != "ended":
            time.sleep(poll_s)
            batch = self.client.messages.batches.retrieve(batch.id)
        return batch

    def _collect(self, batch) -> dict[int, str]:
        """Parse succeeded results → {taxon_id: common_name}. Results arrive in
        any order and the model may reorder items, so we key on the echoed
        taxon_id, never on position."""
        names: dict[int, str] = {}
        for result in self.client.messages.batches.results(batch.id):
            if result.result.type != "succeeded":
                logger.warning(
                    "batch item %s: %s", result.custom_id, result.result.type
                )
                continue
            text = next(
                (b.text for b in result.result.message.content if b.type == "text"),
                None,
            )
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("unparseable output for %s", result.custom_id)
                continue
            for item in data.get("species", []):
                tid, cn = item.get("taxon_id"), (item.get("common_name") or "").strip()
                if tid is not None and cn:
                    names[int(tid)] = cn
        return names

    def _upsert(self, names: dict[int, str]) -> None:
        """Fill only rows still NULL — never clobber an existing common name."""
        if not names:
            return
        with self.session() as s:
            for tid, cn in names.items():
                s.execute(
                    update(InatTaxa)
                    .where(
                        InatTaxa.taxon_id == tid, InatTaxa.common_name.is_(None)
                    )
                    .values(common_name=cn)
                )
            s.commit()
        logger.info("Upserted %d common names.", len(names))

    def run(self, limit: int | None = None, dry_run: bool = False) -> None:
        load_dotenv()
        rows = self._pending(limit)
        n_req = (len(rows) + _CHUNK - 1) // _CHUNK
        logger.info(
            "%d species pending → %d batch requests (Haiku, ~$%.2f est.).",
            len(rows),
            n_req,
            len(rows) * 0.00006,  # rough: input+output per species, batch-discounted
        )
        if dry_run or not rows:
            return
        requests = [
            self._request(i // _CHUNK, rows[i : i + _CHUNK])
            for i in range(0, len(rows), _CHUNK)
        ]
        batch = self._submit_and_wait(requests)
        names = self._collect(batch)
        self._upsert(names)
        logger.info("Done — filled %d of %d pending.", len(names), len(rows))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Seed inat_taxa.common_name for untrained species via Batch API"
    )
    p.add_argument("--limit", type=int, default=None, help="cap species (test slice)")
    p.add_argument(
        "--dry-run", action="store_true", help="count + cost only, no API call"
    )
    args = p.parse_args()
    SeedCommonNames(get_session_factory()).run(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
