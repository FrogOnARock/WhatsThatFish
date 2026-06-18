"""Builds the app_taxa catalogue served to the app.

Joins the classification dataset to taxon names, picks a representative image
per species, and enriches each species with an LLM-generated common name,
description, location, and depth (via Anthropic Claude) before upserting.
"""

import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from sqlalchemy import select, func, desc
from typing import Any
from sqlalchemy.dialects.postgresql import insert

from ..database.models import AppTaxa, InatClassificationDataset, InatTaxa
from sqlalchemy.orm import sessionmaker
from ..serving.schemas import SpeciesInfo
from ..config import _get_logger
from ..retry import llm_retry
import anthropic

logger = _get_logger(__name__)


class BuildAppTaxa:
    """Assembles and LLM-enriches the per-species app_taxa catalogue.

    `run()` fetches one catalogue row per species, fills missing descriptive
    fields via Claude (concurrently), and upserts in chunks — COALESCE-merging
    so a failed enrichment leaves existing good data untouched.
    """

    def __init__(self, session_factory: sessionmaker):

        self.session = session_factory
        # One client, reused across all (threaded) calls — it's thread-safe and
        # pools connections, unlike constructing a fresh client per request.
        self.client = anthropic.Anthropic(api_key=os.getenv("ANT_KEY"))

    def _cte(self):
        """Fetch one catalogue row per species, joined to taxon names + counts.

        Resolves species/genus/family names via per-rank CTEs, left-joins any
        existing app_taxa enrichment, and picks the single highest-UIQM image
        per taxon (DISTINCT ON taxon_id ordered by uiqm desc) as its thumbnail.
        """

        species_stmt = select(InatTaxa.taxon_id, InatTaxa.name).where(
            InatTaxa.rank == "species"
        )
        cte1 = species_stmt.cte("species_stmt")

        genus_stmt = select(InatTaxa.taxon_id, InatTaxa.name).where(
            InatTaxa.rank == "genus"
        )
        cte2 = genus_stmt.cte("genus_stmt")

        family_stmt = select(InatTaxa.taxon_id, InatTaxa.name).where(
            InatTaxa.rank == "family"
        )
        cte3 = family_stmt.cte("family_stmt")

        query = (
            select(
                InatClassificationDataset.taxon_id,
                InatClassificationDataset.zero_indexed_species,
                InatClassificationDataset.filename,
                AppTaxa.description,
                AppTaxa.location,
                AppTaxa.depth,
                AppTaxa.common_name,
                cte1.c.name.label("species"),
                cte2.c.name.label("genus"),
                cte3.c.name.label("family"),
                func.count()
                .over(partition_by=InatClassificationDataset.taxon_id)
                .label("img_count"),
            )
            .join(cte1, InatClassificationDataset.species == cte1.c.taxon_id)
            .join(cte2, InatClassificationDataset.genus == cte2.c.taxon_id)
            .join(cte3, InatClassificationDataset.family == cte3.c.taxon_id)
            .outerjoin(AppTaxa, InatClassificationDataset.taxon_id == AppTaxa.taxon_id)
            .distinct(InatClassificationDataset.taxon_id)
            .where(InatClassificationDataset.zero_indexed_species.isnot(None))
            .order_by(
                InatClassificationDataset.taxon_id, desc(InatClassificationDataset.uiqm)
            )
        )

        with self.session() as session:
            rows = session.execute(query).all()

        logger.info("Retrieved %d catalogue rows for AppTaxa.", len(rows))

        return [
            {
                "taxon_id": row.taxon_id,
                "zero_indexed_species": row.zero_indexed_species,
                "filename": row.filename,
                "species": row.species,
                "genus": row.genus,
                "family": row.family,
                "img_count": row.img_count,
                "description": row.description,
                "location": row.location,
                "depth": row.depth,
                "common_name": row.common_name,
            }
            for row in rows
        ]

    @llm_retry
    def _get_descriptions(self, species: str):
        """Ask Claude for a species' common name, description, location, depth.

        Uses tool-calling with the SpeciesInfo schema to force a structured
        response; returns the four fields as a tuple. Retried on failure.
        """

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=[
                {
                    "name": "species_response_schema",
                    "description": "The schema used for response to this request",
                    "input_schema": SpeciesInfo.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": "species_response_schema"},
            system="You are an expert at marine biology and are providing information on species.",
            messages=[
                {
                    "role": "user",
                    "content": f"Please provide the following items for the following species: {species}",
                }
            ],
        )

        tool_block = next(b for b in response.content if b.type == "tool_use")
        info = SpeciesInfo(**tool_block.input)

        return info.common_name, info.description, info.location, info.depth

    def _upsert(self, taxa: list[dict[str, Any]]):
        """Upsert a batch into app_taxa, keyed on taxon_id.

        Dataset-derived columns are always overwritten; enrichment columns are
        COALESCE-merged so an incoming NULL never clobbers an existing value.
        """

        stmt = insert(AppTaxa).values(taxa)
        stmt = stmt.on_conflict_do_update(
            index_elements=[AppTaxa.taxon_id],
            set_={
                # Refresh group — dataset-derived, always overwrite with latest.
                "zero_indexed_species": stmt.excluded.zero_indexed_species,
                "filename": stmt.excluded.filename,
                "species": stmt.excluded.species,
                "genus": stmt.excluded.genus,
                "family": stmt.excluded.family,
                "img_count": stmt.excluded.img_count,
                # Enrichment group — COALESCE keeps the existing value when the
                # incoming one is NULL (a skipped/failed enrichment), so a bad LLM
                # run can never wipe a good description; a NULL still gets filled.
                "description": func.coalesce(
                    stmt.excluded.description, AppTaxa.description
                ),
                "location": func.coalesce(stmt.excluded.location, AppTaxa.location),
                "depth": func.coalesce(stmt.excluded.depth, AppTaxa.depth),
                "common_name": func.coalesce(
                    stmt.excluded.common_name, AppTaxa.common_name
                ),
            },
        )
        with self.session() as session:
            session.execute(stmt)
            session.commit()
        logger.info("Upserted batch of %d rows into app_taxa.", len(taxa))

    def run(self, batch_size: int = 50, max_workers: int = 8):
        """Build the catalogue: enrich missing rows per chunk, then upsert each.

        Only rows lacking a description are sent to the LLM (threaded up to
        `max_workers`); a failed enrichment is logged and left NULL to retry on
        the next run. Chunked upserts bound how much progress a late failure
        can cost.
        """
        load_dotenv()
        logger.info(
            "Starting AppTaxa build (batch_size=%d, max_workers=%d).",
            batch_size,
            max_workers,
        )
        taxa = self._cte()

        enriched = failed = 0
        # Process in chunks: enrich a chunk's missing rows concurrently, then upsert
        # the whole chunk before moving on. Concurrency gives the speed; chunked
        # upserts keep the durability (a late failure can't lose earlier chunks).
        for start in range(0, len(taxa), batch_size):
            chunk = taxa[start : start + batch_size]
            to_enrich = [r for r in chunk if not r["description"]]

            if to_enrich:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(self._get_descriptions, r["species"]): r
                        for r in to_enrich
                    }
                    for fut in futures:
                        row = futures[fut]
                        try:
                            (
                                row["common_name"],
                                row["description"],
                                row["location"],
                                row["depth"],
                            ) = fut.result()
                            enriched += 1
                        except Exception as e:
                            # Leave enrichment NULL — the null-guard retries it next
                            # run. One bad call must not abort the chunk or the pass.
                            logger.warning(
                                "Enrichment failed for %s (taxon %s); leaving NULL for retry: %s",
                                row["species"],
                                row["taxon_id"],
                                e,
                            )
                            failed += 1

            self._upsert(chunk)

        logger.info(
            "AppTaxa run complete — %d enriched, %d failed (will retry next run).",
            enriched,
            failed,
        )
