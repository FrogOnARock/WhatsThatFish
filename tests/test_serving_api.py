"""
Integration tests for the FastAPI serving layer (`whatsthatfish.serving.app`).

These exercise the real HTTP stack via Starlette's TestClient against the real
test Postgres (same fixture stack as test_integration.py). They guard the
serving-layer CONTRACT — the JSON shape the SPA deserialises — which nothing
else in the suite covers.

Prerequisites:
    docker compose -f docker-compose.test.yml up -d
    cd whatsthatfish && .venv/bin/python -m pytest tests/test_serving_api.py -v

Why patch `_session_factory`?
    `app.py` binds `_session_factory = get_session_factory()` at import time,
    pointing at the PROD engine. We rebind that module global to the test
    factory so endpoints read from the throwaway test DB.
"""

import pytest
from fastapi.testclient import TestClient

from whatsthatfish.serving import app as app_module
from whatsthatfish.database.models import AppTaxa


@pytest.fixture
def client(session_factory):
    """A TestClient whose endpoints read from the test Postgres.

    `session_factory` (from conftest) creates tables and truncates after each
    test, so every test starts from an empty `app_taxa`.
    """
    original = app_module._session_factory
    app_module._session_factory = session_factory
    try:
        yield TestClient(app_module.app)
    finally:
        app_module._session_factory = original


def _seed_app_taxa(session_factory, rows: list[dict]):
    with session_factory() as session:
        session.add_all([AppTaxa(**row) for row in rows])
        session.commit()


def _row(**overrides) -> dict:
    """A fully-populated AppTaxa row; override only what a test cares about."""
    base = dict(
        taxon_id=4001,
        zero_indexed_species=0,
        species="Carcharhinus melanopterus",
        genus="Carcharhinus",
        family="Carcharhinidae",
        description="A reef shark.",
        common_name="Blacktip reef shark",
        location=["Indo-Pacific"],
        depth="1-5 meters, 0-15 feet",
        filename="4001.jpg",
        img_count=287,
    )
    base.update(overrides)
    return base


# ─── /health ──────────────────────────────────────────────────────────


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ─── /species ─────────────────────────────────────────────────────────


class TestSpeciesCatalogue:
    def test_empty_catalogue(self, client):
        """No app_taxa rows → empty list, total 0 (not a 500)."""
        resp = client.get("/species")
        assert resp.status_code == 200
        body = resp.json()
        assert body["species"] == []
        assert body["total"] == 0

    def test_catalogue_shape_and_mapping(self, client, session_factory):
        """One row round-trips into the SpeciesEntry contract verbatim."""
        _seed_app_taxa(session_factory, [_row()])

        resp = client.get("/species")
        assert resp.status_code == 200
        body = resp.json()

        assert body["total"] == 1
        entry = body["species"][0]
        # Field-name mapping is the fragile part (AppTaxa column → SpeciesEntry
        # field): species_id←zero_indexed_species, image_count←img_count.
        assert entry["species_id"] == 0
        assert entry["name"] == "Carcharhinus melanopterus"
        assert entry["genus"] == "Carcharhinus"
        assert entry["family"] == "Carcharhinidae"
        assert entry["image_count"] == 287
        assert entry["common_name"] == "Blacktip reef shark"
        assert entry["location"] == ["Indo-Pacific"]
        assert entry["depth"] == "1-5 meters, 0-15 feet"
        assert entry["filename"] == "4001.jpg"

    def test_total_matches_row_count(self, client, session_factory):
        _seed_app_taxa(
            session_factory,
            [
                _row(taxon_id=4001, zero_indexed_species=0),
                _row(
                    taxon_id=3001, zero_indexed_species=1, species="Thalassoma lunare"
                ),
                _row(
                    taxon_id=3002,
                    zero_indexed_species=2,
                    species="Amphiprion ocellaris",
                ),
            ],
        )
        body = client.get("/species").json()
        assert body["total"] == 3
        assert len(body["species"]) == 3

    # ────────────────────────────────────────────────────────────────
    # TODO(you): enrichment-null contract — YOUR design decision.
    #
    # AppTaxa's enrichment columns (description, common_name, location,
    # depth, filename) are nullable — a row can exist BEFORE the LLM
    # enrichment pass fills them. But SpeciesEntry currently types
    # `common_name: str`, `location: list[str]`, `depth: str` as NON-optional.
    #
    # So: what should /species do for a row whose enrichment is still NULL?
    #   (a) 500 (Pydantic validation error) — current behaviour, probably wrong
    #   (b) omit un-enriched species from the catalogue
    #   (c) coerce NULL → "" / [] in the query or schema
    #
    # Seed a row with description=None etc., decide the contract, then assert
    # it here. This is the call I shouldn't make for you — it shapes what the
    # frontend has to handle.
    # ────────────────────────────────────────────────────────────────
    def test_unenriched_species_contract(self, client, session_factory):
        pytest.skip("TODO(you): decide + assert the NULL-enrichment contract")
