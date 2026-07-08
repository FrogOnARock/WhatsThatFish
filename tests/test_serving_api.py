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

from whatsthatfish.database.models import AppTaxa

# `client` + `session_factory` come from conftest (the patched-app TestClient).


# ─── /predict upload size cap (DoS guard) ─────────────────────────────────────


class TestPredictUploadLimit:
    """`/predict` is unauthenticated, so the 15 MB upload cap is the guard against
    a single request amplifying memory / disk-spool / CPU-inference cost. The cap
    must reject an oversized upload BEFORE inference runs."""

    def _stub_service(self, serving_app, service):
        # Bypass the real ONNX-backed service (never built — the `client` fixture
        # skips the model-loading lifespan) and inject a stub whose behaviour lets
        # us prove whether inference was reached.
        from whatsthatfish.serving.dependencies import get_prediction_service

        serving_app.dependency_overrides[get_prediction_service] = lambda: service

    def test_oversized_upload_rejected_before_inference(
        self, client, serving_app, monkeypatch
    ):
        from whatsthatfish.serving.routers import predictions

        # Shrink the cap so the test needn't allocate 15 MB of payload.
        monkeypatch.setattr(predictions, "MAX_UPLOAD_BYTES", 10)

        class _NoRunService:
            def get_prediction(self, *_a, **_k):
                raise AssertionError("inference ran despite an oversized upload")

        self._stub_service(serving_app, _NoRunService())

        resp = client.post(
            "/predict",
            files={"img": ("big.jpg", b"12345678901234567890", "image/jpeg")},  # 20 B
        )

        assert resp.status_code == 422
        assert "limit" in resp.text.lower()
        # The rejection body echoes the offending size, computed from the part.
        assert resp.json()["body"]["size"] == 20

    def test_within_limit_reaches_inference(self, client, serving_app):
        from whatsthatfish.serving.schemas import Prediction

        sentinel = Prediction(bbox=[], species=[], genus=[], family=[], detected=False)

        class _OkService:
            def get_prediction(self, *_a, **_k):
                return sentinel

        self._stub_service(serving_app, _OkService())

        resp = client.post(
            "/predict",
            files={"img": ("small.jpg", b"tiny-bytes", "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["detected"] is False


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

    # Enrichment-null contract — DECIDED: option (c), coerce NULL → ""/[]/0.
    # A trained species (has zero_indexed_species + core taxonomy) can exist in
    # app_taxa before the LLM enrichment pass fills common_name/description/
    # location/depth/filename. It must stay VISIBLE in the catalogue with blank
    # enrichment — not 500 (Pydantic), not omitted. Keeps the SPA's non-null
    # SpeciesEntry contract stable.
    def test_unenriched_species_contract(self, client, session_factory):
        _seed_app_taxa(
            session_factory,
            [
                _row(
                    common_name=None,
                    description=None,
                    location=None,
                    depth=None,
                    filename=None,
                    img_count=None,
                )
            ],
        )
        resp = client.get("/species")
        assert resp.status_code == 200  # not a 500
        body = resp.json()
        assert body["total"] == 1  # not omitted
        entry = body["species"][0]
        # Core taxonomy survives; enrichment is coerced to blanks.
        assert entry["name"] == "Carcharhinus melanopterus"
        assert entry["common_name"] == ""
        assert entry["description"] == ""
        assert entry["location"] == []
        assert entry["depth"] == ""
        assert entry["filename"] == ""
        assert entry["image_count"] == 0
