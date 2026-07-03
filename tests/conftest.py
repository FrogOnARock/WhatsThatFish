"""
Shared fixtures for integration tests.

Requires a running test Postgres instance:
    docker compose -f docker-compose.test.yml up -d

The session_factory fixture creates all tables fresh for each test,
then rolls back via truncation after each test for isolation.
"""

from pathlib import Path

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from whatsthatfish.database.base import Base

TEST_DATABASE_URL = "postgresql://test:test@localhost:5433/wtf_test"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Verified Google-token claims for the test user. The serving auth seam is
# bypassed in route tests (get_current_user is overridden), but we still funnel
# through UserService.get_or_create so first-login user creation runs for real.
TEST_CLAIMS = {
    "sub": "test-google-subject-001",
    "email": "diver@test.dev",
    "name": "Test Diver",
    "picture": "https://example.com/avatar.png",
}
# A fish ancestry path (…/Actinopterygii 47178/…) so search_species' LIKE filter
# matches seeded taxa.
_FISH_ANCESTRY = "48460/1/2/355675/47178/85497"


@pytest.fixture(scope="session")
def engine():
    """Create a single engine for the entire test session.

    Creates all tables on first use. The engine is shared across tests
    for efficiency — isolation is handled per-test via truncation.
    """
    eng = create_engine(TEST_DATABASE_URL, echo=False)

    # Verify connection before running any tests
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        pytest.skip(
            f"Test Postgres not available at {TEST_DATABASE_URL}. "
            f"Run: docker compose -f docker-compose.test.yml up -d\n"
            f"Error: {e}"
        )

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture
def session_factory(engine):
    """Provide a session factory pointing at the test DB.

    After each test, truncates all tables to ensure isolation.
    TRUNCATE CASCADE handles FK ordering automatically.
    """
    factory = sessionmaker(bind=engine)
    yield factory

    # Teardown: truncate all tables for clean slate
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f'TRUNCATE TABLE "{table.name}" CASCADE'))
        conn.commit()


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory containing parquet files."""
    return FIXTURES_DIR


@pytest.fixture
def tracker_dir(tmp_path) -> Path:
    """Writable, per-test directory for WAL progress files (the crash-recovery
    tracker's `data_path`). tmp_path is unique per test, so WAL state never
    bleeds between runs."""
    return tmp_path


# ─── Serving-layer fixtures ───────────────────────────────────────────────────
# Shared by the Phase 3 API tests (repository/service/route). The serving layer
# binds `_session_factory = get_session_factory()` in dependencies.py at import
# time (PROD engine); we rebind that module global to the test factory so every
# endpoint reads the throwaway test DB.


@pytest.fixture
def serving_app(session_factory):
    """The FastAPI app with its session factory + dependency overrides repointed
    at the test DB. Yields the app; clears overrides on teardown."""
    from whatsthatfish.serving import dependencies as deps
    from whatsthatfish.serving.main import app

    original = deps._session_factory
    deps._session_factory = session_factory
    try:
        yield app
    finally:
        deps._session_factory = original
        app.dependency_overrides.clear()


@pytest.fixture
def client(serving_app):
    """Unauthenticated TestClient — for /health, /species, and 401 checks.
    (No `with` block, so the model-loading lifespan never runs.)"""
    return TestClient(serving_app)


@pytest.fixture
def authed_client(serving_app, tmp_path):
    """TestClient with the auth gate + photo storage overridden.

    get_current_user is resolved IN-REQUEST (mirroring production: same session
    as the endpoint, so the user is persistent and settings PATCH commits stick).
    Photos write to a tmp dir instead of data/test-history."""
    from whatsthatfish.serving.dependencies import (
        get_current_user,
        get_session,
        get_observation_service,
    )
    from whatsthatfish.serving.services.service import UserService, ObservationService
    from whatsthatfish.serving.utils import LocalContribution

    def _current_user(session=Depends(get_session)):
        return UserService(session).get_or_create(TEST_CLAIMS)

    def _obs_service(session=Depends(get_session)):
        return ObservationService(
            session=session, storage=LocalContribution(folder=tmp_path)
        )

    serving_app.dependency_overrides[get_current_user] = _current_user
    serving_app.dependency_overrides[get_observation_service] = _obs_service
    return TestClient(serving_app)


@pytest.fixture
def make_user(session_factory):
    """Factory: persist a User (defaults to TEST_CLAIMS) and return it. For
    direct repository/service tests that need a known user_id."""
    from whatsthatfish.serving.services.service import UserService

    def _make(claims: dict | None = None):
        with session_factory() as s:
            user = UserService(s).get_or_create(claims or TEST_CLAIMS)
            s.refresh(user)
            s.expunge(user)
            return user

    return _make


@pytest.fixture
def seed_taxa(session_factory):
    """Insert paired inat_taxa (FK target for observations) + app_taxa
    (zero-index ↔ taxon_id translation + display) rows.

    `specs`: list of dicts, each needs taxon_id, zero_index, species, genus,
    family; common_name/ancestry/etc. optional. Returns the taxon_ids.
    """
    from whatsthatfish.database.models import InatTaxa, AppTaxa

    def _seed(specs: list[dict]) -> list[int]:
        with session_factory() as s:
            for spec in specs:
                s.add(
                    InatTaxa(
                        taxon_id=spec["taxon_id"],
                        name=spec["species"],
                        rank=spec.get("rank", "species"),
                        active=spec.get("active", True),
                        ancestry=spec.get("ancestry", _FISH_ANCESTRY),
                    )
                )
                s.add(
                    AppTaxa(
                        taxon_id=spec["taxon_id"],
                        zero_indexed_species=spec["zero_index"],
                        zero_indexed_genus=spec.get("zero_genus", spec["zero_index"]),
                        zero_indexed_family=spec.get("zero_family", spec["zero_index"]),
                        species=spec["species"],
                        genus=spec["genus"],
                        family=spec["family"],
                        common_name=spec.get("common_name"),
                        location=spec.get("location"),
                        depth=spec.get("depth"),
                        filename=spec.get("filename", f"{spec['taxon_id']}.jpg"),
                        img_count=spec.get("img_count", 100),
                    )
                )
            s.commit()
        return [spec["taxon_id"] for spec in specs]

    return _seed
