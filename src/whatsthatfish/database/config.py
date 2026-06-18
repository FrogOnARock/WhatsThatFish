"""Engine and session-factory builders for the Postgres backend.

Both a sync (psycopg2) and an async (asyncpg) path are offered off the same
`DATABASE_URL`, so callers pick whichever fits their execution model.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


def get_database_url(async_: bool = False) -> str:
    """Read DATABASE_URL from the environment, swapping in the asyncpg driver when async.

    Raises if it's unset rather than silently connecting to the wrong place.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Expected format: postgresql://user:pass@host:port/dbname"
        )
    if async_:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def get_engine(echo: bool = False):
    """Build a sync (psycopg2) engine; `echo=True` logs emitted SQL."""
    return create_engine(get_database_url(), echo=echo)


def get_async_engine(echo: bool = False):
    """Build an async (asyncpg) engine for the asyncio ingestion pipelines."""
    return create_async_engine(get_database_url(async_=True), echo=echo)


def get_session_factory(engine=None):
    """A sync `sessionmaker`, defaulting to a fresh engine if none is supplied."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine)


def get_async_session_factory(engine=None):
    """An AsyncSession factory; `expire_on_commit=False` keeps loaded attrs usable post-commit."""
    if engine is None:
        engine = get_async_engine()
    return sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
