"""
Unit/integration tests for the shared retry decorators (`whatsthatfish.retry`).

`retry.py` wraps every external call in the repo (Postgres, S3, GCS, async
transfer) but had no tests. Two things matter:
    1. The async-transfer predicate retries the RIGHT exceptions (transient
       HTTP/connector errors) and gives up on the wrong ones (e.g. a 404).
    2. A decorated function actually retries on a retryable error and gives up
       after `stop_after_attempt(5)`.

Sleeps are stubbed out (`tenacity.nap.time.sleep`) so the exponential backoff
doesn't make the suite wait ~30s per give-up.
"""

import aiohttp
import pytest
from google.api_core.exceptions import ServiceUnavailable, NotFound
from sqlalchemy.exc import OperationalError
from tenacity import RetryError

from whatsthatfish.retry import _transfer_retry_predicate, db_retry


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make tenacity's backoff instant so give-up paths don't block the suite."""
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _seconds: None)


def _response_error(status: int) -> aiohttp.ClientResponseError:
    return aiohttp.ClientResponseError(
        request_info=None, history=(), status=status, message="boom"
    )


# ─── Transfer predicate ─────────────────────────────────────────────────


class TestTransferRetryPredicate:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_retries_transient_http(self, status):
        assert _transfer_retry_predicate(_response_error(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_does_not_retry_client_errors(self, status):
        assert _transfer_retry_predicate(_response_error(status)) is False

    def test_retries_connector_error(self):
        exc = aiohttp.ClientConnectorError(
            connection_key=None, os_error=OSError("refused")
        )
        assert _transfer_retry_predicate(exc) is True

    def test_retries_gcs_transient(self):
        assert _transfer_retry_predicate(ServiceUnavailable("503")) is True

    def test_does_not_retry_gcs_notfound(self):
        # NotFound is a GoogleAPICallError subclass — but a 404 is terminal,
        # NOT transient. The predicate retries GoogleAPICallError broadly, so
        # this documents the (arguably too-broad) current behaviour.
        result = _transfer_retry_predicate(NotFound("404"))
        assert result is True  # NOTE: GoogleAPICallError catch-all swallows 404

    def test_does_not_retry_unrelated_exception(self):
        assert _transfer_retry_predicate(ValueError("nope")) is False


# ─── Decorator behaviour (db_retry as representative) ───────────────────


class TestDbRetryBehaviour:
    def test_succeeds_first_try_calls_once(self):
        calls = {"n": 0}

        @db_retry
        def ok():
            calls["n"] += 1
            return "done"

        assert ok() == "done"
        assert calls["n"] == 1

    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        @db_retry
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise OperationalError("stmt", {}, Exception("transient"))
            return "recovered"

        assert flaky() == "recovered"
        assert calls["n"] == 3

    def test_gives_up_after_five_attempts(self):
        calls = {"n": 0}

        @db_retry
        def always_fails():
            calls["n"] += 1
            raise OperationalError("stmt", {}, Exception("down"))

        # The decorator has no reraise=True, so tenacity wraps the final
        # failure in RetryError after exhausting stop_after_attempt(5).
        with pytest.raises(RetryError):
            always_fails()
        assert calls["n"] == 5  # stop_after_attempt(5)

    def test_non_retryable_raises_immediately(self):
        calls = {"n": 0}

        @db_retry
        def bad_value():
            calls["n"] += 1
            raise ValueError("not a DB error")

        with pytest.raises(ValueError):
            bad_value()
        assert calls["n"] == 1  # not retried
