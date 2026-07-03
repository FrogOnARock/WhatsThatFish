"""Shared retry decorators for all external service interactions.

Three retry domains:
    db_retry   — Postgres: OperationalError, InterfaceError
    s3_retry   — AWS S3 (boto3): ClientError, ConnectionError, ConnectionClosedError, tarfile.ReadError
    gcs_retry  — GCS (sync google-cloud-storage): ClientError, ServerError
    transfer_retry — async aiohttp + GCS: 429/5xx HTTP errors, connector errors, GCS transient errors
    llm_retry  — Anthropic: RateLimitError (429), APIConnectionError, APITimeoutError,
                 InternalServerError (5xx, incl. 529 overloaded)

Each decorator uses exponential backoff with 5 attempts.
"""

import tarfile

from google.api_core.exceptions import (
    ClientError as GCSClientError,
    ServerError as GCSServerError,
    ServiceUnavailable,
    TooManyRequests,
    GoogleAPICallError,
)
from sqlalchemy.exc import OperationalError, InterfaceError
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Heavy, training/ETL-only deps. Guarded so the slim serving image — which only
# uses db_retry + gcs_retry (sqlalchemy + google-api-core + tenacity, all in the
# serving install) — can import this module without boto3/anthropic/aiohttp. The
# s3/llm/transfer decorators below are only real when these are present; a
# serving process never applies them.
try:
    import aiohttp
    import anthropic
    from botocore.exceptions import (
        ClientError as BotoClientError,
        ConnectionError as BotoConnectionError,
        ConnectionClosedError,
    )

    _TRAIN_DEPS = True
except ImportError:
    _TRAIN_DEPS = False

from .config import _get_logger

logger = _get_logger("retry")


def _log_retry(retry_state):
    """tenacity before-sleep hook — logs which call failed and how long until the next try."""
    logger.warning(
        f"[{retry_state.fn.__qualname__}] "
        f"Attempt {retry_state.attempt_number} failed: "
        f"{retry_state.outcome.exception()}. "
        f"Retrying in {retry_state.next_action.sleep:.1f}s..."
    )


# ── Database (Postgres via SQLAlchemy) ─────────────────────────────────
db_retry = retry(
    retry=(
        retry_if_exception_type(OperationalError)
        | retry_if_exception_type(InterfaceError)
    ),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
)

# ── GCS (sync google-cloud-storage) — serving-safe ────────────────────
gcs_retry = retry(
    retry=(
        retry_if_exception_type(GCSClientError)
        | retry_if_exception_type(GCSServerError)
    ),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
)


# ── training/ETL-only retries (boto3 / anthropic / aiohttp) ────────────
# Real decorators only when the training extras are installed. In the slim
# serving image these are stubs that raise if ever applied — but a serving
# process never imports the ETL/LLM modules that use them, so they stay inert.
if _TRAIN_DEPS:
    # AWS S3 (boto3 sync)
    s3_retry = retry(
        retry=(
            retry_if_exception_type(BotoClientError)
            | retry_if_exception_type(BotoConnectionError)
            | retry_if_exception_type(ConnectionClosedError)
            | retry_if_exception_type(tarfile.ReadError)
        ),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry,
    )

    # Anthropic LLM (sync). InternalServerError covers any 5xx (incl. 529
    # "overloaded"). Permanent 4xx are NOT retried — they won't fix themselves.
    llm_retry = retry(
        retry=(
            retry_if_exception_type(anthropic.RateLimitError)
            | retry_if_exception_type(anthropic.APIConnectionError)
            | retry_if_exception_type(anthropic.APITimeoutError)
            | retry_if_exception_type(anthropic.InternalServerError)
        ),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry,
    )

    # Async transfer (aiohttp + gcloud-aio-storage)
    def _transfer_retry_predicate(exc: BaseException) -> bool:
        """Retry 429/5xx HTTP errors, connector errors, and GCS transient errors."""
        if isinstance(exc, aiohttp.ClientResponseError):
            return exc.status in (429, 500, 502, 503, 504)
        if isinstance(exc, aiohttp.ClientConnectorError):
            return True
        if isinstance(exc, (ServiceUnavailable, TooManyRequests, GoogleAPICallError)):
            return True
        return False

    transfer_retry = retry(
        retry=retry_if_exception(_transfer_retry_predicate),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry,
    )
else:

    def _needs_train_extras(name):
        def _decorator(_fn):
            raise ImportError(
                f"{name}_retry requires the training extras — "
                "`pip install '.[train]'` (boto3 / anthropic / aiohttp are not "
                "in the serving install)."
            )

        return _decorator

    s3_retry = _needs_train_extras("s3")
    llm_retry = _needs_train_extras("llm")
    transfer_retry = _needs_train_extras("transfer")
