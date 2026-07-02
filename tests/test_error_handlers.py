"""Error layer tests (P2) — exception class status codes + the FastAPI handlers
that turn them into JSON the SPA can read ({detail} / {detail, body}).
"""

import uuid

from whatsthatfish.serving.error import (
    BaseAppException,
    ResourceNotFoundException,
    ValidationException,
    AuthenticationException,
    InvalidPredictionResponse,
    InvalidPredictionRequest,
)


class TestExceptionClasses:
    def test_status_codes(self):
        assert ResourceNotFoundException("x").status_code == 404
        assert ValidationException("x").status_code == 400
        assert AuthenticationException().status_code == 401
        assert InvalidPredictionResponse("x").status_code == 500
        assert BaseAppException("x").status_code == 500
        assert BaseAppException("x", status_code=503).status_code == 503

    def test_message_is_carried(self):
        assert ResourceNotFoundException("missing thing").message == "missing thing"
        assert AuthenticationException().message == "Not authenticated"

    def test_invalid_prediction_request_carries_body(self):
        exc = InvalidPredictionRequest(message="bad", body={"reason": "empty_body"})
        assert exc.status_code == 422
        assert exc.body == {"reason": "empty_body"}


class TestHandlerIntegration:
    def test_base_exception_handler_returns_detail_json(self, authed_client):
        """A ResourceNotFoundException (404) flows through the base handler →
        {"detail": <message>} with the right status."""
        resp = authed_client.get(f"/observation_photos/{uuid.uuid4()}/image")
        assert resp.status_code == 404
        body = resp.json()
        assert set(body.keys()) == {"detail"}
        assert isinstance(body["detail"], str)

    def test_authentication_exception_returns_401_detail(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401
        assert "detail" in resp.json()
